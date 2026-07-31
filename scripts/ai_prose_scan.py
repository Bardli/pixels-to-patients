"""Scan the site's prose for AI-writing tells.

Deterministic detector, no LLM. Reports one line per hit so a rewrite is driven
by evidence rather than by eyeballing. Scores are hit-counts per 1000 words, so
long pages are not penalised for being long.

Only prose is scanned. Anything inside <pre>, <code>, <script>, <style> is
stripped first: those carry verbatim source that must not be reworded, and a
false hit there would invite exactly the wrong edit.

Usage:  python3 scripts/ai_prose_scan.py [path ...]
Exit 1 if any P0 (unambiguous tell) survives, so this can gate a commit.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

# (severity, label, regex). P0 = unambiguous AI tell, P1 = usually wrong,
# P2 = flag only when it clusters.
PATTERNS: list[tuple[str, str, str]] = [
    ("P0", "em-dash emphasis", r"\s—\s"),
    ("P0", "filler opener", r"\b(?:It is important to note|It is worth noting|"
                            r"In order to|Due to the fact that|It should be noted)\b"),
    ("P0", "additive adverb opener", r"(?:^|[.!?]\s|>)\s*(?:Moreover|Furthermore|"
                                     r"Additionally|Notably|Importantly)\b"),
    ("P0", "AI vocabulary", r"\b(?:delve|delves|delving|tapestry|crucial|crucially|"
                            r"pivotal|landscape|realm|seamless|seamlessly|myriad|"
                            r"plethora|testament|underscore|underscores|foster|fosters|"
                            r"harness(?:es|ing)?|utilize|utilizes|utilizing|"
                            r"comprehensive|holistic|multifaceted|intricate|nuanced)\b"),
    ("P0", "negative parallelism", r"\b(?:not just|not only|isn't just|isn't only)\b"
                                   r"[^.!?]{0,60}\b(?:but|it's|it is)\b"),
    ("P0", "engagement hook", r"\b(?:Let's|Let us)\s+(?:explore|dive|delve|break|unpack|"
                              r"take a look)\b|\bHere's the thing\b|\bThe catch\?"),
    ("P1", "copula avoidance", r"\b(?:serves as|serve as|stands as|stands for|"
                               r"boasts|showcases|showcase)\b"),
    ("P1", "intensifier", r"\b(?:actually|really|truly|simply|genuinely|quite|very)\b"),
    ("P1", "promotional adjective", r"\b(?:powerful|elegant|beautiful|vibrant|"
                                    r"remarkable|striking|impressive|rich)\b"),
    ("P1", "vague attribution", r"\b(?:Experts|Researchers|Observers|Many)\s+"
                                r"(?:believe|note|agree|suggest|argue)\b"),
    ("P1", "self-congratulation", r"\bfrom scratch\b|\bhands-on\b|\bcarefully\s+"
                                  r"(?:crafted|designed|chosen)\b"),
    ("P1", "superficial -ing analysis", r",\s+(?:highlighting|ensuring|demonstrating|"
                                        r"showcasing|reflecting|underscoring|"
                                        r"emphasizing|illustrating)\b"),
    ("P2", "hedge stack", r"\b(?:may|might|could)\s+(?:potentially|possibly|arguably)\b"),
    ("P2", "generic closer", r"\b(?:the future looks|only time will tell|"
                             r"a step in the right direction)\b"),
]

STRIP = re.compile(r"<(pre|code|script|style)\b.*?</\1>", re.S | re.I)
# <b id="chan-zero">—</b> is a slot the page fills from exported data at
# runtime, not a dash the author typed. Neutralise it before extraction so
# the score is not inflated by placeholders.
PLACEHOLDER = re.compile(r"<(b|span|em)\b[^>]*\bid=[^>]*>\s*[—–-]\s*</\1>", re.I)
TAG = re.compile(r"<[^>]+>")
# Prose-bearing blocks. Nav labels, buttons and table cells are excluded: they
# are labels, not prose, and rewriting them breaks the JS that reads them.
BLOCK = re.compile(
    r"<(p|li|h1|h2|h3|figcaption|blockquote|summary)\b[^>]*>(.*?)</\1>", re.S | re.I
)


def prose_blocks(path: Path) -> list[tuple[str, str]]:
    """Return (tag, inner-html) pairs. The tag matters: a list item's leading em
    dash is a term-definition glossary, which is correct typography, while the
    same dash mid-paragraph is the AI reveal dash."""
    raw = path.read_text()
    if path.suffix == ".md":
        body = STRIP.sub(" ", re.sub(r"```.*?```", " ", raw, flags=re.S))
        out = []
        for ln in body.splitlines():
            s = ln.strip()
            if not s or s.startswith(("#", "|", ">")):
                continue
            out.append(("li" if s.startswith(("-", "*", "1.", "2.")) else "p", s))
        return out
    body = PLACEHOLDER.sub("<b>N</b>", STRIP.sub(" ", raw))
    return [(m.group(1).lower(), m.group(2)) for m in BLOCK.finditer(body)]


def is_glossary_dash(tag: str, text: str, pos: int) -> bool:
    """A list item opening `term — definition` is a glossary, not a reveal.

    Qualifies when the text before the dash is a bare term: at most three words,
    no finite verb. `Ak — the k-th channel` qualifies; `The data is real — each
    plane is...` does not, even inside a list.
    """
    if tag not in ("li", "dd", "dt"):
        return False
    head = text[:pos].strip().lstrip("*_`").strip()
    words = head.split()
    if not words:
        return False
    # A math expression is a single term however many tokens it spans:
    # `ΔA = A(x) - A(baseline)` and `P(c | x with a cube blanked at p)` are
    # glossary heads, not clauses.
    if set("=|∂ΣΔ∇") & set(head) or (head.count("(") and head.endswith(")")):
        return len(head) <= 60
    if len(words) > 3 or len(head) > 42:
        return False
    verbs = {"is", "are", "was", "were", "has", "have", "does", "do", "means",
             "gives", "shows", "keeps", "makes", "runs", "needs", "costs", "lives"}
    return not any(w.lower().strip(",.;:") in verbs for w in words)


def scan(path: Path) -> tuple[list[tuple[str, str, str, str]], int]:
    hits: list[tuple[str, str, str, str]] = []
    words = 0
    for tag, block in prose_blocks(path):
        text = html.unescape(TAG.sub("", block)).replace(" ", " ")
        text = " ".join(text.split())
        if not text:
            continue
        words += len(text.split())
        for sev, label, pat in PATTERNS:
            for m in re.finditer(pat, text):
                if label == "em-dash emphasis" and is_glossary_dash(tag, text, m.start()):
                    continue
                lo, hi = max(0, m.start() - 32), min(len(text), m.end() + 32)
                hits.append((sev, label, m.group(0).strip(), f"…{text[lo:hi]}…"))
    return hits, words


def main() -> None:
    targets = [Path(a) for a in sys.argv[1:]] or sorted(
        [*Path("web").rglob("*.html"), Path("README.md")]
    )
    grand: list[tuple[str, Path, str, str, str]] = []
    total_words = 0
    print(f"{'score':>6} {'P0':>3} {'P1':>3} {'P2':>3} {'words':>6}  file")
    print("-" * 78)
    for path in targets:
        if not path.is_file():
            raise FileNotFoundError(path)
        hits, words = scan(path)
        total_words += words
        counts = {s: sum(1 for h in hits if h[0] == s) for s in ("P0", "P1", "P2")}
        score = 1000 * len(hits) / words if words else 0.0
        print(f"{score:6.1f} {counts['P0']:3d} {counts['P1']:3d} {counts['P2']:3d} "
              f"{words:6d}  {path}")
        grand += [(h[0], path, h[1], h[2], h[3]) for h in hits]

    print("-" * 78)
    p0 = sum(1 for g in grand if g[0] == "P0")
    print(f"{1000 * len(grand) / total_words:6.1f} "
          f"{p0:3d} {sum(1 for g in grand if g[0] == 'P1'):3d} "
          f"{sum(1 for g in grand if g[0] == 'P2'):3d} {total_words:6d}  ALL "
          f"(hits per 1000 words)")

    if grand:
        print()
        for sev in ("P0", "P1", "P2"):
            rows = [g for g in grand if g[0] == sev]
            if not rows:
                continue
            print(f"=== {sev} ({len(rows)}) " + "=" * 40)
            for _, path, label, match, ctx in rows:
                print(f"  {path.name:26s} {label:26s} {match!r}")
                print(f"  {'':26s} {ctx}")

    sys.exit(1 if p0 else 0)


if __name__ == "__main__":
    main()
