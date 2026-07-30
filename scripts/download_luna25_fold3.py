"""Download the fold-3 val split of FLARE-AutoMSC Dataset005_LUNA25.

Resumable and idempotent: a file already present with a non-trivial size and a
valid gzip header is skipped. Every download is verified before being accepted,
so a partial or HTML-error body is never left on disk masquerading as data.

Writes a per-file manifest CSV (path, bytes, sha256) for provenance.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

LOG = logging.getLogger("luna25_dl")

BASE = (
    "https://huggingface.co/datasets/FLARE-MedFM/FLARE-AutoMSC/"
    "resolve/main/Dataset005_LUNA25"
)
TIMEOUT = (10, 120)  # (connect, read) — never unbounded; see the TCIA freeze incident
RETRIES = 4


def token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    path = Path.home() / ".cache/huggingface/token"
    if not path.exists():
        raise FileNotFoundError(f"no HF token at {path} and HF_TOKEN unset")
    return path.read_text().strip()


def is_valid_gzip(path: Path, min_bytes: int = 512) -> bool:
    """A gated 401 returns a short HTML body; require a real gzip member."""
    try:
        if path.stat().st_size < min_bytes:
            return False
        with gzip.open(path, "rb") as fh:
            fh.read(1024)
        return True
    except Exception:
        return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, tok: str) -> tuple[bool, str]:
    """Download to a temp file, validate, then atomically move into place."""
    if dest.exists() and is_valid_gzip(dest):
        return True, "skip"
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"Authorization": f"Bearer {tok}"}
    last = ""
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                if r.status_code != 200:
                    last = f"http{r.status_code}"
                    if r.status_code in (401, 403, 404):
                        return False, last  # not transient; do not retry
                    raise OSError(last)
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if not is_valid_gzip(tmp):
                last = "bad_gzip"
                raise OSError(last)
            tmp.replace(dest)
            return True, "ok"
        except Exception as exc:  # transient: back off and retry
            last = last or type(exc).__name__
            tmp.unlink(missing_ok=True)
            if attempt < RETRIES:
                time.sleep(2**attempt)
    return False, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--meta", type=Path, default=Path("artifacts/jsc/luna25_meta"))
    ap.add_argument("--out", type=Path, default=Path("data/luna25"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", type=Path, default=Path("artifacts/jsc/fold3_val_manifest.csv"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N cases only")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    splits = json.loads((args.meta / "splits_final.json").read_text())
    cases = splits[args.fold]["val"]
    if args.limit:
        cases = cases[: args.limit]
    LOG.info("fold %d val: %d cases -> %d files", args.fold, len(cases), 2 * len(cases))

    img_dir = args.out / "imagesTr"
    lbl_dir = args.out / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, Path]] = []
    for c in cases:
        jobs.append((c, f"{BASE}/imagesTr/{c}_0000.nii.gz", img_dir / f"{c}_0000.nii.gz"))
        jobs.append((c, f"{BASE}/labelsTr/{c}.nii.gz", lbl_dir / f"{c}.nii.gz"))

    tok = token()
    done = skipped = 0
    failures: list[tuple[str, str, str]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, url, dest, tok): (case, url, dest) for case, url, dest in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            case, url, dest = futs[fut]
            ok, status = fut.result()
            if not ok:
                failures.append((case, dest.name, status))
                LOG.error("FAIL %s (%s)", dest.name, status)
            elif status == "skip":
                skipped += 1
            else:
                done += 1
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                LOG.info(
                    "%d/%d files | new=%d skip=%d fail=%d | %.1fs (%.1f files/s)",
                    i, len(jobs), done, skipped, len(failures), el, i / max(el, 1e-9),
                )

    # Verify on disk rather than trusting the counters.
    present = [(c, p) for c, _u, p in jobs if p.exists() and is_valid_gzip(p)]
    LOG.info("on-disk valid files: %d / %d", len(present), len(jobs))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "path", "bytes", "sha256"])
        for case, path in sorted(present, key=lambda t: str(t[1])):
            w.writerow([case, str(path), path.stat().st_size, sha256(path)])
    LOG.info("manifest -> %s", args.manifest)

    if failures:
        LOG.error("%d file(s) failed:", len(failures))
        for case, name, status in failures[:20]:
            LOG.error("  %s %s %s", case, name, status)
        return 1
    if len(present) != len(jobs):
        LOG.error("count mismatch: %d valid vs %d expected", len(present), len(jobs))
        return 1
    LOG.info("ALL_COMPLETE %d files, %.1f MB", len(present),
             sum(p.stat().st_size for _c, p in present) / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
