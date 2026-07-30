# Project Requirements — Grad-CAM on CT

*The project lead's requirements and directives that shape this site, captured as a durable reference. Complements the design spec ([`linear-narrative-redesign.md`](./linear-narrative-redesign.md)) and the master design doc ([`README.md`](./README.md)). Where a requirement carries a stated rationale, it is kept — the "why" is the point.*

*Status: living document. Last updated 2026-07-16.*

---

## 1. Product goal

- **R1.1** — Submission to the **MICCAI Educational Challenge 2026** ("From Pixels to Patients"). Optimise for the judging levers: accuracy, comprehensiveness, appropriate level, relevance, clarity.
- **R1.2** — The teaching thesis is **"you cannot judge a saliency map by eye — measure it against the ground truth, and the methods disagree."** Every design choice should serve making that lesson unavoidable.
- **R1.3** — Implementation-first and honest: real code, real data, honest quantitative comparison; show methods failing; no hand-drawn figures.

## 2. Narrative & information architecture

- **R2.1** — One **complete, non-branching browsing experience** — a single guided path front to back. *Why: the old homepage had ~17 exits; readers were dropped into a fork with no guided path.*
- **R2.2** — The homepage **shows the network** (the running example) up front — dataset source + network structure + a visualisation — before the methods.
- **R2.3** — Method treatment is **hybrid**: Grad-CAM is the deep worked example ("how to read any attribution method"); the other methods carry the "they disagree — measure" thesis.
- **R2.4** — The methods form a **motivated chain**: each page opens with the *previous* method's shortcoming and what *this* method fixes ("Grad-CAM is coarse → LayerCAM refines → …"), and ends by teasing the next. *Do not delete the method pages — chain them.*
- **R2.5** — The **baseline (mean activation map) comes first** in the chain — it is a degenerate baseline, not an improvement, so it sets up the first shortcoming.
- **R2.6** — A short **measured-comparison page** (all methods on one case, ranked) is the thesis payoff near the end.
- **R2.7** — A **quiz** ("check your understanding") comes at the end of the tour (on the compare page, after the scoreboard).
- **R2.8** — Navigation is a `Next →` chain with a progress rail; **no top-nav hub**.

## 3. Content

- **R3.1** — The **"Reproduce" page documents reproducing the experiment** (data → preprocess → train → attribute + score), **not building the website**. Building/serving the site is at most a demoted, optional footnote.
- **R3.2** — **Every mathematical symbol is explained.** Each formula has a `where:` legend defining each symbol, grounded in the actual implementation (`src/attribution.py`).
- **R3.3** — Formulas use **real mathematical typesetting (MathML)**, not ad-hoc web/HTML math (`<sub>`/`<sup>` + slashes). Real fraction bars, ∑/∫ with limits, correctly positioned indices.
- **R3.4** — On the baseline page, **show all the channels being averaged**, mark the **empty (silent) ones**, and **explain why they are empty** (post-ReLU: filters that don't fire output all-zeros). *Why: the sparse-activation reality is a real teaching point — most post-ReLU channels are ~0.*
- **R3.5** — Because this is a visual method, **show the ground-truth tumour mask beside each attribution map** (side-by-side), plus the quantitative scorecard.
- **R3.6** — All factual content (method years/authors, the "weakness → fix" claims, formulas) must be **verified against primary sources + our own code/metrics**, not asserted from memory. Metric definitions (`mass_in_gt`, `inside_outside_ratio`, `pointing_acc`) shown where produced.

## 4. Interaction

- **R4.1** — It is an **interactive explainer**, with one consistent gesture vocabulary: hover = inspect, click = drill in, drag = orbit a 3D scene, play/step = advance, scroll = read (never hijacked to drive state).
- **R4.2** — The network is shown with **3D visual effects** (CSS-3D architecture block diagram — drag to orbit; forward-pass stepper).
- **R4.3** — The homepage network is a **concrete example**: real activation maps per layer, connected left→right, hover a layer to enlarge and unfold its channels.
- **R4.4** — Math symbols are **interactive**: hovering a symbol in the formula ↔ its legend entry cross-highlight (linked, both directions).
- **R4.5** — The activations deck is **fixed fully fanned** — the flat↔fanned slider is removed.
- **R4.6** — The quiz gives **instant per-question feedback** (correct/wrong + a one-line explanation) and a running score.

## 5. Technical principles & constraints

- **R5.1** — **Zero runtime dependencies, no build step, fully static.** Plain HTML + CSS + vanilla ES-module JS. *This is a stated value and a judging-relevant one (maximally inspectable, clean-environment ethos) — it governs library choices.*
- **R5.2** — Consequences of R5.1, decided explicitly this build:
  - Animation → **native Web Animations API + CSS**, not a library (chosen over Motion One / GSAP).
  - Math → **native MathML**, not KaTeX/MathJax (chosen to keep zero-dep *and* preserve the interactive symbol legend).
  - 3D → **CSS-3D transforms**, not Three.js.
- **R5.3** — **Reproducible + hashed provenance.** Every displayed figure/number is generated by the open-source code and traced (checkpoint + data cache + `uv.lock` + git commit hashed into the manifest). No hand-edited figures.
- **R5.4** — The site is served over **http(s)** (it `fetch()`es a JSON+PNG bundle; `file://` is unsupported).
- **R5.5** — Accessibility is reach: colour is never the only channel for a claim; content degrades to readable static form without JS; keyboard/touch paths tracked in the backlog.

## 6. Deployment

- **R6.1** — Hosted on **GitHub Pages** from a **public** repo (`github.com/Bardli/pixels-to-patients`), deployed by a **GitHub Actions workflow** that publishes `web/` on every push to `main`. Live at `https://bardli.github.io/pixels-to-patients/`.
- **R6.2** — The full repository is the public reproducible artifact (large data / checkpoints stay gitignored).

## 7. Open follow-ups (backlog)

- Add the live URL to `README.md`.
- Verify the reference citations against primary sources before final submission.
- Site-wide data cache-busting (currently only the notgradcam channel grid busts its data fetch).
- 1-minute summary video; finalise the companion notebook.
- Touch fallback for hover-only reveals; social-preview meta tags; colour-vision-safety pass on the turbo colormap; reduced-motion audit; alt-text on figures.
