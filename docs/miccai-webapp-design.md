# MICCAI Educational Challenge 2026 — Interactive Grad-CAM Website

**Requirements & Design Specification**

| | |
|---|---|
| **Project** | `gradcam-repro` → interactive educational website |
| **Target** | MICCAI Educational Challenge 2026 — *"From Pixels to Patients"* |
| **Deadline** | 2026-07-31 (AoE) per the official page — **not the binding constraint**: build is AI-driven and single-pass (~1 day), so timeline is comfortable |
| **Doc created** | 2026-07-09 |
| **Status** | Design approved — proceeding to implementation plan (writing-plans) |
| **Decisions locked** | Submission = **website + repo + notebook**; Interactivity = **precomputed + static site**; Frontend = **plain static HTML + CSS + vanilla JS (no framework, no build step, zero npm deps)**; Executor = **AI (Claude), single-pass build** |

---

## 0. How to read this doc

Sections 1–3 are the *why* (competition constraints, fit, product vision). Sections 4–9 are the *what* (package, site map, interactivity, data contract, stack). Sections 10–13 are the *how* (build order, risks, decisions, compliance). Section 12 records the locked decisions. Section 14 records the **as-built** flow.

**Companion doc:** [`experience-design.md`](./experience-design.md) is the *experience layer* — interaction design, engagement/attraction, and knowledge-dissemination strategy, plus the prioritized improvement backlog. Read it for *how the site should feel and teach*; read this doc for *what it is and how it's built*.

---

## 1. The competition — hard constraints we must satisfy

Source: <https://miccai-sb.github.io/challenge.html> (read 2026-07-09).

- **Theme**: educational materials on fundamental medical-image-computing / CAI concepts, **emphasis on practical, real-world "how to make it work" perspectives** over general overviews.
- **Accepted formats** (explicitly include): videos, blog posts, papers, IPython notebooks, PowerPoint, **GitHub source code with documentation**, and **interactive demos and websites**. → Our chosen format is first-class.
- **Audience**: accessible to **new Masters and PhD students** entering the MICCAI field.
- **Eligibility**: students and academics; individual or collaborative; no MICCAI paper required.
- **Judging** (expert panel → top 30% → popular vote at MICCAI 2026), scored on:
  1. Accuracy of content
  2. Comprehensiveness of content
  3. Appropriate level (new Masters/PhD accessible)
  4. Relevance to the MICCAI community
  5. Clarity
- **Desired characteristics** (submission should do *at least one*):
  - Share first-hand experience from a real project/deployment/collaboration
  - **Reproduce, benchmark, or critically evaluate existing work** ← we do this
  - **Provide verified, runnable code on real medical data with pinned dependencies** ← we do this
  - Bring in a clinician/surgeon/domain expert as a *named* contributor (optional for us)
  - Walk through a specific artifact end to end
  - **Teach a topic where good learning resources are scarce** ← we do this
- **Emphasis**: "teach things an AI assistant can't easily reproduce."
- **Format limits**: English only (text and video); paper component ≤ 4000 words (excl. references); slides ≤ 40; video ≤ 20 min.
- **AI-use disclosure**: allowed but **must be clearly stated**.
- **Submission portal**: OpenReview (MICCAI 2026).
- **Finalist requirement**: a 1-minute summary video for promotion.
- **Prizes**: $300 / $200 / $100 (needs ≥ 3 submissions to award).

### Compliance checklist (tracked to submission)

- [ ] All copy in English
- [ ] AI-use disclosure statement present (About page + repo README)
- [ ] Any write-up ≤ 4000 words; if a video is made, ≤ 20 min
- [ ] Satisfies ≥ 1 "desired characteristic" (we satisfy **three**)
- [ ] Content level readable by a new Masters student (progressive disclosure — §9)
- [ ] Submitted via OpenReview before AoE cutoff, with buffer
- [ ] 1-minute finalist video prepared (only if we advance)
- [ ] Medical-data redistribution license respected (§11)

---

## 2. Why this project fits — and what already exists

### Fit map

| Challenge preference | Our asset |
|---|---|
| "Interactive demos and websites" accepted format | The website is exactly this |
| Practical implementation details over overviews | User mandate: *"implementation details should be the main focus"* |
| "Reproduce / benchmark / critically evaluate existing work" | 7 attribution methods reproduced + benchmarked on real CT |
| "Verified runnable code on real medical data w/ pinned deps" | `uv.lock` + MSD `Task06_Lung` pipeline already runs CPU/MPS |
| "Teach a topic where resources are scarce" | Honest *quantitative* comparison of CAM methods on 3D medical data is genuinely under-taught |

### Already built (the hard ML core — do **not** rebuild)

Located in `src/gradcam_repro/`:

| Module | Role |
|---|---|
| `model.py` | 3D CNN, CAM tap at `stage2`, spatial-evidence head → 2 logits |
| `attribution.py` | The 7 methods (see below) |
| `data.py`, `real_ct.py` | Synthetic toy data + real MSD `Task06_Lung` preprocessing (tumour-present vs non-tumour patch classification derived from the segmentation mask) |
| `train.py`, `evaluate.py` | Training + the 3 attribution-quality metrics |
| `visualize.py` | Heatmap / grid rendering |
| `cli.py` | `gradcam-repro` commands: `real-preprocess`, `real-train`, `demo`, `score`, `deck` |

**Seven methods** (README-verbatim): `notGradCAM`, `Grad-CAM`, `Guided Grad-CAM`, `LayerCAM`, `Occlusion sensitivity`, `Integrated Gradients`, `Integrated Grad-CAM`.

**Three evaluation metrics** (attribution-vs-ground-truth-mask): `mass_in_gt` (share of normalized attribution mass inside the GT lesion mask), `inside_outside_ratio` (mean heat inside ÷ outside), `pointing_acc` (does the heatmap peak land inside the mask).

**Provenance policy already in place**: `demo` writes `artifacts/figures/manifest.json` with checkpoint hash + generation params + figure hashes; `deck` refuses to build if figures are not in the manifest. **We extend this policy to the web bundle — a strong "we don't fake figures" trust signal for judges.**

**Remaining work is packaging, not modeling**: turn existing artifacts into a static interactive site + a clean companion notebook.

---

## 3. Product vision

**One-line pitch**: *"How do you know a medical-imaging CNN is looking at the tumour and not an artifact? An implementation-first tour of seven attribution methods, built from scratch on real lung CT, with honest benchmarks you can reproduce."*

**Homepage — "what problems we solve"** (the user's explicit ask):
- **Trust & interpretability in medical AI**: does the model localize the lesion, or cheat on shortcuts?
- **What you'll learn**: how each attribution method works, *how to implement it*, how to *quantitatively* evaluate it (not just eyeball heatmaps), and where each one fails.
- **Study internal network distributions**: inspect activations / feature-map distributions at the CAM tap point.
- **Reproduce everything**: pinned-dependency runnable code on real data.

**Audience/depth resolution**: every content page uses **progressive disclosure** — an accessible *intuition layer* on top, an *implementation layer* (real annotated code + gotchas) underneath. This satisfies both MICCAI's "appropriate level" criterion and the user's "implementation-detail focus" without compromise.

---

## 4. Submission package (DECIDED: website + repo + notebook)

Three interlinked artifacts, each pulling a different judging lever:

1. **Interactive website** (the face) — teaching + exploration; carries Clarity, Comprehensiveness, popular-vote appeal. Hosted on GitHub Pages.
2. **Reproducible repo** `gradcam-repro` (the proof) — pinned `uv.lock`, one-command reproduction; carries "verified runnable code on real medical data." The website's `web-export` command lives here.
3. **Companion IPython notebook** (the walkthrough) — a linear, runnable narrative that regenerates the site's key figures; carries the "IPython notebook" desired-format and doubles as the reproduce path. Base it on the existing `notebooks/real_ct_msd_lung_gradcam.ipynb` and `notebooks/gradcam_3d_visual_walkthrough.ipynb`.

Cross-linking: website → "Reproduce it" page → repo + notebook; notebook → website for interactive exploration; repo README → website + challenge disclosure.

---

## 5. Site map

```
/                         Home — problems we solve, learning goals, entry points
/methods/                 Overview of all 7 methods + how to read a heatmap
/methods/grad-cam         ┐
/methods/guided-grad-cam  │
/methods/layercam         │ one page per method (shared template, §9 layered)
/methods/occlusion        │
/methods/integrated-grad  │
/methods/integrated-gradcam
/methods/not-gradcam      ┘ (baseline / contrast)
/flow                     Computation-flow visualizer (hero interaction #2)
/internals                Network internals & feature distributions (hero interaction #1 home)
/benchmark                Side-by-side comparison + sortable metrics + critical evaluation
/reproduce                Pinned-deps runnable path, repo + notebook links, provenance
/about                    Team, AI-use disclosure, references, license, contact
```

**Per-method page template** (7 pages, mostly data-driven):
1. Intuition (1 short paragraph — accessible)
2. The math (the actual formula)
3. **Implementation** — the real code from `attribution.py`, annotated line-by-line
4. Interactive result — slice navigator + pixel-hover heatmap over the CT, overlaid with GT mask
5. Quantitative scorecard — `mass_in_gt`, `inside_outside_ratio`, `pointing_acc`
6. Failure modes / gotchas — e.g. *occlusion fill uses the per-volume mean, not zero, to avoid injecting artificial black-cube edge evidence* (a real, teachable implementation detail from this codebase)

---

## 6. Interactivity model (DECIDED: precomputed + static)

No live inference, no backend. A Python **export step** bakes everything the frontend needs; the site is pure static assets on GitHub Pages. This is the lowest-risk path for the deadline and is **100% reproducible** (every displayed number traces back to a manifest hash).

### The three signature interactions

1. **Hover network internals** (`/internals`, `/flow`): hover a layer node → tooltip shows layer type, input/output shape, parameter count, one-line role, and the *precomputed activation summary* (feature-map thumbnail + histogram) for the currently selected example.
2. **Animated computation flow** (`/flow`): step-through of Grad-CAM: forward pass → capture `stage2` activations → backward pass → per-channel gradient weighting → ReLU → upsample → overlay on the CT. Each step reads precomputed intermediate tensors; a "play/step" control advances the stages.
3. **Slice + heatmap hover** (method pages, `/benchmark`): the 3D CT volume is presented as a **2D slice navigator** (montage/slider — *not* a WebGL volume renderer, to control risk); hovering a pixel reads the raw attribution value at that voxel from a compact array.

---

## 7. Data export contract

New command in the repo (proposed): `gradcam-repro web-export --out web/public/data/`. It runs the existing pipeline on a small, fixed set of showcase examples and emits:

```
web/public/data/
├── manifest.json          # checkpoint hash, uv.lock hash, git commit+dirty, timestamp, schema version
├── model_graph.json       # nodes: {name, type, in_shape, out_shape, param_count, role, cam_tap: bool}
├── benchmark.json         # aggregate metrics table: method × example × {mass_in_gt, io_ratio, pointing_acc}
└── examples/
    └── <example_id>/
        ├── meta.json      # patient/patch id (de-identified), true class, predicted class + logits
        ├── slices/        # CT slices as PNG (grayscale) + GT-mask overlay PNG
        ├── attributions/  # per-method attribution: PNG (display) + compact raw array (hover values)
        └── activations/   # per-layer: feature-map thumbnail PNG + histogram JSON
```

- **Showcase set**: start with the **3 representative CT patches** the deck already uses (README: "default deck grid renders three representative CT patches"); expandable to 6 (`demo --samples 6`) if time allows.
- **Raw hover arrays**: downsample/quantize (e.g. 8-bit, per-slice) to keep payload small; document the quantization.
- **Provenance**: `manifest.json` extends the existing figure-manifest policy; the site footer links "every figure's provenance" to it.

---

## 8. Tech stack & hosting

- **Backend/export**: existing Python package + one new `web-export` CLI command (draws from `real_ct.py`, `attribution.py`, `evaluate.py`, `visualize.py`). No new heavy deps.
- **Frontend**: **plain static HTML + CSS + vanilla JS (ES modules) — no framework, no bundler, no build step, zero npm dependencies.** The site is a set of `.html` pages sharing one `styles.css` and a few small `js/` modules that `fetch()` the precomputed bundle. The three signature interactions use only browser primitives: inline `<svg>` for the network graph + step-through flow animation; HTML `<canvas>` for pixel-hover heatmaps; DOM for the slice slider. Rationale: a handful of content pages over precomputed static JSON does not justify a framework/toolchain; a no-build site is trivially hostable on GitHub Pages, maximally inspectable, and reproducible with zero dependencies (fits MICCAI's clean-environment ethos). (There is **no runtime backend** — `web-export` is an offline one-shot data generator, §7.)
- **Hosting**: **GitHub Pages** (static, free, matches "GitHub source code with documentation"); site built from repo, data bundle committed or built in CI.
- **Repo layout addition**:
  ```
  gradcam-repro/
  ├── src/gradcam_repro/…        # (existing) + web_export.py
  ├── web/                       # NEW — plain static site (no build step)
  │   ├── index.html + methods/*.html + flow.html + internals.html + benchmark.html + reproduce.html + about.html
  │   ├── styles.css             # single shared stylesheet
  │   ├── js/                    # vanilla ES modules: data-loader, network-graph, flow-player, slice-viewer, heatmap-hover, scorecard
  │   └── public/data/…          # export bundle (§7), already generated by web-export --device cpu
  ├── notebooks/…                # (existing) companion notebook, finalized
  └── docs/miccai-webapp-design.md   # this file
  ```

---

## 9. Content depth strategy — the "implementation details" treatment

Progressive disclosure on every page:

- **Top (intuition)**: plain-language "what this does and why," one figure. Readable by a new Masters student.
- **Middle (mechanism)**: the formula + the animated/interactive result.
- **Bottom (implementation)**: the *actual annotated code* from this repo, plus **"gotchas" callouts** — the non-obvious engineering decisions that make it work. Seed list of gotchas already known from this codebase:
  - Occlusion uses per-volume **mean** fill, not zero (avoids fake black-cube edge evidence).
  - A strictly translation-invariant `conv + GAP` head **cannot** learn a left/right position label → this scaffold uses a small non-negative spatial-evidence head (a real "why the obvious architecture fails" lesson).
  - CAM tap point is `stage2` (`16×6×6×6` in the toy config) — why that layer, and how tap choice changes the heatmap.
  - Metrics beyond eyeballing: why `pointing_acc` and `mass_in_gt` disagree, and what each rewards.

These gotchas are the core of "teach things an AI can't easily reproduce."

---

## 10. Build order (dependency-ordered, AI-executed in one pass)

Execution is AI-driven and single-pass, so this is ordered by *dependency*, not calendar. The detailed task breakdown lives in the implementation plan (writing-plans).

| Stage | Deliverable | Depends on |
|---|---|---|
| **0. Data contract** | `web-export` command; Grad-CAM example end-to-end bundle; **freeze the schema** | existing pipeline |
| **1. Skeleton + hero** | Site scaffold + design system; Home; Grad-CAM page fully working; `/flow` visualizer for Grad-CAM | stage 0 schema |
| **2. Breadth** | Remaining 6 method pages (templated); `/internals` distributions | stage 1 template |
| **3. Synthesis** | `/benchmark` + critical evaluation; `/reproduce`; `/about`; finalize notebook | stage 2 data |
| **4. Polish** | Accessibility, responsive, English copy-edit (writing-anti-ai pass), provenance footer, deploy to Pages, 1-min video draft, OpenReview submission | stages 1–3 |

**Priority order if scope is ever trimmed** (not expected under AI execution): volumetric 3D renderer → 2D montage; `/internals` standalone → inline; 6 examples → 3; animations → static steps. **Never cut**: reproducibility path, English polish, AI disclosure.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 3D-in-browser complexity | Present volumes as **2D slice navigators** (the deck already renders 2D slices); no WebGL volume renderer |
| Scope creep | MVP-first ordering + explicit priority cut list (§10) |
| **MSD `Task06_Lung` redistribution license** | Do **not** redistribute raw MSD data; ship only small **derived, de-identified** patches/figures + instructions to download MSD. **Verify Medical Segmentation Decathlon license terms before shipping any derived image.** |
| English polish (judged on Clarity) | `writing-anti-ai` pass + careful copy-edit; keep any prose within 4000 words |
| Accessibility / popular-vote appeal | Colorblind-safe colormaps (e.g. viridis), keyboard nav, alt text, mobile-responsive |
| Overclaiming (no named clinician) | We satisfy 3 desired characteristics without one; **do not fabricate** a contributor |
| Data payload size on Pages | Quantize hover arrays; limit showcase examples; lazy-load per page |

---

## 12. Decisions (locked; executor's defaults — override anytime)

Since execution is delegated to the AI, remaining choices are resolved with sensible defaults:

1. **Frontend framework** — **none**: plain static HTML + CSS + vanilla JS (ES modules), no framework, no build step, zero npm deps. Network graph + flow via inline `<svg>`; heatmap pixel-hover via `<canvas>`; slice slider via DOM. Trivially hostable on GitHub Pages; maximally inspectable/reproducible.
2. **Language** — website + this doc in **English** (submission requirement); no Chinese mirror unless requested.
3. **Named clinician contributor** — **none** (not fabricated); we already satisfy three "desired characteristics" without one.
4. **Showcase examples** — **start at 3** (deck default), auto-expand to 6 if the export bundle stays small.
5. **Companion** — **plain Jupyter notebook** (not Quarto), so the site and notebook stay independently simple.
6. **KB linkage** — deferred; mirror a summary into `medical-imaging-ai` only on explicit request (per KB rules).

---

## 13. Definition of done

- Website live on GitHub Pages: Home + 7 method pages + `/flow` + `/internals` + `/benchmark` + `/reproduce` + `/about`.
- Every displayed figure/number traces to `manifest.json` (provenance intact).
- Repo reproduces the export bundle from pinned deps in one documented path.
- Companion notebook runs top-to-bottom and regenerates key figures.
- Compliance checklist (§1) fully green.
- Submitted on OpenReview before 2026-07-31 AoE with buffer.

---

## 14. As-built presentation flow & 3D visualizations (updated 2026-07-11)

The site evolved during the single-pass build. This section records what is **actually built**; §5–§6 above are the original plan and are kept for history. Where they conflict, §14 wins.

### 14.1 Narrative arc (talk order)

The site is structured like a talk: **the network we probe → the code → the methods → trust**. Nav order (top-left): **Network · Code · Methods · Reproduce · About**. The `/benchmark` page was **removed** (a standalone leaderboard didn't serve the teaching arc; per-method scorecards carry the metrics instead). The original `/internals` and `/flow` pages are **merged into one `network.html`**.

### 14.2 Homepage (`index.html`) — leads with the network

- **Hero** — the hook (*"Is the network looking at the tumour — or an artifact?"*) + the running-example reading panel (case `lung_038`: CT / GT mask / Grad-CAM).
- **01 · The network** — states the **running example** (the hero scan), the **dataset source** (MSD `Task06_Lung`, reframed tumour-present-vs-not), and embeds a concrete **activation-map flow** of that example: `CT input → stage1 → stage2 (CAM tap) → stage3 → Grad-CAM`, real per-layer activation maps connected by lines; **hover any layer → popover enlarges it and unfolds its top channels**. CTA → `network.html`.
- **02 · Seven neighbouring methods** — the method chips.
- **03 · Built to be trusted** — real data / implementation-first / hashed provenance.
- *Removed*: the old "What you can explore" nav-card grid (redundant with the top nav).

### 14.3 Network page (`network.html`) — three complementary 3D/interactive views

All CSS-3D (`transform-style: preserve-3d`), vanilla JS, no library. Engine: `web/js/scene3d.js`.

1. **The architecture** — an isometric 3D **block diagram**: each layer a stack of slabs ("loaf") where **face = spatial resolution** and **depth = channel count**, so resolution-shrinks-as-channels-grow reads at a glance. Blocks **slide/assemble in** on scroll-into-view (one-shot, staggered; robust reveal — never stays invisible); **drag to orbit**; **click a conv stage → drives the activation inspector below**. CAM-tap (`stage2`) glows; crisp flat legend row with shapes + params.
2. **The forward pass, in 3D** — one CT volume walked through the network beat by beat (input → stage1 → stage2 tap → stage3 → GAP→logit → Grad-CAM), feature planes rendered turbo-coloured + per-channel contrast-stretched. **Advanced by a click stepper — rail buttons + Prev / Play / Next (no scroll-scrubbing).** Drag to orbit. *(An earlier scroll-driven "scrollytelling" version was replaced at the user's request — it hijacked page scroll and forced a very tall section.)*
3. **Activations & distributions** — a **fan-out feature deck** (channels spread along Z with a flat↔fanned slider + drag-orbit) beside the ReLU-sparsity histogram; driven by the example tabs + layer tabs, and by clicks on the architecture above. No scroll animation.

The homepage flow (§14.2) is the concrete, example-first alternative to the architecture block diagram; the block diagram is the structural view and lives here.

### 14.4 Rendering honesty note

Feature-map planes/tiles are **per-channel contrast-stretched and turbo-colour-mapped** so the sparse post-ReLU activations are legible (most voxels sit near zero — shown in the distribution histogram). This is a display transform only, stated in-page.

### 14.5 Asset cache-busting

Shared assets are version-queried (`styles.css?v=N`, `js/scene3d.js?v=N`) in each page's references so returning browsers pick up updates without a manual hard-reload. Bump the query when the shared file changes.
