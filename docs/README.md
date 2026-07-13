# Grad-CAM on CT — Design Document

*An implementation-first, interactive tour of seven attribution methods for medical-imaging CNNs, measured on real lung CT.*
**Submission to the MICCAI Educational Challenge 2026 — "From Pixels to Patients."**

This is the front door to the project's design. Read it top-to-bottom for the *why* and the *shape*; follow the links in §5 for the *how*.

---

## 1. Background — the problem we teach

A tumour classifier can be **right for the wrong reason**: keying off a scanner artifact, the patient's position, or an image border instead of the lesion. Before anyone trusts such a model in a clinical pipeline, they have to answer one question:

> **Is the network looking at the tumour — or an artifact?**

Attribution / saliency methods (Grad-CAM and its neighbours) claim to answer this by painting a heat-map of "where the model looked." But there is a trap the field under-teaches: **you cannot judge a saliency method by eyeballing its heat-map.** Two methods produce different maps on the same case; a map can look convincing and still miss the lesion. The honest way to compare them is to **measure each map against the ground-truth tumour mask** — and when you do, the methods disagree.

This project reproduces **seven attribution methods** from scratch and measures them on **real lung CT**, so a learner can *see* the disagreement and *reproduce* the numbers. It is deliberately implementation-first: the real code, the non-obvious engineering ("gotchas"), and honest quantitative comparison — not a survey.

**The running example.** One lung-CT patch (case `lung_038`) from the **Medical Segmentation Decathlon** (`Task06_Lung`), reframed as a *tumour-present-vs-not* classifier, runs through the entire piece — from the network's activations to each method's heat-map — so every claim is anchored to something concrete and reproducible.

**The seven methods.** Mean activation map (baseline), Grad-CAM, Guided Grad-CAM, LayerCAM, Occlusion sensitivity, Integrated Gradients, Integrated Grad-CAM.

**The three metrics** (map-vs-mask): `mass_in_gt` (share of heat inside the lesion), `inside_outside_ratio` (heat inside ÷ outside), `pointing_acc` (does the hottest voxel land on the tumour).

---

## 2. The MICCAI Educational Challenge 2026 — what we submit to

Source: <https://miccai-sb.github.io/challenge.html> (read 2026-07-09).

- **Theme**: educational materials on fundamental medical-image-computing concepts, with an **emphasis on practical, real-world "how to make it work"** over general overviews.
- **Audience/level target**: accessible to **new Masters and PhD students** entering the field.
- **Accepted formats** include **interactive demos and websites** and **GitHub source with documentation** — our format is first-class.
- **Judging** — an expert panel scores on **(1) accuracy, (2) comprehensiveness, (3) appropriate level, (4) relevance, (5) clarity**; the top ~30% then go to a **popular vote** at MICCAI 2026. Finalists provide a **1-minute summary video**.
- **Constraints**: English only; AI-use must be **clearly disclosed**; submitted via **OpenReview**; deadline **2026-07-31 (AoE)** — comfortable, since the build is AI-driven and single-pass.

**Why we fit** — the challenge asks a submission to do *at least one* of its "desired characteristics"; we do **three**:

| Desired characteristic | How we satisfy it |
|---|---|
| Reproduce / benchmark / critically evaluate existing work | 7 methods reproduced + measured on real CT, with honest disagreement |
| Verified runnable code on real medical data with pinned dependencies | `uv.lock` + MSD `Task06_Lung` pipeline; one-command reproduction |
| Teach a topic where good learning resources are scarce | *Quantitative*, honest comparison of CAM methods on 3D medical data is genuinely under-taught |

We also lean into the challenge's explicit ask to **"teach things an AI assistant can't easily reproduce"** — the payload is the engineering gotchas and the measure-don't-eyeball thesis, not a definition dump.

**Submission package** (three interlinked artifacts, each pulling a different judging lever):

1. **Interactive website** (the face) — teaching + exploration; hosted on GitHub Pages.
2. **Reproducible repo** `gradcam-repro` (the proof) — pinned deps, one-command reproduction; the site's `web-export` command lives here.
3. **Companion notebook** (the walkthrough) — a linear runnable narrative that regenerates the site's key figures.

---

## 3. Design philosophy — the principles behind every choice

1. **Implementation-first.** The main content is *real, readable code* and the non-obvious decisions that make it work (e.g. occlusion fills with the per-volume **mean**, not zero, to avoid injecting fake black-cube edges; the CAM tap is `stage2`, and why). This is the project's mandate and its differentiator.
2. **Progressive disclosure — never dumb down, layer instead.** Every content page has an **intuition** layer (plain language, one figure), a **mechanism** layer (the formula + an interactive result), and an **implementation** layer (annotated code + gotchas). The reader chooses depth by scrolling/hovering. This satisfies *both* the panel's "appropriate level" and our implementation focus.
3. **Honesty as pedagogy — and as credibility.** We **measure** attributions against ground truth instead of eyeballing them; we **show methods failing** (cool/empty maps, misses) and the sparse-activation reality (most post-ReLU values ≈ 0); we **hash every figure** into a manifest so each displayed number traces back to a checkpoint. Showing failure and provenance teaches more and earns the panel's trust.
4. **Teach the scarce thesis.** The whole site is built to make one lesson unavoidable: *saliency methods must be measured, and they disagree.*
5. **Interaction has one vocabulary.** A gesture learned on page 1 works everywhere — **hover = inspect, click = drill in, drag = orbit a 3D scene, play/step = advance a sequence, slider = scrub, scroll = read (never hijacked to drive state).**
6. **Engagement through restraint.** A scientist's-notebook aesthetic (warm paper, serif reading voice, one brick accent, dark "artifact" cards for scans/code) — not a product landing page. Motion is used only where it *explains* (data flowing through layers, a heat-map projecting back), and each page earns its length with a single memorable "aha."
7. **Accessibility is reach.** Colour is never the only channel for a claim; text equivalents for every figure; touch and keyboard paths (tracked in the backlog). The more people who can read it, the further it travels.
8. **No-build, fully static, fully reproducible.** Plain HTML + CSS + vanilla JS, zero dependencies, offline data export — trivially hostable, maximally inspectable, and in keeping with the challenge's clean-environment ethos.

---

## 4. The site at a glance (as built)

A talk-like arc: **the hook → the network we probe → the code → the methods → trust.** Nav: **Network · Code · Methods · Reproduce · About.**

- **Home** — the hook question + the running-example scan, then **"01 · The network"**: the dataset source and an **activation-map flow** of the example (`CT → stage1 → stage2 tap → stage3 → Grad-CAM`); **hover a layer to enlarge it and unfold its channels.**
- **Network** — three complementary views of the same net: a **3D architecture block diagram** (drag to orbit; click a stage to inspect), a **forward-pass stepper** (Play/Prev/Next, no scroll hijack), and a **fan-out feature deck** beside the activation-distribution histogram.
- **Methods** — 7 pages on a shared template: intuition → math → the real code → an interactive **reading bench** (slice + pixel-hover heat-map over CT with GT overlay) → a **quantitative scorecard** → failure modes.
- **Reproduce / About** — one-command reproduction path (repo + notebook) and the AI-use disclosure, references, and license.

---

## 5. Document map

| Document | What it owns | Read it for |
|---|---|---|
| **This file** (`docs/README.md`) | Background, MICCAI context, design philosophy, site overview | The *why* and the *shape* |
| [`miccai-webapp-design.md`](./miccai-webapp-design.md) | Build spec — constraints, site map, interactivity model, data contract, tech stack, decisions; **§14 records the as-built flow** | *What it is and how it's built* |
| [`experience-design.md`](./experience-design.md) | Experience layer — interaction design, engagement/attraction, knowledge dissemination, and the **prioritized improvement backlog** | *How it should feel and teach, and what's next* |

---

## 6. Status & what's next

The site is built and runs (static, served over http). The prioritized backlog lives in [`experience-design.md` §4](./experience-design.md); the highest-leverage remaining items before sharing/submission are:

- **P0** — touch fallback for hover-only reveals (mobile is much of the popular-vote traffic); social-preview meta tags so shared links render a hook.
- **P1** — a 1-minute video draft; a colour-vision-safety decision on the turbo colormap; a keyboard + reduced-motion audit; alt-text on figures; finalize the companion notebook.

---

## 7. AI-use disclosure

Per MICCAI rules, AI use is disclosed: portions of this project — including site implementation and these design documents — were built with the assistance of an AI coding assistant (Claude). The ML core, methods, metrics, and all displayed figures are generated by the accompanying open-source code and hashed for provenance.
