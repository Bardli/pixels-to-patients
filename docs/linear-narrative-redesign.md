# Linear-Narrative Redesign — Design Spec

*Status: approved design, pre-implementation. Date: 2026-07-13.*
*Supersedes the as-built site map recorded in [`miccai-webapp-design.md`](./miccai-webapp-design.md) §14. Design philosophy in [`README.md`](./README.md) §3 is unchanged and still governs.*

---

## 1. Problem

The site is a **hub-and-spoke of 13 pages** (`index` → `network`, `code`, `methods/` hub → 7 method pages, `reproduce`, `about`). The homepage alone offers **~17 exits** (5 nav links + 2 hero CTAs + 1 network CTA + 7 method chips + 2 footer links). A first-time reader is dropped into a fork with no guided path.

**Goal:** one complete, **non-branching** browsing experience — a single guided path the audience walks front to back, with the project's thesis delivered in order.

Non-goal restated: "non-branching" means *no competing exits / no hub*, **not** "few pages." A long single-direction chain is acceptable; a hub with many exits is not.

## 2. Decisions on record

Settled through the brainstorming dialogue (each an explicit user choice):

1. **Method treatment = hybrid.** Grad-CAM is the deep worked example ("how to read any attribution method"); the rest deliver the "they disagree — measure, don't eyeball" thesis. *(Not: one method only; not: 7 co-equal deep dives.)*
2. **Structure = guided chain (approach B).** A small sequence of pages joined only by `Next →`; the top-nav hub is removed. *(Not: single mega-page; not: spine + optional appendices.)*
3. **The other methods are kept, not deleted** — restructured into a **motivated chain**: each page opens with the *previous* method's shortcoming and what *this* method fixes, and ends by teasing the next.
4. **Baseline (`notgradcam`) = first chain link, own page.** It is a degenerate baseline (no gradients), not an improvement, so it opens the chain rather than sitting "after Grad-CAM."
5. **A short synthesis page is kept** at the end of the chain as the thesis payoff (all 7 measured on one case, ranked).
6. **Motion tech = native, zero-dependency** (decided earlier): vanilla JS + Web Animations API + CSS scroll-driven animation (with an IntersectionObserver fallback for Safari/Firefox). CSS-3D scenes retained. No library, no build. `README.md` §3 principle 8 stands.

## 3. The spine (10 pages, one `Next →` chain)

```
①  index.html                        Hook + The Network (told in full)
②  methods/notgradcam.html           Baseline: mean activation map — no gradients, class-agnostic
③  methods/gradcam.html              + class-specific gradients = Grad-CAM   ← ANCHOR / deep dive
④  methods/layercam.html             + per-voxel positive gradients (finer)
⑤  methods/guided_gradcam.html       × guided backprop (sharp edges)
⑥  methods/occlusion.html            drop gradients — perturb & measure output drop
⑦  methods/integrated_gradients.html axiomatic, input-space path integral
⑧  methods/integrated_gradcam.html   IG's axioms in feature space × ΔA (synthesis)
⑨  compare.html                      7 methods on one case, measured + ranked (thesis payoff)
⑩  reproduce.html                    one-command reproduction + sources + AI disclosure
```

**Narrative arc:** see the model and what it sees → the naive baseline can't tell tumour from not → add gradients (Grad-CAM), in full depth → each successor fixes the last one's flaw → put them side by side and *measure* (they disagree) → reproduce it yourself. The thesis lands on ⑨.

The "Grad-CAM evolution" the user asked for is carried by **②→③** (mean-map baseline → add gradients), not on the homepage.

## 4. Method chain — motivation table (PROVISIONAL)

Order and every "fixes / shortcoming" claim below are **provisional**. They are factual claims about published methods and must be **verified at content-writing time** (see §9) and **cross-checked against our own `benchmark.json` numbers** — including the honest cases where a "fix" looks better but does *not* win on the metric.

| # | Page | What it adds | Shortcoming that motivates the next |
|---|------|--------------|--------------------------------------|
| ② | notgradcam | Mean of feature channels — the pre-gradient baseline | Class-agnostic: identical for tumour vs. not |
| ③ | gradcam | Class-specific gradient weighting (the anchor) | Coarse, low-resolution (upsampled from the stage2 tap) |
| ④ | layercam | Per-voxel positive-gradient weights → finer localization | Still not pixel-sharp; feature-space |
| ⑤ | guided_gradcam | × guided backprop → sharp edges | Guided backprop is edge-detector-like; class sensitivity questioned (sanity checks) |
| ⑥ | occlusion | No gradients — perturb input, measure probability drop | Slow; resolution tied to patch size |
| ⑦ | integrated_gradients | Axiomatic (sensitivity, completeness); input-space | Input-space maps are noisy, not feature-localized |
| ⑧ | integrated_gradcam | IG's path integral in feature space × ΔA (synthesis) | — (closes the chain) |

Each chain page ③–⑧ has the same shape: **motivation (prior flaw → this fix) → intuition → formula → real code → interactive reading bench → its own scorecard → tease next.** Page ② (baseline) is the exception — it *establishes* the first shortcoming rather than fixing one, then hands off to Grad-CAM.

## 5. Navigation model

- **Remove** the 5-link nav hub.
- **Shared footer component** on every page: `← Prev` · a slim 10-step progress rail (current highlighted; grouped visually as Network · Methods · Compare · Reproduce) · a prominent `Next: <title> →`.
- Top bar keeps only the brand (→ `index`) plus the same slim progress rail.
- Progress dots are **clickable** (orientation / jump) but visually subordinate to `Next` — the primary forward gesture is always `Next`.
- Interaction vocabulary unchanged (`README.md` §3.5): hover = inspect, click = drill in, drag = orbit, play/step = advance, scroll = read.

## 6. Components

- **New:** a shared `chain-nav` (top progress rail + bottom Prev/Next), driven by a single ordered list of `{href, label}` so the sequence is defined once and every page reads its position from it.
- **Reused as-is:** `bench.js` (the reading bench already switches method + example and shows GT beside the heat-map); `scene3d.js` (`mountFlow`, `mountArchitecture`, `initPipeline`, `mountDeck`); `data.js`.
- **Refactored:** the reveal/scroll plumbing in `scene3d.js`/`reveal.js` moves onto WAAPI + CSS scroll-timeline with an IO fallback (removes the hand-rolled rAF coordination that previously wedged).

## 7. Data flow

Unchanged bundle contract (`web/DATA_CONTRACT.md`). Each page fetches only what it renders:
- ① `manifest` + first example's `activations` + `ct_slice` + `gradcam` attribution.
- ②–⑧ per-method: `meta` + `ct_slice` + `attributions/<method>` for each example.
- ⑨ all 7 attributions + `metrics` for the shared case.
- Served over http(s) (GitHub Pages). `file://` fails `fetch()` — documented on ⑩ with the `python -m http.server` note.

## 8. Old-page disposition (13 → 10)

| Old page | Fate |
|---|---|
| `index.html` | Restructured: Hook + Network only; methods-strip removed; trust band → ⑩ |
| `network.html` | **Deleted** — 3D architecture + forward-pass stepper + activation flow fold into ① |
| `code.html` | **Deleted** — Grad-CAM code walkthrough folds into ③; generic parts trimmed |
| `methods/index.html` | **Repurposed** into `compare.html` (⑨); the branching grid is gone |
| `methods/*.html` (7) | **Kept**, re-sequenced as chain links ②–⑧ with motivation intros + Prev/Next |
| `about.html` | **Deleted** — merged into ⑩ |
| `reproduce.html` | Kept (⑩) + absorbs `about` + index's trust band |

## 9. Content risk & verification

The evolution/motivation narrative (§4) — method years, authors, contributions, the "fixes X" transitions, and every formula — is **factual and must not be frozen from memory**. At content-writing time:
- run **citation-verification** on each claim and formula against the source papers;
- **cross-check each "fix" against `benchmark.json`** so the site shows where the metric agrees *and where it doesn't* (the honest-pedagogy thesis);
- lock the chain order only after the verified lineage + our metrics are in.

## 10. Tech & motion (confirmed)

Vanilla JS + WAAPI + CSS scroll-timeline (IO fallback); zero dependencies; no build; CSS-3D retained; `chain-nav` is pure CSS + a few lines of JS. WAAPI is universally supported on judges' browsers; CSS scroll-timeline is progressive enhancement (Chromium stable; Safari/Firefox fall back to IO).

## 11. Out of scope (YAGNI)

No framework, no bundler, no animation library, no new runtime dependency. The ML core, data export (`web-export`), metrics, and the bundle format are untouched. No new figures — everything still comes from the hashed bundle.

## 12. Verification plan

Browser-preview workflow after each page is wired:
- the `Next`/`Prev` chain is unbroken end-to-end and matches the §3 order (link-check);
- console + network clean; data bundle loads over http;
- reveals fire in Chromium (scroll-timeline) and in a Safari/Firefox-style fallback (IO);
- responsive + dark-mode pass on ① (heaviest page) and one chain page.

## 13. Follow-ups (post-approval)

Hand to **writing-plans** for the implementation plan. Likely phases: (a) build `chain-nav` + reorder/rewire the 10 pages behind it, no content change; (b) fold `network`/`code`/`about` into their targets and delete; (c) refactor motion to WAAPI/CSS; (d) write the verified §4 chain content (gated on citation-verification); (e) build `compare.html`; (f) full-chain verification pass.
