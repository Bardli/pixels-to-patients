# Experience Design — Interaction, Engagement & Knowledge Dissemination

**Companion to [`miccai-webapp-design.md`](./miccai-webapp-design.md).** That doc is the *build spec* (constraints, site map, data contract, stack, as-built §14). **This doc is the experience layer**: how the site is meant to be *interacted with*, how it *attracts and holds* a reader, and how it *teaches and travels*. Where they overlap, the build spec owns facts; this doc owns intent and the improvement backlog.

| | |
|---|---|
| **Created** | 2026-07-11 |
| **Scope** | The static site (`web/`) — home, network, code, 7 method pages, reproduce, about |
| **Owns** | Interaction principles, engagement strategy, pedagogy/dissemination, prioritized backlog (§4) |
| **Does not own** | Data schema, CLI, hosting (see build spec) |

---

## 0. Who we're designing for, and what "good" means

Two readers, one funnel:

1. **Expert panel** (first gate) — scores on *accuracy, comprehensiveness, appropriate level, relevance, clarity*. Wins on **depth + honesty + correctness**.
2. **New Masters/PhD student** (the level target) *and* the **MICCAI popular vote** (second gate, for finalists). Wins on **first-impression pull + a memorable "aha" + genuine learning they can act on**.

Design tension we resolve throughout: *depth for the panel* vs *approachability for the student/voter*. Our resolution is **progressive disclosure** — never dumb it down, layer it. This is the spine of every decision below.

Design north star (one line): **"You should be able to *see*, in under a minute, whether the network looks at the tumour — and then be able to *reproduce* that claim yourself."** Seeing = engagement; reproducing = credibility.

---

## 1. Interaction design

### 1.1 Principle — progressive disclosure, three layers on every content page

| Layer | Who it serves | Form |
|---|---|---|
| **Intuition** | new student, casual voter | one plain-language paragraph + one figure |
| **Mechanism** | student going deeper | the formula + an interactive result |
| **Implementation** | panel, practitioners | the *real annotated code* + "gotchas" callouts |

The reader chooses depth by scrolling/hovering, not by us guessing. This satisfies "appropriate level" **and** "implementation-detail focus" without compromise.

### 1.2 Interaction vocabulary — one verb, one meaning (consistency is the whole game)

The site must teach its own controls implicitly. We keep a fixed mapping so a gesture learned on page 1 works everywhere:

| Gesture | Always means | Where it lives |
|---|---|---|
| **Hover** | *inspect / reveal detail* (non-destructive) | home activation-flow (unfold channels), method reading-bench (voxel value), figure captions |
| **Click** | *drill in / select* | architecture conv-block → activation inspector; method chips → method page; rail step → beat |
| **Drag** | *orbit a 3D scene* | network 3D block diagram, forward-pass stage, fan-out deck |
| **Play / Prev / Next** | *advance a sequence* | forward-pass stepper (replaced scroll-scrubbing) |
| **Slider** | *scrub a continuous parameter* | fan-out deck (flat↔fanned), slice navigator |
| **Scroll** | *read* — never hijacked to drive state | whole site (the one scroll-scrubbed section was removed by design) |

> **Rule:** scroll is for reading. Any state change is driven by an explicit control (hover/click/drag/play/slider), so the reader never loses their place or feels the page fighting them.

### 1.3 As-built interaction inventory

- **Home — activation-map flow**: `CT → stage1 → stage2(tap) → stage3 → Grad-CAM`, real maps joined by lines; **hover a layer → popover enlarges it and unfolds its top channels.** Concrete, example-first.
- **Network — 3D architecture block diagram**: layers as "loaves" (face = spatial, depth = channels); slide-in on scroll-into-view; **drag to orbit; click a conv stage → drives the activation inspector.**
- **Network — forward-pass stepper**: one CT volume walked beat-by-beat via **rail buttons + Prev/Play/Next**; drag to orbit.
- **Network — fan-out feature deck**: channels spread along Z via a slider; drag to orbit; beside the ReLU-sparsity histogram.
- **Method pages — reading bench**: slice + pixel-hover heatmap over CT with GT overlay + a quantitative scorecard (`mass_in_gt`, `inside_outside_ratio`, `pointing_acc`) + failure-mode callouts.

### 1.4 Feedback, discoverability, and forgiveness

- **Discoverability**: every interactive affordance states itself in a caption ("drag to orbit", "hover a layer to expand"), and cursors change (`grab`, `pointer`). Nothing hidden depends on the reader guessing.
- **Feedback**: hover → lift/glow; selection → accent; active beat → rail highlight. Every input produces an immediate visible response.
- **Forgiveness**: all interactions are non-destructive and reversible; there is no state the reader can get stuck in (the forward-pass stepper can never wedge — a real bug we fixed).

### 1.5 Interaction gaps (→ backlog)

- **Touch has no fallback for hover-only reveals** — the home flow popover, method voxel-hover, and any tooltip are invisible/unreachable on phones/tablets. (P0 — the popular vote and much MICCAI traffic is mobile.)
- **Keyboard/AT can't reach the 3D and hover reveals** — drag-orbit and hover popovers are mouse-only; no focus targets or ARIA state. (P1)
- **`prefers-reduced-motion`** is honored in CSS for transitions but not audited across *all* 3D/flow entrances. (P1)

---

## 2. Engagement & attraction

### 2.1 Identity — why it looks like it does

A deliberate **"interactive explainer" aesthetic** (Distill / seeing-theory lineage): warm paper (`#faf8f4`), a serif reading voice (Charter), mono labels, a single brick accent (`#b03a2e`), and **dark "artifact" cards** for scans/code so the data pops. This reads as *a scientist's notebook, not a product landing page* — which is the trust signal the panel and students respond to, and it avoids generic-AI-slop aesthetics.

### 2.2 The hook — first 30 seconds

- **One question, above the fold**: *"Is the network looking at the tumour — or an artifact?"* — a real anxiety in medical AI, not a feature list.
- **Immediate proof-of-substance**: the hero shows the running example (CT / GT / Grad-CAM), so within one screen the reader sees this is *real data, real output*, not slideware.
- **One "aha" per page**: home = *watch activations light up and unfold*; network = *the volume walks through a 3D net*; method = *the heatmap lands (or misses) the tumour, with a number*. Memorability comes from a single strong moment, not many small ones.

### 2.3 Motion budget — restraint as credibility

Motion is used where it *explains* (data flowing through layers, channels fanning, a heatmap projecting back), never as decoration. Slow eases, no bounce/neon. Over-animation would undercut the scientific tone and hurt the "appropriate level" / clarity scores.

### 2.4 Shareability — engineered, not hoped-for

The popular vote is won by things people *share*. Today the site is not optimized to be shared:

- **No social preview** (`og:image`, `og:title`, `twitter:card`) — a pasted link renders as a bare URL. A single striking still (the 3D net or a hot Grad-CAM over CT) as `og:image` is the highest-leverage change. (P0)
- **No deep links** — you can't link "the stage2 tap on lung_038" or a specific method; every share lands on the homepage. Hash-routing to a case/method would make the site quotable. (P2)
- **No 1-minute video** — required for finalists *and* the most shareable artifact; it should open on the hook question and end on the reproduce path. (P1, gated on advancing but cheap to pre-draft)

### 2.5 Flow & retention

The talk-arc (**hook → the network → the code → the methods → trust**) gives a spine; each section ends with a single clear next-step CTA. Risk to watch: the site must not become a long scroll of 3D toys — every interactive must pay for its scroll-cost with a lesson. (We already cut a benchmark page and a scroll-scrubbed section for this reason.)

### 2.6 Engagement gaps (→ backlog)

- Social preview meta tags (P0). Video script/draft (P1). Deep-linkable cases/methods (P2). A one-screen "start here" affordance for readers who land mid-site (P2).

---

## 3. Knowledge dissemination & pedagogy

### 3.1 The thesis worth teaching

MICCAI explicitly rewards *"teach things an AI can't easily reproduce."* Our teachable, scarce thesis: **you cannot judge a saliency method by eyeballing heatmaps — you must measure them against ground truth, and the methods disagree.** The whole site is built to make that lesson unavoidable (every method carries `mass_in_gt` / `inside_outside_ratio` / `pointing_acc`, not just a pretty picture).

### 3.2 Learning design

- **Scaffolding**: baseline (mean activation map) → gradient methods (Grad-CAM, LayerCAM) → perturbation (occlusion) → path-integrated (IG, Integrated Grad-CAM). Each page names *what it adds over the last*.
- **Gotchas as the payload**: the non-obvious engineering (occlusion uses per-volume **mean** fill not zero; the CAM tap is `stage2` and why; why a translation-invariant head can't learn a position label; why `pointing_acc` and `mass_in_gt` disagree). This is the "AI-can't-reproduce" content.
- **Honesty as pedagogy**: we show methods *failing* (cool/empty maps, misses) and the sparse-activation reality (most post-ReLU values are ~0). Showing failure teaches more than cherry-picked wins and buys credibility with the panel.

### 3.3 Multi-channel distribution (map to MICCAI's accepted formats)

| Channel | Role | Status |
|---|---|---|
| **Interactive website** | the face — explore + learn | built |
| **Repo + pinned deps** | the proof — one-command reproduce | built (web-export) |
| **Companion notebook** | linear runnable narrative | exists; needs finalize |
| **1-min video** | reach + finalist requirement | not started |
| **Blog/social thread** | top-of-funnel spread | not started |

Each channel should **cross-link** and restate the one-line thesis, so any entry point routes to the others.

### 3.4 Accessibility *is* reach (not a checkbox)

- **Colormap**: turbo is punchy (good for attraction) but not perceptually uniform / fully CVD-safe. Mitigations already partly in place: magnitude also reads as brightness, and the **GT mask is a separate green overlay** (a non-colour cue). Decide explicitly: keep turbo for appeal *and* guarantee a non-colour channel for every claim, or offer a viridis toggle. (P1)
- **Alt text / semantics**: 3D scenes carry `role="img"` + labels; verify every canvas figure has a text equivalent so a screen-reader user (and a reader pasting the page into Claude) still gets the point. (P1)
- **Mobile**: see §1.5 — hover-only content must have a tap path or the mobile reader learns nothing. (P0)
- **Plain-language on-ramp**: a short "new to this?" line or a 6-term glossary (CAM, saliency, ReLU, GAP, logit, voxel) would lower the barrier for the exact audience MICCAI targets. (P2)

### 3.5 Comprehension safeguards

Static site = no analytics = we can't watch readers struggle. Substitute: **reader-testing** — hand a fresh reader (or a fresh Claude) the page cold and check they can state the lesson and the controls. Run this on the home + one method page before submission.

---

## 4. Prioritized backlog (the actionable output)

Ranked by *impact on the two judging gates ÷ effort*.

### P0 — do before sharing/submission
1. **Touch fallback for hover reveals** — tap-to-open the home activation-flow popover and method voxel readouts. Without this, mobile voters get a broken story. *(interaction/§1.5, reach/§3.4)*
2. **Social preview meta tags** (`og:image/title/description`, `twitter:card`) on every page, with a striking still. Turns every shared link into a hook. *(engagement/§2.4)*

### P1 — do before submission if time; strong ROI
3. **Draft the 1-minute video** (script + capture the built interactions) — finalist requirement + best spread artifact. *(§2.4/§3.3)*
4. **Colormap accessibility decision** — guarantee a non-colour channel for every claim, or add a viridis toggle. *(§3.4)*
5. **Keyboard + reduced-motion audit** across the 3D/flow/stepper; add focus states + ARIA. *(§1.5)*
6. **Alt-text/text-equivalent pass** on all canvas figures. *(§3.4)*
7. **Finalize the companion notebook**; cross-link all channels to the one-line thesis. *(§3.3)*

### P2 — polish / stretch
8. Deep-linkable cases + methods (hash routing). *(§2.4)*
9. "Start here" affordance + 6-term glossary for newcomers. *(§2.5/§3.4)*
10. Reader-test home + one method page cold; fix whatever the fresh reader misses. *(§3.5)*

---

## 5. How this doc was made / keep it live

Written after the single-pass build, grounded in the as-built site (§14 of the build spec). Update the backlog as items land; when an item ships, move it to §14 of the build spec (facts) and strike it here (intent). Re-run a cold reader-test before the OpenReview submission.
