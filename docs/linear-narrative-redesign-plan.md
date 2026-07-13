# Linear-Narrative Redesign — Implementation Plan (Plan 1 of 3: Structural Rewire)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a **static frontend** project — there are no unit tests; each task's "test" is a **browser-preview verification** run by the orchestrator (edit → serve over http → read_page / read_console_messages / screenshot).

**Goal:** Convert the 13-page hub-and-spoke site into a single non-branching `Next →` chain of 10 pages, with no content rewrites.

**Architecture:** One shared `chain.js` defines the reading order and renders a top progress rail + bottom Prev/Next on every page. The old `network.html` / `code.html` / `about.html` / `methods/index.html` fold their content into chain pages and are deleted. No prose is invented here — pages keep their existing copy; the verified "evolution" narrative is Plan 3.

**Tech Stack:** Vanilla HTML + CSS + ES-module JS. No framework, no build, no new dependency. Data via `fetch()` from the existing bundle (`web/public/data`).

**Source of truth:** design spec [`docs/linear-narrative-redesign.md`](./linear-narrative-redesign.md). This plan inherits every decision there; if the spec changes, revisit affected tasks.

## Global Constraints

- Zero runtime dependencies; no build step; no bundler. (spec §10, §11)
- Preserve the data-bundle contract (`web/DATA_CONTRACT.md`) and the `web-export` ML core — untouched. (spec §11)
- Every changed asset gets a bumped `?v=N` query string in the HTML that references it (existing cache-busting convention).
- The reading order is defined **once** in `chain.js` `CHAIN`; no page hard-codes its neighbours. (spec §5, §6)
- CSS-3D scenes and `scene3d.js` mount functions are reused as-is; do not rewrite them here. (spec §6)
- Served over http(s); `file://` breaks `fetch()` — the orchestrator serves `web/` for every verification.
- Git: work on a branch off `main` (never commit to `main`); one commit per task; **do not push**. The orchestrator creates the branch before Task 1 and confirms with the user.
- Match the existing code style in each file (indentation, mono-label patterns, `reveal` classes).

---

### Task 1: `chain.js` — shared reading-order + nav component

**Files:**
- Create: `web/js/chain.js`
- Modify: `web/styles.css` (append a `chain-*` block)

**Interfaces:**
- Produces: `export const CHAIN` (ordered array of `{file, label, group}`); `export function mountChainNav(currentFile, base = "")` — `currentFile` is a `CHAIN[].file`; `base` is `""` for root pages, `"../"` for pages under `methods/`. Renders into `[data-chain-rail]` (top) and `[data-chain-foot]` (bottom) if present; throws if `currentFile` is unknown.
- Consumes: nothing.

- [ ] **Step 1: Create `web/js/chain.js` with this exact content**

```js
/* web/js/chain.js
   Single source of truth for the guided reading order, plus the shared
   top progress rail and bottom Prev/Next. Defined once; each page calls
   mountChainNav() with its own position. Zero deps, no build. */

export const CHAIN = [
  { file: "index.html",                        label: "Network",              group: "Network"   },
  { file: "methods/notgradcam.html",           label: "Baseline",             group: "Methods"   },
  { file: "methods/gradcam.html",              label: "Grad-CAM",             group: "Methods"   },
  { file: "methods/layercam.html",             label: "LayerCAM",             group: "Methods"   },
  { file: "methods/guided_gradcam.html",       label: "Guided Grad-CAM",      group: "Methods"   },
  { file: "methods/occlusion.html",            label: "Occlusion",            group: "Methods"   },
  { file: "methods/integrated_gradients.html", label: "Integrated Gradients", group: "Methods"   },
  { file: "methods/integrated_gradcam.html",   label: "Integrated Grad-CAM",  group: "Methods"   },
  { file: "compare.html",                      label: "Compare",              group: "Compare"   },
  { file: "reproduce.html",                    label: "Reproduce",            group: "Reproduce" },
];

export function mountChainNav(currentFile, base = "") {
  const i = CHAIN.findIndex((p) => p.file === currentFile);
  if (i < 0) throw new Error(`chain.js: unknown page "${currentFile}"`);
  const href = (p) => base + p.file;

  const rail = document.querySelector("[data-chain-rail]");
  if (rail) {
    rail.innerHTML = CHAIN.map((p, k) => {
      const cls = "chain-step" + (k === i ? " current" : k < i ? " done" : "");
      const cur = k === i ? ' aria-current="page"' : "";
      return `<a class="${cls}" href="${href(p)}"${cur}><span class="cs-dot"></span>` +
             `<span class="cs-lab">${p.label}</span></a>`;
    }).join("");
  }

  const foot = document.querySelector("[data-chain-foot]");
  if (foot) {
    const prev = CHAIN[i - 1], next = CHAIN[i + 1];
    foot.innerHTML =
      (prev ? `<a class="chain-prev" href="${href(prev)}">← ${prev.label}</a>`
            : `<span class="chain-prev-spacer"></span>`) +
      (next ? `<a class="chain-next" href="${href(next)}"><span class="cn-k">Next</span> ${next.label} →</a>`
            : `<span class="chain-end">End of the tour</span>`);
  }
}
```

- [ ] **Step 2: Append the `chain-*` styles to `web/styles.css`**

Add a block styling: `.chain-top` (slim sticky top bar: brand + rail), `.chain-rail` (horizontal flex of `.chain-step`), `.chain-step .cs-dot` (small circle) + `.cs-lab` (mono micro-label), `.chain-step.current` (accent dot + full-opacity label), `.chain-step.done` (muted), `.chain-foot` (space-between row), `.chain-next` (prominent button reusing `.btn.primary` look), `.chain-prev` (quiet link), `.chain-end` (muted). Reuse existing CSS variables (`--accent`, `--ink-dim`, `--mono`, `--line`, `--radius`). On mobile (`max-width: 640px`) hide `.cs-lab`, keep dots. Respect `prefers-reduced-motion` (no transitions).

- [ ] **Step 3: Verify in the browser**

Temporarily add `<nav class="chain-top"><div class="chain-rail" data-chain-rail></div></nav>` and `<div class="chain-foot" data-chain-foot></div>` + a module script calling `mountChainNav("methods/gradcam.html", "../")` to a scratch copy, serve `web/`, and confirm: rail shows 10 steps, "Grad-CAM" is `.current`, the two before it (`Network`, `Baseline`) are `.done`, and the chain-foot shows Prev = `← Baseline` and Next = `LayerCAM →`. `read_page` the rail; `read_console_messages` clean. Remove the scratch page.

- [ ] **Step 4: Commit**

```bash
git add web/js/chain.js web/styles.css
git commit -m "feat(web): add chain.js shared reading-order + progress nav"
```

---

### Task 2: `index.html` = Hook + The Network (fold in `network.html`)

**Files:**
- Modify: `web/index.html`
- Read (source of content to fold): `web/network.html` (its `#arch-mount`, `#forward`/`#pipe`, `#deck-mount`/`#hist` sections + its module script importing `mountDeck, initPipeline, mountArchitecture`)

**Interfaces:**
- Consumes: `mountChainNav` from Task 1; `mountArchitecture`, `initPipeline`, `mountDeck`, `mountFlow` from `scene3d.js`; `loadJSON` from `data.js`.
- Produces: the chain entry point (`Next → Baseline`).

- [ ] **Step 1: Replace the 5-link nav** with the chain top bar

Swap the existing `<nav class="nav">…</nav>` for:
```html
<nav class="chain-top">
  <a class="brand" href="index.html"><span class="mark" aria-hidden="true"></span><b>GRAD-CAM</b><span>/ CT</span></a>
  <div class="chain-rail" data-chain-rail></div>
</nav>
```

- [ ] **Step 2: Keep the hero**, but replace its two CTAs (`Explore the methods` / `Read the code`) with a single primary CTA `Start the tour →` linking to `methods/notgradcam.html` (the bottom chain-foot will carry the canonical Next).

- [ ] **Step 3: Fold the full Network onto the page.** After the hero, keep the existing `#the-network` section (the `#flow-mount` activation flow as the intro), then append the `network.html` sections verbatim: the `#arch-mount` block, the `#forward` / `#pipe` forward-pass stepper block, and the `#deck-mount` + `#hist` activations block. Remove the old `network.html` "Explore the network →" CTA. Delete the `#methods-strip` section and the `#build` section from index (build → Task 5 / reproduce).

- [ ] **Step 4: Merge the scripts.** Extend index's module script so it does what `network.html`'s did: after `mountFlow`, also `mountArchitecture(#arch-mount, graph, {onSelect: syncLayer})`, `initPipeline({...#pipe elements...})`, and the example/layer-tab + `mountDeck` + histogram wiring. Copy the exact element IDs and mount calls from `network.html` lines 98–184. Bump `scene3d.js?v=8`, `data.js?v=3`. Add at the end:
```html
<script type="module">
  import { mountChainNav } from "./js/chain.js?v=1";
  mountChainNav("index.html", "");
</script>
```
and add `<div class="chain-foot" data-chain-foot></div>` just before `<footer>`.

- [ ] **Step 5: Verify.** Serve `web/`, load `/index.html`. Confirm: hero + activation flow + 3D architecture (drag-orbit) + forward-pass stepper (Play/Prev/Next) + activations deck + histogram all render; no `#methods-strip`; chain rail current = "Network"; chain-foot Next = "Baseline" → `methods/notgradcam.html`; `read_console_messages` clean; data loads over http. Screenshot for the record.

- [ ] **Step 6: Commit**

```bash
git add web/index.html web/styles.css
git commit -m "feat(web): index becomes Hook + full Network; drop nav hub + methods strip"
```

---

### Task 3: Wire the 7 method pages onto the chain

**Files (modify all 7):**
- `web/methods/notgradcam.html`, `gradcam.html`, `layercam.html`, `guided_gradcam.html`, `occlusion.html`, `integrated_gradients.html`, `integrated_gradcam.html`
- Read (for the gradcam fold-in only): `web/code.html`

**Interfaces:**
- Consumes: `mountChainNav(currentFile, "../")` from Task 1.
- Produces: 7 chain links. Prev/Next are derived from `CHAIN` automatically — the only per-page variable is `currentFile`.

Per-page argument (base is `"../"` for all 7):

| Page | `currentFile` |
|---|---|
| notgradcam.html | `methods/notgradcam.html` |
| gradcam.html | `methods/gradcam.html` |
| layercam.html | `methods/layercam.html` |
| guided_gradcam.html | `methods/guided_gradcam.html` |
| occlusion.html | `methods/occlusion.html` |
| integrated_gradients.html | `methods/integrated_gradients.html` |
| integrated_gradcam.html | `methods/integrated_gradcam.html` |

> **Orchestration:** these 7 edits are identical in shape and touch disjoint files → dispatch **one Sonnet 5 subagent per page in parallel** (spec §13a; superpowers:dispatching-parallel-agents). `gradcam.html` gets one extra sub-step (Step 4).

- [ ] **Step 1: Replace the 5-link nav** on each page with the chain top bar (note the `../` on brand href):
```html
<nav class="chain-top">
  <a class="brand" href="../index.html"><span class="mark" aria-hidden="true"></span><b>GRAD-CAM</b><span>/ CT</span></a>
  <div class="chain-rail" data-chain-rail></div>
</nav>
```

- [ ] **Step 2: Add `<div class="chain-foot" data-chain-foot></div>`** immediately before each page's `<footer>`, and remove any in-page "next/prev method" or "back to methods" links and the `methods/index.html` "Go deeper" cross-links.

- [ ] **Step 3: Add the chain-nav script** before `</body>` (bump bench/other module versions if edited):
```html
<script type="module">
  import { mountChainNav } from "../js/chain.js?v=1";
  mountChainNav("methods/<THIS_PAGE>.html", "../");
</script>
```

- [ ] **Step 4 (gradcam.html only): fold in `code.html`'s Grad-CAM walkthrough.** Move the Grad-CAM-specific annotated-code section(s) from `web/code.html` into `gradcam.html` after its existing code block. Keep it as static `<pre><code>` matching the page's existing `.codeblock` markup. Do not invent commentary — move the existing text.

- [ ] **Step 5: Verify (per page).** Serve `web/`, load each method page. Confirm: chain rail current highlights the right step; chain-foot Prev/Next match `CHAIN` order (e.g. occlusion → Prev "Guided Grad-CAM", Next "Integrated Gradients"); the reading bench still renders (GT beside heat, hover read-out); no dangling links to `methods/index.html`; console clean. For gradcam.html also confirm the folded code block renders.

- [ ] **Step 6: Commit** (one commit; list all 7 files + gradcam's code fold-in)

```bash
git add web/methods/*.html
git commit -m "feat(web): wire 7 method pages into the Next-chain; fold code.html into gradcam"
```

---

### Task 4: `compare.html` — the measured synthesis (from `methods/index.html`)

**Files:**
- Create: `web/compare.html` (repurpose the shell of `methods/index.html`, which is then deleted in Task 6)
- Reuse: `web/js/bench.js`, `web/js/data.js`

**Interfaces:**
- Consumes: `mountChainNav("compare.html", "")`; `loadJSON`, `drawHeat`, `drawGray`, `METRIC_LABELS`, `fmtMetric` from `data.js`; the bundle's per-example `meta.metrics`.
- Produces: the thesis-payoff page (`Prev ← Integrated Grad-CAM`, `Next → Reproduce`).

- [ ] **Step 1: Create `web/compare.html`** with the standard `<head>` (`styles.css?v=8`), the `chain-top` bar (brand href `index.html`, `data-chain-rail`), a `page-head` (“Seven methods, one case — measured”), a results section, `chain-foot`, footer, and the chain-nav script `mountChainNav("compare.html", "")`.

- [ ] **Step 2: Render the comparison.** For the first bundle example, draw all 7 methods' heat-maps on the same CT slice in a grid, each with its 3 metrics (`mass_in_gt`, `inside_outside_ratio`, `pointing_acc`) from `meta.metrics[method]`, sorted by `mass_in_gt` descending so the ranking is visible. Add a one-line honest caption: the sharpest-looking map is not necessarily the top-ranked. Use `drawHeat`/`drawGray` + `fmtMetric` + `METRIC_LABELS` (reuse `bench.js` helpers; do not add new metric logic).

- [ ] **Step 3: Verify.** Serve `web/`, load `/compare.html`. Confirm: 7 maps render on one case, metrics shown, sorted by `mass_in_gt`; chain rail current = "Compare"; Prev = "Integrated Grad-CAM", Next = "Reproduce"; console clean. Screenshot.

- [ ] **Step 4: Commit**

```bash
git add web/compare.html
git commit -m "feat(web): add compare.html — 7 methods measured + ranked on one case"
```

---

### Task 5: `reproduce.html` absorbs `about.html` + the trust band

**Files:**
- Modify: `web/reproduce.html`
- Read (fold in): `web/about.html`; the `#build` "Built to be trusted" section removed from `index.html` in Task 2

**Interfaces:**
- Consumes: `mountChainNav("reproduce.html", "")`.
- Produces: the chain terminus (`Prev ← Compare`, no Next → shows "End of the tour").

- [ ] **Step 1:** Replace the nav with the `chain-top` bar; add `chain-foot` before the footer; add the chain-nav script `mountChainNav("reproduce.html", "")`; bump `styles.css?v=8`.
- [ ] **Step 2:** Append the `about.html` content (sources, license, references, AI-use disclosure) and the "Built to be trusted" three-item band (Real data / Implementation-first / Hashed & reproducible) as sections on this page.
- [ ] **Step 3:** Add the http-serving note: "Open the site over http(s) — `python -m http.server` in `web/`; opening `index.html` via `file://` blocks the data `fetch()`." (spec §7)
- [ ] **Step 4: Verify.** Serve; load `/reproduce.html`. Confirm: reproduction steps + sources + disclosure + trust band all present; chain rail current = "Reproduce"; Prev = "Compare"; chain-foot shows "End of the tour" (no Next); console clean.
- [ ] **Step 5: Commit**

```bash
git add web/reproduce.html
git commit -m "feat(web): reproduce.html absorbs about + trust band; chain terminus"
```

---

### Task 6: Delete folded-in pages + fix references + full-chain verification

**Files:**
- Delete: `web/network.html`, `web/code.html`, `web/about.html`, `web/methods/index.html`

- [ ] **Step 1: Grep for surviving references** to the four pages and fix them:
```bash
cd web && grep -rnE "network\.html|code\.html|about\.html|methods/(index\.html|\")|methods/\"" --include="*.html" .
```
Any link to a deleted page must point to its new home (network→`index.html`, code→`methods/gradcam.html`, about→`reproduce.html`, methods hub→`compare.html`).

- [ ] **Step 2: Delete the four pages.**
```bash
cd web && git rm network.html code.html about.html methods/index.html
```

- [ ] **Step 3: Full-chain verification.** Serve `web/`. Starting at `/index.html`, click `Next` through all 10 steps to `reproduce.html` and back via `Prev`; confirm the chain is unbroken and the order matches spec §3. On each page: `read_console_messages` clean, data bundle loads (`read_network_requests` — no 404s). Check `/index.html` and one method page at mobile width + dark mode (`resize_window`). Confirm no request to any deleted page.

- [ ] **Step 4: Commit**

```bash
git add -A web
git commit -m "chore(web): delete network/code/about/methods-index; verify unbroken 10-step chain"
```

---

## Self-Review (against spec)

- **Spec coverage:** §3 spine → Tasks 2–5 build all 10 pages in order; §5 nav → Task 1 + per-page swaps; §6 components → Task 1 (`chain.js`) + reuse of `bench.js`/`scene3d.js`; §8 disposition → Tasks 2/3/4/5 (folds) + Task 6 (deletes); §7 data flow → per-page verify steps; §12 verification → Task 6 Step 3. **Not covered here (by design):** §4 verified content (Plan 3), §10 motion refactor (Plan 2) — noted below.
- **Placeholder scan:** none — `chain.js` is given in full; per-page edits are exact swaps of named blocks; fold-ins move named existing sections.
- **Type consistency:** `mountChainNav(currentFile, base)` signature and the `CHAIN[].file` values are used identically in Tasks 2–5; `base` = `""` for root pages (index, compare, reproduce), `"../"` for the 7 method pages.

## Follow-up plans (scoped, not yet written)

- **Plan 2 — Motion refactor:** migrate `reveal.js` + `scene3d.js` reveal plumbing to WAAPI + CSS scroll-timeline (IO fallback). Independent; current `reveal.js` already works, so low urgency. (spec §10)
- **Plan 3 — Verified evolution content:** write the §4 "weakness → fix" narrative into each method page's motivation slot + the `compare.html` ranking commentary. **Gated on citation-verification + `benchmark.json` cross-check.** (spec §4, §9)
