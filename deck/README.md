# Editable Deck Workflow

Goal: turn the image-only `GradCam.pptx` reference into an editable,
reproducible deck that can keep improving.

## Status (current, 2026-06-24)

The original `pptxgenjs` build pipeline (`build.mjs` + `slides/*.mjs`) depended
on an external Codex `presentations` skill whose builder
(`scripts/build_artifact_deck.mjs`) no longer ships in the installed skill
version. That pipeline is **archived under `deck/archive/`** and is no longer
the way to change the deck.

**Current workflow: edit the `.pptx` directly** with Claude's native PPTX
tools (unpack → edit slide XML → repack). The canonical editable deck is
`artifacts/deck/gradcam-roadmap.pptx` (the single, unversioned live file; older
`v2`–`v21` renders and their layout/preview caches were cleaned up on
2026-06-24). The design docs in this folder (`claim-spine.md`,
`design-system.md`, `source-notes.md`, `contact-sheet-plan.md`) are retained as
the narrative/visual reference.

> Note: `artifacts/` is gitignored, so the deck is **not** in version control.
> Keep your own backup before large edits.

Experiment logic in `src/gradcam_repro/` and the figure-generation script
(`scripts/render_real_ct_deck_figures.py`) are unchanged — demo images on
slides 13–14 are still experiment-derived and regenerated from Python.

### Direct-edit conventions (2026-06-24)

- **Formulas: use native baseline sub/superscript runs, NOT decorative Unicode.**
  Characters like `ₖ ᵢ ⱼ ₚ ₛ ₜ ₀` (subscripts), the modifier letters `ᵏ ᶜ`, and
  astral math glyphs (`𝟙`, `𝑑`) have no glyph in WPS's default font and render as
  tofu boxes (`∑ₖ` → `∑□`). They were all converted to `<a:r><a:rPr baseline=
  "-25000"/>` (sub) / `baseline="30000"` (sup) runs carrying a plain ASCII letter.
  `³` (U+00B3) is universal and kept as-is. When editing a formula, add new
  sub/superscripts as baseline runs, not pasted Unicode.
- **Slide 4 architecture diagram is a generated image**, not vector shapes:
  `artifacts/figures/realct_architecture.png`, produced by
  `scripts/render_architecture_figure.py`, which introspects the real
  `RealCtCNN` (layers, kernels, per-stage shapes) so the figure always matches
  the model. Regenerate it, then re-embed (replace `ppt/media/image10.png`).

---

## Build (ARCHIVED — see deck/archive/)

Run the experiment first so the demo images are current:

```bash
uv run gradcam-repro all
```

Then export the editable PPTX:

```bash
uv run gradcam-repro deck
```

Outputs:

- `artifacts/deck/gradcam-editable.pptx`
- `artifacts/deck/preview/slide-XX.png`
- `artifacts/deck/contact-sheet.png`
- `artifacts/deck/layout/slide-XX.layout.json`

## Editing Model

- Change narrative and slide content in `deck/slides/content.mjs`.
- Change visual grammar in `deck/slides/helpers.mjs`.
- Keep experiment logic in `src/gradcam_repro/`.
- Keep generated files in `artifacts/`; they are rebuildable.
- Keep attribution figures experiment-derived. `deck/build.mjs` checks
  `artifacts/figures/manifest.json` before export.

The PPTX contains editable text boxes, tables, method cards, and diagram blocks.
The two demo slides embed generated PNGs from `artifacts/figures/`; those images
are reproducible from the Python code and should be regenerated before final
deck export.
