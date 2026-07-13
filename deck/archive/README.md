# Archived deck build pipeline

Archived 2026-06-24.

## What this is

The original programmatic deck builder:

- `build.mjs` — Node entry point; shelled out to the Codex `presentations`
  skill's `scripts/build_artifact_deck.mjs` with a fixed CLI
  (`--slides-dir`, `--out`, `--slide-count 19`, `--slide-size 1280x720`, …).
- `slides/content.mjs` — narrative, formulas, card/table text.
- `slides/helpers.mjs` — visual grammar (layouts, method cards, diagrams).
- `slides/slide-01.mjs … slide-19.mjs` — per-slide entry stubs.

## Why it was archived

`build.mjs` hard-codes
`…/presentations/26.601.10930/skills/presentations/scripts/build_artifact_deck.mjs`.
The installed skill is now `26.623.12021`, which restructured the skill and no
longer ships `build_artifact_deck.mjs` (it moved to an `artifact_tool` /
`container_tools` API). Adapting `build.mjs` to the new API would have changed
fonts/layout and risked new regressions, so the project switched to editing the
`.pptx` directly (see `../README.md`).

## If you ever want to revive it

You would need to either (a) restore a `presentations` skill version that still
provides `scripts/build_artifact_deck.mjs` and point `PRESENTATIONS_SKILL_DIR`
at it, or (b) rewrite `build.mjs` against the current `artifact_tool` API. The
content in `slides/content.mjs` is the last source that produced
`gradcam-roadmap-v21.pptx`; the live deck has since diverged via direct PPTX
edits (v22+).
