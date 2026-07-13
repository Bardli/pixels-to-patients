# Design System

Slide size: `1280 x 720`. Page margin: `64 px`.

Tone: technical lecture, methods reference, reproducible experiment.

## Structure — the "sandwich"

- Cover (01) and discussion (19) use the dark background.
- All content slides use the light background.
- This gives the deck dark bookends with a bright, readable middle.

## Palette

Light content:

- Background: `#FFFFFF`
- Surface (cards): `#F1F4F8`
- Rule / hairlines: `#E3E8EE`
- Ink: `#13181E`
- Muted ink: `#5E6873`
- Faint ink: `#9AA6B2`

Dark bookends:

- Background: `#0E141B`
- Surface: `#19222C`
- Rule: `#2A3540`
- Text on dark: `#C7D2DD` (muted), `#FFFFFF` (primary)

Semantic accents (use by meaning, never decoratively):

- Hot `#E8590C` — attribution heat, the primary signal/accent and the motif color
- Analysis blue `#1F6FA8` — gradient-based localization and method reasoning
- Evidence green `#2E7D52` — pros, confirmation, evidence

## Motif

One repeated element: a small filled **square** (`chip`) — a single "heatmap
cell" echoing the 32×32 task. It marks the kicker on every slide, leads section
labels, and appears as a fading row on the cover. Carry it; do not invent new
decorative elements.

## Typography

- Family: `Aptos` (fallback `Arial`); formulas use `Consolas`-style mono.
- Cover title: 56 px bold
- Slide title: 38 px bold
- Section label / kicker: 12 px bold, muted, uppercase
- Body: 16–20 px
- Tables and captions: 12–15 px

## Layout rules

- **No accent rules or full-width bars under titles.** Use the chip + whitespace.
- One dominant proof object per slide.
- Cards are borderless tinted surfaces with a thin (4 px) accent edge on the
  left — never a heavy full border, never a nested card.
- Slide number shown top-right as `NN / 19`, faint.
- Demo figures may be PNGs because they are generated reproducibly.
- Keep every text box inside its container; if content does not fit, shrink the
  type or split the slide rather than overflow.
