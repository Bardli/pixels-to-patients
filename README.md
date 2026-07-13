# gradcam-repro

Reproducible codebase for the editable `GradCam.pptx` attribution deck.

The current deck uses a real 3D CT patch-classification experiment derived from
MSD `Task06_Lung`: tumour-present versus non-tumour CT patches, with the
provided tumour segmentation mask used to evaluate whether attribution lands on
the lesion. The deck compares seven neighboring attribution methods:

- notGradCAM
- Grad-CAM
- Guided Grad-CAM
- LayerCAM
- Occlusion sensitivity
- Integrated Gradients
- Integrated Grad-CAM

The original PPT is image-only, so the extracted slide screenshots and OCR are
kept under `references/`.

## Environment

```bash
cd /Users/baiduli/ProgramProject/gradcam-repro
uv sync
```

Dependencies are managed in `pyproject.toml`; the initial environment was
created with:

```bash
uv init --package --name gradcam-repro /Users/baiduli/ProgramProject/gradcam-repro
uv add torch numpy matplotlib pillow
```

## Reproduce Current Deck

Download/extract MSD `Task06_Lung` into `data/msd/Task06_Lung`, then run the
real CT pipeline:

```bash
uv run gradcam-repro real-preprocess --size 32
uv run gradcam-repro real-train --data artifacts/real_ct/msd_lung_presence_32.pt --output artifacts/real_ct/real_ct_presence_cnn_32.pt --device cpu
uv run python scripts/render_real_ct_deck_figures.py
```

Build the editable PPTX after the real figures exist:

```bash
uv run gradcam-repro deck
```

Useful legacy synthetic commands still exist for controlled toy debugging:

```bash
uv run gradcam-repro train
uv run gradcam-repro demo
uv run gradcam-repro score
```

## Real CT Example

The first real-data extension uses the Medical Segmentation Decathlon
`Task06_Lung` CT tumour dataset. The source task is segmentation, but this
project derives a lightweight classification target from the real tumour mask:

```text
class 0 = non-tumour CT patch
class 1 = tumour-present CT patch
```

That keeps Grad-CAM in its natural classification-logit setting while using the
real tumour mask for attribution evaluation.

Place the extracted dataset at:

```text
data/msd/Task06_Lung/
├── imagesTr/
└── labelsTr/
```

Then run:

```bash
uv run gradcam-repro real-preprocess --size 32
uv run gradcam-repro real-train --data artifacts/real_ct/msd_lung_presence_32.pt --output artifacts/real_ct/real_ct_presence_cnn_32.pt --device cpu
uv run python scripts/render_real_ct_deck_figures.py
```

Default real-data outputs:

- cache: `artifacts/real_ct/msd_lung_presence_32.pt`
- checkpoint: `artifacts/real_ct/real_ct_presence_cnn_32.pt`
- figures: `artifacts/real_ct/figures/`
- deck figures: `artifacts/figures/`
- scores: `artifacts/real_ct/real_ct_scores_32.json`

The companion notebook is:

```text
notebooks/real_ct_msd_lung_gradcam.ipynb
```

The default deck grid renders three representative CT patches for readability.
Use `uv run gradcam-repro demo --samples 6` when you want the denser six-volume
audit grid.

Current deck outputs:

- method grid: `artifacts/figures/method_grid.png`
- class-discriminability check: `artifacts/figures/class_discriminability.png`
- figure provenance manifest: `artifacts/figures/manifest.json`
- real CT checkpoint: `artifacts/real_ct/real_ct_presence_cnn_32.pt`
- real CT attribution scores: `artifacts/real_ct/real_ct_scores_32.json`
- editable deck: `artifacts/deck/gradcam-editable.pptx`
- deck contact sheet: `artifacts/deck/gradcam-editable-contact-sheet.png`

The score command reports:

- `mass_in_gt`: share of normalized attribution mass inside the `7 x 7 x 7` GT mask
- `inside_outside_ratio`: mean heat inside the GT mask divided by mean heat outside
- `pointing_acc`: whether the heatmap peak lands inside the GT mask

## Figure Provenance Policy

Deck heatmaps and attribution visualizations must come from experiment code, not
from hand-drawn mockups or copied PPT screenshots. The `demo` command writes
`artifacts/figures/manifest.json` with the checkpoint hash, generation
parameters, and figure hashes. `uv run gradcam-repro deck` refuses to build if
the embedded experiment figures are not listed in that manifest.

## Web Export

`uv run gradcam-repro web-export` renders a static data bundle (CT slices,
attribution overlays, activation summaries, and benchmark metrics) for a
browser-based frontend:

```bash
uv run gradcam-repro web-export \
  --data artifacts/real_ct/msd_lung_presence_32.pt \
  --checkpoint artifacts/real_ct/real_ct_presence_cnn_32.pt \
  --out web/public/data --samples 3 --device cpu
```

Use `--device cpu` on machines where the Torch build lacks a 3D-conv kernel
for the accelerator (e.g. Apple MPS).

The bundle is fully regenerated from the checkpoint on every run — the
top-level JSON files are overwritten and `examples/` is fully cleared and
rebuilt from scratch, so no stale example directories survive a re-run with
fewer samples or methods — and hashed in `web/public/data/manifest.json`,
extending the figure-manifest provenance policy above to the web bundle. See
`web/DATA_CONTRACT.md` for the frozen schema (`gradcam-repro.web-bundle.v1`)
that the Astro frontend targets.

## Legacy Synthetic Experiment Shape

The synthetic dataset matches the deck's setup:

- input: `1 x 24 x 24 x 24`
- label: class `0` when the bright 3-axis cross is in the left half, class `1`
  when it is in the right half
- target: 1-voxel-wide 3D cross with three `7`-voxel arms along x, y, and z
- distractors: random 3D box/cuboid distractors, `3` to `6` voxels per side
- ground truth: `7 x 7 x 7` cube around the target cross
- training split: `1200`
- validation split: `200`
- test split: `120`
- optimizer: Adam, `lr=3e-3`, `weight_decay=1e-3`, `batch_size=16`
- early stop: first validation check with `val_acc >= 0.95`
- validation cadence: every `10` optimizer steps by default
- CAM tap point: `stage2`, shape `16 x 6 x 6 x 6`
- occlusion fill: per-volume mean, not zero, to avoid introducing artificial
  black-cube edge evidence

The model architecture preserves the PPT's stage2 CAM tap point:

```text
input 1x24x24x24 -> stage1 8x12x12x12 -> stage2 16x6x6x6 -> stage3 32x6x6x6 -> spatial evidence head -> 2 logits
```

One implementation note: a strictly translation-invariant `conv + global average
pool` head cannot learn the deck's left/right-position label. This scaffold uses
a small non-negative spatial evidence head and mean-pools evidence over the left
and right 3D halves. That keeps the stage2 attribution target inspectable while
making the classifier actually learnable.

## Notes

This is a local CPU/MPS-friendly reproduction scaffold. It does not require
external datasets. The generated figures are intended to reproduce the deck's
qualitative comparisons rather than exactly matching the embedded screenshots.

The editable deck is maintained as code under `deck/`. The source PPT was
image-only, so the sustainable workflow is to edit slide modules and source
notes, rebuild the deck, inspect the previews, then iterate.
