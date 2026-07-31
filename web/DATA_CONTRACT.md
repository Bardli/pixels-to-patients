# Web Export Data Contract

> **v2 (2026-07-29).** The bundle is now written by
> `python scripts/jsc_web_export.py` from the **published JSC / LUNA25** model,
> replacing the toy `RealCtCNN` on MSD `Task06_Lung`. Schema string is
> `gradcam-repro.web-bundle.v2`. Changes forced by the model swap:
>
> | Change | Why |
> |---|---|
> | `logits` is length **1**, not 2; `pred_label` is `logit > 0`, not `argmax` | JSC's head emits a single logit for binary tasks |
> | `model_graph.json` nodes are `input, encoder, fpn, conv_block, pool, classifier, logits, seg_head`; `cam_tap` is `"conv_block"` | Real PlainConvUNet + FPN path, not 3 toy conv stages |
> | New `examples/<id>/pred_mask_slice.png` | JSC segments as well as classifies, so its own predicted mask is an independent check on localisation. The toy CNN had no such output. |
> | New metric `enrichment` = `mass_in_gt / gt_volume_fraction`; **replaces `mass_in_gt` as the headline** | The nodules here span 0.006% to 2.2% of the volume, so raw mass collapses to ~0.00x and stops being comparable between a small and a large lesion. A uniform heat-map scores exactly 1.0. |
> | New `meta.json` fields: `prob_malignant`, `gt_voxels`, `gt_volume_fraction`, `pred_mask_voxels`, `seg_dice`, `outcome` | Needed to select and label success/failure exemplars |
> | New `manifest.json` fields: `model`, `deviations` | Records the published checkpoint's identity and every deliberate departure from the official inference script |
> | Slices are **128×128** | Patch size is `(64,128,128)`, not 32³ |
> | `activations.json` entries gain `real_name`, `all_channels`, `n_silent`, `n_channels`, `frac_zero` | The `layer` ids stay `stage1/stage2/stage3` as an interface contract with `web/js/scene3d.js`, but they now tap `encoder.stages[2]`, `conv_block`, and `encoder.stages[5]`. `real_name` carries the truth and is what the UI labels. `all_channels` lists every channel's mean/max (no slice payload) so the baseline page can show the **whole** set being averaged (R3.4) without shipping 640 slice arrays. |
>
> Sections below describe v1 field-by-field and remain accurate except where
> the table above overrides them.

This document freezes the schema of the static data bundle written into
`web/public/data/`. The three top-level index files — `manifest.json`,
`model_graph.json`, and `benchmark.json` — each carry
`"schema": "gradcam-repro.web-bundle.v2"`. The
per-example files (`meta.json`, `ct_slice.json`, `attributions/<method>.json`,
`activations.json`) carry no `schema` key; `activations.json` in particular is
a bare JSON array, not an object. The bundle is fully static (plain files, no
server) so it can be committed, hashed, and served directly by a static-site
frontend.

This contract is frozen; the Astro frontend (Plan 2) targets exactly these
shapes.

## Directory layout

```
web/public/data/
├── manifest.json
├── model_graph.json
├── benchmark.json
└── examples/<id>/
    ├── meta.json
    ├── ct_slice.png        # grayscale axial slice at z_slice
    ├── ct_slice.json       # slice payload (hover values)
    ├── mask_slice.png      # CT with GT tumour mask in green
    ├── pred_mask_slice.png # CT with the model's OWN predicted mask in orange (v2)
    ├── attributions/<method>.png    # turbo heatmap overlay
    ├── attributions/<method>.json   # slice payload (already-normalised heatmap)
    └── activations.json
```

`<id>` is an example directory name of the form `{index:02d}_{case_id}`, e.g.
`00_lung_034_pos0`. `<method>` is one of the attribution method keys in
`manifest.json`'s `methods` list (e.g. `gradcam`, `layercam`,
`integrated_gradcam`).

## manifest.json

Bundle-level index and provenance. Written with `indent=2` (the only
pretty-printed file in the bundle).

| Field | Type | Description |
|---|---|---|
| `schema` | string | Always `"gradcam-repro.web-bundle.v1"`. |
| `generated_at` | string | ISO-8601 UTC timestamp passed in by the CLI at export time. |
| `generated_by` | string | Fixed literal `"uv run scripts/jsc_web_export.py"` (not the actual invoked argv). |
| `checkpoint` | object | `{path: string, sha256: string}` — checkpoint path exactly as passed on the CLI (may be relative), and its full-file SHA-256 hex digest. |
| `data_cache` | object | `{path: string, sha256: string}` — same shape, for the preprocessed dataset cache. |
| `uv_lock_sha256` | string \| null | SHA-256 hex digest of `uv.lock` resolved relative to the process CWD; `null` if that file does not exist. |
| `git` | object | `{commit: string \| null, dirty: bool}` from `git rev-parse HEAD` / `git status --porcelain` at export time; `{null, false}` if git is unavailable or the call fails. |
| `methods` | list[string] | Attribution method keys included in this bundle, in the order used consistently across `benchmark.json` and every `examples/<id>/`. |
| `examples` | list[string] | Example ids, in generation order; matches the `examples/<id>/` subdirectory names. |
| `num_examples` | int | `len(examples)`. |
| `policy` | string | Fixed provenance statement: `"Every displayed figure and number is generated by web-export and hashed here."` |

```json
{
  "schema": "gradcam-repro.web-bundle.v1",
  "generated_at": "2026-07-09T00:00:00+00:00",
  "generated_by": "uv run scripts/jsc_web_export.py",
  "checkpoint": {
    "path": "artifacts/real_ct/real_ct_presence_cnn_32.pt",
    "sha256": "9f2b7a5c1e6d4f803a2b5c7e9f1a3d6c8b0e2f4a6c8e0b2d4f6a8c0e2f4a6c81"
  },
  "data_cache": {
    "path": "artifacts/real_ct/msd_lung_presence_32.pt",
    "sha256": "3ac0d8e1f4b7c2a5e8d1f4b7c0a3e6d9f2b5c8e1f4b7c0a3e6d9f2b5c8e1f4b7"
  },
  "uv_lock_sha256": "b1d4f7c0a3e6d9f2b5c8e1f4b7c0a3e6d9f2b5c8e1f4b7c0a3e6d9f2b5c8e1f4",
  "git": { "commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0", "dirty": false },
  "methods": ["gradcam", "layercam", "integrated_gradcam"],
  "examples": ["00_lung_034_pos0", "01_lung_034_neg1", "02_lung_057_pos0"],
  "num_examples": 3,
  "policy": "Every displayed figure and number is generated by web-export and hashed here."
}
```

## model_graph.json

Static description of the model's forward path, independent of any single
example. `nodes` always has exactly 7 entries in this fixed order:
`input, stage1, stage2, stage3, pool, classifier, logits`.

| Field | Type | Description |
|---|---|---|
| `schema` | string | `"gradcam-repro.web-bundle.v1"`. |
| `input_shape` | `[int, int, int, int]` | `[channels, D, H, W]` of one input volume — channels is always `1`; **no batch dimension**. |
| `cam_tap` | string | Always `"stage2"` — the node id that CAM-family methods read activations from. |
| `nodes` | list[object] | See below, 7 entries, fixed order. |

Each `nodes[i]` object:

| Field | Type | Description |
|---|---|---|
| `id` | string | One of `input`, `stage1`, `stage2`, `stage3`, `pool`, `classifier`, `logits`. |
| `name` | string | Human-readable label, e.g. `"Stage 2 conv block (CAM tap)"`. |
| `type` | string | One of `input`, `conv`, `pool`, `linear`, `output`. |
| `out_shape` | list[int] | Output shape with the batch dimension stripped. Conv stages: `[C, D, H, W]`. `pool`: `[C, 1, 1, 1]`. `classifier`/`logits`: `[num_classes]`. `input`: same 4-element form as top-level `input_shape`. |
| `param_count` | int | Trainable parameter count owned by that node's module. `0` for `input`, `pool`, and `logits` (they own no parameters). |
| `cam_tap` | bool | `true` only for the `stage2` node. |

```json
{
  "schema": "gradcam-repro.web-bundle.v1",
  "input_shape": [1, 32, 32, 32],
  "cam_tap": "stage2",
  "nodes": [
    { "id": "input", "name": "CT patch", "type": "input", "out_shape": [1, 32, 32, 32], "param_count": 0, "cam_tap": false },
    { "id": "stage1", "name": "Stage 1 conv block", "type": "conv", "out_shape": [8, 16, 16, 16], "param_count": 1960, "cam_tap": false },
    { "id": "stage2", "name": "Stage 2 conv block (CAM tap)", "type": "conv", "out_shape": [16, 8, 8, 8], "param_count": 10400, "cam_tap": true },
    { "id": "stage3", "name": "Stage 3 conv", "type": "conv", "out_shape": [32, 8, 8, 8], "param_count": 13856, "cam_tap": false },
    { "id": "pool", "name": "Global avg pool", "type": "pool", "out_shape": [32, 1, 1, 1], "param_count": 0, "cam_tap": false },
    { "id": "classifier", "name": "Linear classifier", "type": "linear", "out_shape": [2], "param_count": 66, "cam_tap": false },
    { "id": "logits", "name": "Class logits", "type": "output", "out_shape": [2], "param_count": 0, "cam_tap": false }
  ]
}
```

(Shapes above assume a 32³ input patch; `out_shape` scales with whatever
`--size` was used to preprocess the cache. `param_count` values are fixed by
the architecture and do not depend on input size.)

## benchmark.json

Aggregated and per-example attribution-quality metrics, one row per method.
The three metric keys are always `mass_in_gt`, `inside_outside_ratio`,
`pointing_acc`:

- `mass_in_gt` (float, `[0, 1]`): share of total attribution mass that falls
  inside the GT tumour mask.
- `inside_outside_ratio` (float, `>= 0`, unbounded above): mean heat inside
  the mask divided by mean heat outside it. `0.0` when the mask is empty
  (non-tumour example).
- `pointing_acc` (float): **per-example**, this is a hard `0.0`/`1.0` hit
  indicator (whether the single heatmap peak voxel lands inside the mask).
  In `aggregate`, it is the mean of that indicator across examples, i.e. a
  true fraction in `[0, 1]`.

| Field | Type | Description |
|---|---|---|
| `schema` | string | `"gradcam-repro.web-bundle.v1"`. |
| `methods` | list[string] | Same list and order as `manifest.json.methods`. |
| `examples` | list[string] | Same list and order as `manifest.json.examples`. |
| `per_example` | object | `{method: {example_id: {mass_in_gt, inside_outside_ratio, pointing_acc}}}`. |
| `aggregate` | object | `{method: {mass_in_gt, inside_outside_ratio, pointing_acc}}` — mean of each metric across all examples for that method (`0.0` for all three if there are no examples). |

```json
{
  "schema": "gradcam-repro.web-bundle.v1",
  "methods": ["gradcam", "layercam"],
  "examples": ["00_lung_034_pos0", "01_lung_034_neg1"],
  "per_example": {
    "gradcam": {
      "00_lung_034_pos0": { "mass_in_gt": 0.62, "inside_outside_ratio": 3.4, "pointing_acc": 1.0 },
      "01_lung_034_neg1": { "mass_in_gt": 0.0, "inside_outside_ratio": 0.0, "pointing_acc": 0.0 }
    },
    "layercam": {
      "00_lung_034_pos0": { "mass_in_gt": 0.58, "inside_outside_ratio": 2.9, "pointing_acc": 1.0 },
      "01_lung_034_neg1": { "mass_in_gt": 0.0, "inside_outside_ratio": 0.0, "pointing_acc": 0.0 }
    }
  },
  "aggregate": {
    "gradcam": { "mass_in_gt": 0.31, "inside_outside_ratio": 1.7, "pointing_acc": 0.5 },
    "layercam": { "mass_in_gt": 0.29, "inside_outside_ratio": 1.45, "pointing_acc": 0.5 }
  }
}
```

## examples/&lt;id&gt;/meta.json

Per-example metadata: the sample's identity, model prediction, and its own
copy of the 3 metrics (identical values to `benchmark.json.per_example.*[example_id]`).

| Field | Type | Description |
|---|---|---|
| `example_id` | string | Matches the containing directory name. |
| `case_id` | string | Source case identifier from the preprocessed cache (e.g. `lung_034_pos0`). |
| `true_label` | int | Ground-truth class (`0` = non-tumour, `1` = tumour-present). |
| `pred_label` | int | `argmax` of `logits`. |
| `logits` | list[float] | Raw (pre-softmax) model outputs, length 2. |
| `z_slice` | int | Axial index used for `ct_slice.*`, `mask_slice.png`, and `attributions/*` (the input-resolution slice, not the per-layer `feature_z` used inside `activations.json`). |
| `input_shape` | `[int, int, int]` | `[D, H, W]` of this sample's volume — **no channel dimension** (contrast with `model_graph.json.input_shape`, which is `[channels, D, H, W]`). |
| `methods` | list[string] | Methods rendered for this example (same as `manifest.json.methods`). |
| `metrics` | object | `{method: {mass_in_gt, inside_outside_ratio, pointing_acc}}`, same metric definitions as `benchmark.json`. |

```json
{
  "example_id": "00_lung_034_pos0",
  "case_id": "lung_034_pos0",
  "true_label": 1,
  "pred_label": 1,
  "logits": [-1.42, 1.87],
  "z_slice": 16,
  "input_shape": [32, 32, 32],
  "methods": ["gradcam", "layercam"],
  "metrics": {
    "gradcam": { "mass_in_gt": 0.62, "inside_outside_ratio": 3.4, "pointing_acc": 1.0 },
    "layercam": { "mass_in_gt": 0.58, "inside_outside_ratio": 2.9, "pointing_acc": 1.0 }
  }
}
```

## Slice payload (`ct_slice.json`, `attributions/<method>.json`)

Both `examples/<id>/ct_slice.json` and every
`examples/<id>/attributions/<method>.json` share this exact shape — a single
quantised 2D slice for hover/inspection UIs.

| Field | Type | Description |
|---|---|---|
| `shape` | `[int, int]` | `[H, W]` of the slice. |
| `values` | list[int] | Flat **row-major** list of length `H * W`, quantised to `0`–`255`. |
| `vmin` | float | Pre-normalisation minimum of the raw 2D slice (for hover display), **not** `0`. |
| `vmax` | float | Pre-normalisation maximum of the raw 2D slice (for hover display), **not** `255`. |

See [Quantisation convention](#quantisation-convention) below for how
`values` relates to `vmin`/`vmax`.

```json
{
  "shape": [4, 4],
  "values": [12, 40, 40, 12, 40, 210, 230, 40, 40, 230, 210, 40, 12, 40, 40, 12],
  "vmin": 0.0004,
  "vmax": 0.982
}
```

(A real slice is `H × W` values, e.g. `32 × 32 = 1024` entries for a 32³
patch; the example above is truncated to 4×4 for readability. This example is
a `ct_slice.json`-style payload: `vmin`/`vmax` are in the cache's
normalized-intensity space (`[0, 1]`), not raw Hounsfield units — CT volumes
are HU-normalized to `[0, 1]` at preprocessing time, before this per-slice
payload is ever computed.)

## activations.json

**The one file whose JSON root is a bare array, not an object** — a list of
exactly 3 entries, in order `stage1`, `stage2`, `stage3`. In v2 those ids are an
interface contract only; the real tap points are `encoder.stages[2]`,
`conv_block` (the CAM tap), and `encoder.stages[5]`, reported per entry as
`real_name`.

Measured note: only the `conv_block` tap is sparse (~52–72% of activations
exactly zero, varying by case). The encoder taps sit before their activation, so
their `frac_zero` is 0.000 and essentially no channel is ever fully silent —
across the five shipped examples `n_silent` at the tap is 0 or 1 out of 640.
This is the opposite of the toy CNN's behaviour and the baseline page's prose
reflects it.

Each array entry:

| Field | Type | Description |
|---|---|---|
| `layer` | string | `"stage1"`, `"stage2"`, or `"stage3"`. |
| `feature_shape` | `[int, int, int, int]` | `[C, D, H, W]` of that layer's activation tensor — no batch dimension. |
| `feature_z` | int | Axial index into this layer's own (smaller) spatial resolution — independent from `meta.json.z_slice`, computed by rescaling the sample's z-index into this layer's depth. |
| `channels` | list[object] | Top `min(6, C)` channels by mean activation (descending), each `{index, slice, mean, max}` — see below. |
| `histogram` | object | `{bins: list[float] len 21, counts: list[int] len 20}` by default (20 bins) — computed over the entire 3D activation tensor for that layer (all channels, all voxels), not per-channel. |

Each `channels[i]` object:

| Field | Type | Description |
|---|---|---|
| `index` | int | Original channel index (0-based) in the layer's channel ordering — not a rank. |
| `slice` | object | Slice payload (see above) for `activation[0, index]` at `feature_z`. Quantised via its own 2D min/max, independent of `mean`/`max` below. |
| `mean` | float | Mean activation of the **full 3D channel volume** (not just the displayed slice). |
| `max` | float | Max activation of the **full 3D channel volume** (not just the displayed slice). |

```json
[
  {
    "layer": "stage1",
    "feature_shape": [8, 16, 16, 16],
    "feature_z": 8,
    "channels": [
      { "index": 3, "slice": { "shape": [16, 16], "values": ["... 256 ints ..."], "vmin": 0.0, "vmax": 4.7 }, "mean": 0.82, "max": 4.7 },
      { "index": 0, "slice": { "shape": [16, 16], "values": ["... 256 ints ..."], "vmin": 0.0, "vmax": 3.1 }, "mean": 0.41, "max": 3.1 }
    ],
    "histogram": { "bins": [0.0, 0.24, "...", 4.7], "counts": [512, 340, "...", 2] }
  },
  { "layer": "stage2", "feature_shape": [16, 8, 8, 8], "feature_z": 4, "channels": ["... up to 6 entries ..."], "histogram": { "bins": ["... 21 floats ..."], "counts": ["... 20 ints ..."] } },
  { "layer": "stage3", "feature_shape": [32, 8, 8, 8], "feature_z": 4, "channels": ["... up to 6 entries ..."], "histogram": { "bins": ["... 21 floats ..."], "counts": ["... 20 ints ..."] } }
]
```

(The `stage1` entry above is truncated to 2 of up to 6 `channels`, and the
histogram arrays are truncated with `"..."`; real values are plain numbers,
not strings — the `"..."` markers exist only in this documentation example.)

## attributions/&lt;method&gt;.decomp.json (v2)

One method's computation terms, so a page can step through the arithmetic rather
than stopping at the result. **A method with no exportable terms writes no
`.decomp.json` at all** — never an empty one, so the front end can branch on
existence.

| Field | Type | Description |
|---|---|---|
| `tap` | string | `"conv_block"` — the layer the CAM family reads. |
| `tap_shape` | `[int,int,int,int]` | `[640,16,16,16]` for a `(64,128,128)` patch. |
| `terms` | object | `{name: slice payload}`; see the per-method table below. |
| `channel` | int \| null | Which channel the single-channel terms belong to, chosen from the data. `null` for methods with no per-channel term. |
| `alpha` | float | That channel's pooled weight (CAM family only). |
| `alpha_all` | object | `{values, vmin, vmax, signed}` — every channel's weight. |
| `alpha_negative` | int | How many of those weights are negative. |
| `grad_is_constant` | bool | True when the gradient carries no spatial structure (see below). |
| `channel_means` | object | notGradCAM only: each channel's mean activation. |
| `intact` | float | Occlusion only: the intact probability, on MONAI's scale. |
| `path` | list[object] | Integration methods only: slice payloads each with an extra `frac`, ascending, last one the finished integral. |
| `steps` | int | Integration methods only: the step count actually used. |

Every slice payload here carries three fields beyond the base shape:
`signed` (centre a diverging scale at zero), `feature_z` (the slice index within
that term's own depth) and `depth` (that depth), because terms live at different
resolutions — the CAM family at the `16³` tap, Integrated Gradients and occlusion
at the `64×128×128` input.

**Slices are capped at 64×64** by block-mean. Four Integrated-Gradients path
checkpoints at full 128×128 cost 235 KB per case per method, an order of
magnitude over budget for something that renders into a 288 px canvas with
`image-rendering: pixelated`. Measured after the cap: 152 KB per case, ~760 KB
across the five exemplars.

| Method | Terms |
|---|---|
| `notgradcam` | `activation`, `relu` — **no gradient term**, deliberately: its absence is that page's argument |
| `gradcam` | `activation`, `gradient`, `weighted`, `summed`, `relu` |
| `layercam` | `activation`, `hadamard`, `relu` |
| `occlusion` | `drop`, `occluded` (+ `intact` scalar) |
| `integrated_gradients` | `path` only |
| `integrated_gradcam` | `path` only |
| `guided_gradcam` | none — no `.decomp.json` is written |

**Read `grad_is_constant` before designing a gradient panel.** On JSC the tap is
followed by `AdaptiveAvgPool3d`, so ∂y/∂A[k,z,y,x] = (1/N)·∂y/∂pooledₖ for every
voxel: the per-channel spatial std is exactly 0.0 across all 640 channels.
Two consequences the renderer must not paper over — `gradient` is one uniform
value rather than a texture, and `weighted` is `activation` times a scalar, so
after per-panel min-max it is pixel-identical to `activation` (measured max diff
6.0e-08). That is why `alpha_all` ships: the class-specific signal lives in the
spread of the weights across channels, not in any one panel's texture, and on the
lead case 415 of 640 are negative — the direct reason the closing ReLU can erase
part of the map.

## Quantisation convention

Slice `values` are always a flat row-major list of ints in `0`–`255`.
`vmin`/`vmax` are the pre-normalisation extremes of that specific 2D slice
(for hover display), computed **before** quantisation and **before** any
clamping:

- **CT slices** (`ct_slice.json`, and every `channels[i].slice` in
  `activations.json`): not pre-normalised. The slice is stretched by its own
  `(value - vmin) / (vmax - vmin)` before quantising to `0`–`255`.
- **Attribution slices** (`attributions/<method>.json`): already normalised
  to `[0, 1]` upstream by the attribution method (`normalize_map` bounds the
  full 3D heatmap to `[0, 1]` before any 2D slice is taken), so `vmin`/`vmax`
  for these slices are already within `[0, 1]`. The clamp to `[0, 1]` before
  quantising is a no-op safeguard, not a correction for out-of-range values.

In both cases `vmin`/`vmax` describe the *displayed slice only* — for
`activations.json` in particular, they are **not** the same as that entry's
`mean`/`max`, which are computed over the full 3D channel volume.

## Images (`*.png`)

`ct_slice.png` (grayscale), `mask_slice.png` (CT slice with the GT mask
blended in green), and `attributions/<method>.png` (CT slice blended with a
`turbo`-colormap heatmap) are final rendered raster images, meant for direct
`<img>` display. They carry no separate JSON contract; `ct_slice.json` and
`attributions/<method>.json` expose the same underlying values as raw,
hoverable numbers alongside the pre-rendered picture.

## Implementation notes

- `manifest.json` is written with `indent=2`; every other JSON file in the
  bundle is written compact (no indentation, no trailing newline).
- Every JSON file is UTF-8 text with no BOM.
- Re-running `web-export` fully regenerates the bundle: the three top-level
  JSON files (`manifest.json`, `model_graph.json`, `benchmark.json`) are
  overwritten, and `examples/` is fully cleared (`shutil.rmtree`) and rebuilt
  from scratch on every run. No stale per-example directories survive a
  re-run with fewer samples or methods — there is no incremental/merge
  behavior.
