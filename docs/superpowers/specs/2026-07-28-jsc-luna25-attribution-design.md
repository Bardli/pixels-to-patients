# Design: replace the MSD backend with JSC + LUNA25

Date: 2026-07-28
Status: design agreed, implementation pending
Supersedes: the MSD `Task06_Lung` real-CT path (`src/gradcam_repro/real_ct.py`)
Rollback: `git checkout pre-jsc-luna25-swap` (tag points at commit `982cc86`)

## Goal

Re-run the seven-method attribution comparison against a real published
medical model instead of the project's hand-rolled toy CNN:

- **model** — [bowang-lab/JSC](https://github.com/bowang-lab/JSC), the
  FLARE26 AutoMSC joint segmentation + classification baseline
- **data** — `FLARE-MedFM/FLARE-AutoMSC` `Dataset005_LUNA25`, lung nodule
  crops from screening CT with malignancy labels

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Repo structure | **In-place replacement** of the MSD path | User's call. Cost accepted: loses the MSD baseline, the frozen `web-bundle.v1` contract, and the existing deck's provenance chain. Mitigated by the `pre-jsc-luna25-swap` tag. |
| Backbone | **PlainConvUNet** (published `nnUNetPlans` weights) | A ResEncM run needs 250k iterations from scratch on an H100; no ResEnc checkpoint exists for any AutoMSC dataset. ResEncM is deferred to its own project. |
| Compute | **Local, Apple MPS** | Measured 0.673 s/forward vs 1.264 s on CPU, numerically identical. |
| Case count | **4 cases, no selection** | First four entries of fold 3's val list, in `splits_final.json` order. |

Synthetic toy commands (`train` / `demo` / `score`) are **kept** — they are
independent of the MSD path and cost nothing to retain.

## Verified facts

All verified in-session on 2026-07-28; evidence in parentheses.

### Model

`SegmentationNetworkFusionClassificationHead` wraps the nnU-Net seg network
(`nnunetv2/training/nnUNetTrainer/nnUNetCLSTrainer.py:281-358`):

```
PlainConvUNet.encoder(x) -> skips
        |-> PlainConvUNet.decoder(skips) -> seg_output      (classification ignores this)
        `-> FPN(skips[-3], skips[-2], skips[-1])
              -> conv_block  (Conv3d-BN-ReLU x2, 320 -> 640)
              -> AdaptiveAvgPool3d(1,1,1)
              -> Linear(640,320) -> LeakyReLU -> Dropout(0.5) -> Linear(320,1)
```

`forward()` returns the tuple `(seg_output, cls_logits)`. Binary tasks get a
**single logit** (`cls_head_output = cls_class_num if cls_class_num > 2 else 1`,
L372).

Architecture confirmed against the published fold-3 weights:

| Derived | Weight tensor | Shape |
|---|---|---|
| FPN in = `features_per_stage[-3:]` = [256,320,320] | `feature_fusion_block.conv1x1_{1,2,3}.weight` | (320,256,1,1,1) / (320,320,1,1,1) x2 |
| FPN upsample stride [2,2,2] x2 | `feature_fusion_block.deconv{1,2}.weight` | (320,320,2,2,2) |
| conv_block 320 -> 640 -> 640 | `conv_block.{0,3}.weight` | (640,320,3,3,3) / (640,640,3,3,3) |
| head 640 -> 320 -> 1 | `classifier.{0,3}.weight` | (320,640) / **(1,320)** |

`load_state_dict(..., strict=True)` reports *All keys matched successfully*.
The 621 tensors cover 329 unique parameters: `self.encoder = seg_network.encoder`
and the same for `decoder` register aliased submodules, so `encoder.*` /
`decoder.*` appear both standalone and under `seg_network.*`. This resolves
automatically on load and needs no special handling.

### Checkpoint

`cyyu96/AutoMSC-Baselines/Dataset005_LUNA25/nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth`

- 443,676,001 bytes, `sha256 33ad8109aafd5f0b6d6c82a7f6911fa4167c36e44c9c4e50bed3c19aef25968f`
- not gated
- `init_args` embeds the full `plans` and `dataset_json`, so the file is
  self-contained; the separately downloaded `plans.json` matches field-for-field
- `trainer_name = nnUNetCLSTrainerMTL`, `fold = 3`, `cls_class_num = 2`,
  `current_epoch = 766 / 1000`, `inference_allowed_mirroring_axes = (0,1,2)`
- **Safe load recipe** — `weights_only=True` works after allowlisting six numpy
  globals: `numpy._core.multiarray.scalar`, `numpy.dtype`,
  `numpy.dtypes.{Float32,Float64,Int64,Int32}DType`. Do **not** fall back to
  `weights_only=False`.

FLARE's own `FLARE-AutoMSC-Val-Baseline` repo ships only Dataset007/072; the
LUNA25 weights exist solely in the author's personal repo. They are a published
training baseline, not the official validation docker baseline.

### Plans (3d_fullres)

`patch_size [64,128,128]`, `spacing [2.0, 0.6640620231628418, 0.6640620231628418]`,
`batch_size 4`, `CTNormalization`, `network_class_name =
dynamic_network_architectures.architectures.unet.PlainConvUNet`,
`n_stages 6`, `features_per_stage [32,64,128,256,320,320]`,
`strides [[1,1,1],[1,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]`,
`kernel_sizes[0] = [1,3,3]`, `InstanceNorm3d`, `LeakyReLU(inplace=True)`,
no dropout.

The first two stages are anisotropic because z spacing (2.0 mm) is ~3x the
in-plane spacing (0.664 mm).

### Dataset

6132 imagesTr + 6132 labelsTr, 8.38 GB, single CT channel,
seg labels `{background:0, nodule:1}`, classification `malignancy
{0:Benign, 1:Malignant}`, CC BY 4.0. HF access is `gated: "manual"`.

- Class balance **Benign 5,589 / Malignant 543 = 10.3 : 1**
- `splits_final.json` folds are **not** equal-sized (val 1118/1285/1214/**1289**/1226),
  so it is a custom stratified split, **not** nnU-Net's default `KFold`
- fold 3: train 4843 / val 1289, of which 104 Malignant (8.1%)

**Gotcha — the gated `dataset.json` carries the wrong dataset name.** The gated
`Dataset005_LUNA25/dataset.json` sets `"name": "Dataset009_LUNA25"`; the
checkpoint's embedded copy says `Dataset005_LUNA25`. The two are byte-identical
apart from that line. nnU-Net resolves paths through this field
(`maybe_convert_to_dataset_name`), so the pipeline must use the **checkpoint's
embedded `dataset_json`**, never the gated file. (It also confirms dataset
identity: same labels, same `classification_labels`, same `numTraining` 6132,
same reference.)

`dataset_fingerprint.json` records 6132 cases with `shapes_after_crop` all equal
to `(64,128,128)` and 782 distinct spacings (z from 0.5 to 3.2 mm). Because
`dataset.json` has no `dataset` key, nnU-Net enumerates cases through
`get_identifiers_from_splitted_dataset_folder`, which returns `np.unique(...)`
(**sorted**); the fingerprint arrays therefore align index-wise with the sorted
case-ID list. This alignment was confirmed empirically: for all four selected
cases the on-disk NIfTI shape and spacing match the fingerprint entry exactly.

## Architecture of the new pipeline

### M1 — model loader (`src/gradcam_repro/jsc/model.py`)

1. `torch.load(..., weights_only=True)` with the six-global allowlist
2. `PlansManager(ck["init_args"]["plans"])` -> `ConfigurationManager`
3. `nnUNetCLSTrainerMTL.build_network_architecture(...)` with
   `emb_dim = features_per_stage[-1] = 320` and `cls_class_num = 1`
   (static method — do **not** instantiate the trainer, whose `initialize()`
   calls `wandb.init()` at L399)
4. `load_state_dict(strict=True)`; `net.decoder.deep_supervision = False`; `eval()`
5. **Disable every `inplace` activation** (26 modules) before registering hooks —
   in-place ops corrupt guided-backprop gradient surgery, and the CAM tap point
   (`conv_block[-1]`) is itself a `ReLU(inplace=True)`
6. Expose:
   - `logits(x)` — adapter returning only `cls_logits`
   - `logits_and_seg(x)` — both outputs, for the three-way comparison figure
   - `cls_only(x)` — encoder + FPN + conv_block + GAP + classifier, **skipping
     the decoder**. Exact, not an approximation: the decoder output feeds only
     `seg_output`. Two benchmark runs measured 3.0x and 1.5x speedups on CPU;
     the spread is machine contention, so treat the magnitude as "meaningfully
     faster, exact factor unmeasured". The MPS path uses it unconditionally.
   - `cam_target` — the `conv_block` module

### M2 — data layer (`src/gradcam_repro/jsc/data.py`)

Preprocessing runs through nnU-Net's own preprocessor (plans `CTNormalization`
plus `foreground_intensity_properties_per_channel`, resample to
`[2.0, 0.664, 0.664]`). Hand-rolled normalisation is forbidden: attributions
computed on out-of-distribution inputs are meaningless.

**Input-shape constraint (measured).** The FPN's `ConvTranspose3d` doubles
exactly while the encoder's stride-2 convolutions round up, so the size chain
must align at the last three levels. Non-conforming shapes raise
`RuntimeError: The size of tensor a (N) must match the size of tensor b (N+1)`.

| Input (z,y,x) | Result | CAM tap |
|---|---|---|
| (64,128,128) | OK | (16,16,16) |
| (64,126,126) | OK (coincidental alignment) | (16,16,16) |
| (64,160,160) | OK | (16,20,20) |
| (80,128,128) | OK | (20,16,16) |
| (64,105,105) / (64,113,113) / (64,135,135) / (64,136,136) / (64,144,144) | **CRASH** | — |

Every case is therefore brought to exactly `(64,128,128)`:

| # | case_id | label | resampled | action |
|---|---|---|---|---|
| 0 | `100012_1_19990102` | Malignant | (64,113,113) | pad |
| 1 | `100012_1_20000102` | Malignant | (64,105,105) | pad |
| 2 | `100289_4_20010102` | Benign | (64,126,126) | pad |
| 3 | `100438_1_19990102` | Benign | (64,135,135) | **centre-crop** |

`artifacts/jsc/case_index.csv` caches `case_id`, original shape/spacing,
resampled shape and a `single_patch` flag for all 6132 cases.

### M3 — attribution layer (`attribution.py`, rewritten)

CAM tap point: output of `conv_block`, shape `(1, 640, 16, 16, 16)` for a
`[64,128,128]` patch (z/4, y/8, x/8), trilinearly upsampled to the patch.

| Method | Adaptation |
|---|---|
| notGradCAM, Grad-CAM, LayerCAM, Integrated Grad-CAM | new tap point; target scalar is the single logit |
| Guided Grad-CAM | encoder activations are `LeakyReLU`, which has no standard guided-backprop rule. Hook it with the ReLU rule (clamp negative gradients) and state the approximation in the figure caption. |
| Integrated Gradients | 16 steps; baseline is zero in CTNormalization space |
| Occlusion sensitivity | **full-patch** sliding window, 16^3 occluder, stride 8 -> 1575 positions; occluded value is the per-volume mean (keeps the existing anti-black-box-artefact policy); runs through `cls_only` under `inference_mode` |

Occluder note: 16^3 in voxels is physically 32 mm (z) x 10.4 mm x 10.4 mm — 3x
elongated in z. The figure caption must say so, or the occluder must be changed
to a physically isotropic 8x24x24 (2940 positions, 1.9x cost).

### M4 — scoring (`evaluate.py`, rewritten)

Scored against the **real nodule mask**, not a 7^3 cube. Measured mask sizes:

| case | voxels | % of volume | mm^3 | equiv. diameter |
|---|---|---|---|---|
| `100012_1_19990102` | 760 | 0.072% | 521.9 | 9.99 mm |
| `100012_1_20000102` | 5225 | 0.498% | 3125.3 | 18.14 mm |
| `100289_4_20010102` | 241 | 0.023% | 205.1 | 7.32 mm |
| `100438_1_19990102` | 416 | 0.040% | 406.8 | 9.19 mm |

The old toy ground truth (7^3 cube in a 32^3 patch) covered 1.05% of the volume;
the real masks cover 2.1x to 45.6x less. Raw `mass_in_gt` therefore collapses to
values around 0.00x **and stops being comparable across cases** — the 0.023%
case cannot reach the same mass as the 0.498% case even with identical
attribution quality.

Metrics:

- `enrichment` = `mass_in_gt / gt_volume_fraction` — **replaces raw
  `mass_in_gt` as the headline number.** A uniform heatmap scores exactly 1.0;
  above 1.0 means the heat really is concentrated on the lesion. Comparable
  across cases.
- `mass_in_gt` — retained as a raw diagnostic, not headlined
- `inside_outside_ratio` — unchanged, already a ratio
- `pointing_acc` — unchanged
- `cam_seg_agreement` — **new.** Soft-IoU between the normalised heatmap and the
  model's *own predicted* nodule mask. JSC's segmentation head gives an
  independent read of where the model localises the lesion, so attribution
  faithfulness can be cross-checked against the model's explicit output. The
  toy CNN could not do this.

### M5 — figures and web bundle

- Method grid: CT / GT mask / **predicted mask** / seven heatmaps
- `class_discriminability.png` is replaced by a **Benign vs Malignant case
  pairing**. With one logit, per-case class flipping is degenerate: the
  "class 0" Grad-CAM is the negation of the "class 1" Grad-CAM.
- **New figure — longitudinal stability.** `100012_1` is the same patient at
  1999-01-02 and 2000-01-02; the nodule grows 521.9 -> 3125.3 mm^3 (5.99x,
  equivalent diameter 9.99 -> 18.14 mm), both labelled Malignant. Does the
  attribution track the growth? Caveat to state: the two scans are not
  independent samples.
- `web/DATA_CONTRACT.md` bumps to `gradcam-repro.web-bundle.v2`; the seven
  `tests/test_web_export_*.py` files move with it.

Axial slices need no aspect correction: the displayed y-x plane is isotropic at
0.664 mm. The 2.0 mm z spacing only affects the CAM tap resolution (z/4 vs
y,x/8), not pixel aspect.

## Declared deviations

These must appear in the spec, the figure captions, and the web bundle
provenance — silent deviations would make the figures misleading.

1. **Case 3 is centre-cropped** from (64,135,135) to (64,128,128), discarding
   3–4 voxels per in-plane edge: 85 mm of an 89.5 mm field is kept, the nodule
   is centred by construction and is 9.19 mm across. Alternatives were 2x2
   sliding-window aggregation (4x cost, plus CAM stitching seams) or padding to
   (64,160,160) (3.4x slower per forward at 2.274 s, and shifts the GAP extent
   and InstanceNorm statistics away from training conditions). Cropping also
   keeps all four cases at one size, so the method grid is visually uniform.
2. **Occlusion runs `cls_only`**, skipping the decoder. Exact for the
   classification logit.
3. **Guided backprop treats LeakyReLU as ReLU.**
4. **Attribution runs on MPS.** Verified equivalent to CPU: logit
   `-2.762353` (CPU) vs `-2.762354` (MPS), absolute difference `4.8e-07`.
5. **No AUROC reproduction.** Dropped at the user's direction in favour of four
   cases. The available anchors, unused: README fold-3 AUROC `0.884` and the
   checkpoint's own embedded `val_auc[-1] = 0.8875`.

## Verification strategy

Every check raises rather than degrading silently.

1. `load_state_dict(strict=True)` must report all keys matched — **passed**
2. `conv_block` output must equal `(1, 640, 16, 16, 16)` — **passed**
3. Every case must reach exactly `(64,128,128)` before the forward pass
4. On-disk NIfTI shape/spacing must match the fingerprint entry — **passed for
   all four cases**
5. Nodule masks must be non-empty — **passed** (241–5225 voxels)
6. Each attribution map: shape equals the patch, no NaN, not all-zero
7. `enrichment` for a uniform heatmap must equal 1.0 within tolerance
   (self-test of the metric)

## Compute budget (measured, MPS, 1575 occlusion positions)

| Stage | Per case |
|---|---|
| Grad-CAM / LayerCAM / Guided / notGradCAM | ~1 s each |
| Integrated Gradients (16 steps), Integrated Grad-CAM | ~16 s each |
| Occlusion, full patch | **17.4–17.7 min** |
| **Total** | **~18 min** |

Four cases: **~72 min**, 98% of it occlusion.

Measurement notes. The benchmark was run twice; the two runs overlapped in time,
so the **CPU** figures are contaminated by contention for the 8 threads and
16 GB (`cls_only` measured 1.26 s in one run and 6.66 s in the other) and should
not be quoted. The **MPS batch-1 figure is stable across both runs**
(0.673 and 0.662 s/position) and is the number the budget rests on.

Batching brings no benefit on MPS: in the uncontended run, batch 1 / 2 / 4 / 8
gave 0.662 / 0.669 / 0.678 / 0.785 s per position. Batch 1 is therefore the
default — not because larger batches are catastrophic, but because they are
flat-to-slightly-worse and use more memory.

Machine: MacBookPro18,3, Apple M1 Pro, 10 cores, 16 GB, torch 2.12.0,
`torch.get_num_threads() == 8`.

## Environment

Fetch commands, SHA-256 digests for every external artifact, and the full list of
install gotchas live in [`docs/jsc-luna25-sources.md`](../../jsc-luna25-sources.md).
Summary:

- JSC pinned at commit `49511ef01c414014afb7e7a3265d820544bf93cc` (2026-03-27),
  cloned to `third_party/JSC` (gitignored), installed with
  `uv pip install -e third_party/JSC` so `pyproject.toml` and `uv.lock` stay
  untouched. **Caveat: `uv sync` will prune it — reinstall after any sync.**
- Two local patches to the clone:
  - `pyproject.toml` needs `[tool.setuptools.packages.find] include = ["nnunetv2*"]`;
    without it setuptools refuses to build (`Multiple top-level packages
    discovered in a flat-layout: ['configs', 'nnunetv2']`).
  - `torchmetrics` is imported by `nnUNetCLSTrainer.py:73` but missing from
    JSC's dependency list; installed separately (1.9.0).
- JSC's README says to edit `nnunetv2/paths.py`, but the fork reads the
  `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` environment variables
  and only warns when they are unset. Set the env vars instead.

## Out of scope

- Training a ResEncM (or any) backbone — separate project, needs Fir
- Reproducing the published 5-fold metrics
- Any use of folds 0, 1, 2, 4
- Full-dataset inference (8.38 GB; only 5.6 MB of images is downloaded)
