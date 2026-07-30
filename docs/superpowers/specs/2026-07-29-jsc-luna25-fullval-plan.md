# Plan: JSC/LUNA25 attribution over the full fold-3 val set, with selected exemplars

Date: 2026-07-29
Status: proposed, awaiting go-ahead
Supersedes: the 4-case scope in
[`2026-07-28-jsc-luna25-attribution-design.md`](2026-07-28-jsc-luna25-attribution-design.md)
(that document's model/data/metric decisions stand; only scope, device, and the
compute-driven compromises change)
Rollback: tag `pre-jsc-luna25-swap` at commit `982cc86` — **does not exist on
this machine, Phase 0 recreates it**

## What changed since the 2026-07-28 design

Two things, and they pull in opposite directions.

**The machine got much bigger.** 3x RTX 6000 Ada (48 GB each, all idle), 56
cores, 502 GB RAM, CUDA 13.0. The prior design was written against a MacBook
Pro M1 Pro with 16 GB, and several of its decisions were consequences of that
laptop rather than of the model or the science.

**The scope got much bigger.** Full fold-3 val — 1289 cases — instead of the
first four, with success/failure exemplars *selected from measured results*
rather than fixed in advance.

The scope increase is the reason the hardware increase does not simply buy us
free time: per-case cost falls sharply, but case count rises 322x.

## Goal

1. Run the seven-method attribution comparison over all 1289 fold-3 val cases
   of `Dataset005_LUNA25` against the published JSC PlainConvUNet fold-3
   checkpoint.
2. Reproduce the published fold-3 AUROC (`0.884` README / `0.8875` embedded
   `val_auc[-1]`) as a correctness anchor — this becomes possible at full-val
   scale and un-drops deviation #5 of the prior design.
3. Select and present **success** and **failure** exemplars chosen on measured
   criteria, with the selection rule stated and the full distribution shown
   behind them.

## Decisions carried over unchanged

From the 2026-07-28 design; none of these are compute-bound, so scale does not
touch them.

- Model: `bowang-lab/JSC` `SegmentationNetworkFusionClassificationHead`, pinned
  commit `49511ef01c414014afb7e7a3265d820544bf93cc`
- Backbone: PlainConvUNet, published `nnUNetPlans` fold-3 weights. ResEncM
  stays deferred — no checkpoint exists for any AutoMSC dataset.
- Checkpoint: `cyyu96/AutoMSC-Baselines/.../fold_3/checkpoint_best.pth`,
  sha256 `33ad8109...5968f`, ungated
- Safe load: `weights_only=True` + the six-numpy-global allowlist. Never
  `weights_only=False`.
- **Use the checkpoint's embedded `dataset_json`, never the gated
  `dataset.json`** — the gated copy's `"name"` field wrongly says
  `Dataset009_LUNA25`, and nnU-Net resolves paths through that field.
- Preprocessing through nnU-Net's own preprocessor. Hand-rolled normalisation
  stays forbidden: attributions on out-of-distribution inputs are meaningless.
- Disable all in-place activations before hooking (in-place ops corrupt guided
  backprop, and the CAM tap is itself a `ReLU(inplace=True)`).
- CAM tap: output of `conv_block`, `(1, 640, 16, 16, 16)` for a `[64,128,128]`
  patch.
- Head emits **one logit**. Class 1 = Malignant, class 0 = Benign.
- `enrichment = mass_in_gt / gt_volume_fraction` is the headline metric;
  raw `mass_in_gt` is a retained diagnostic. `cam_seg_agreement` (soft-IoU vs
  the model's own predicted mask) is new and kept.
- Guided backprop treats `LeakyReLU` as `ReLU` — declared approximation.
- Legacy synthetic `train`/`demo`/`score`/`deck`/`all` keep working.
- Every file under 400 lines; type hints; no `print` in library modules; all
  verification raises, no silent degradation.
- Bundle schema string `gradcam-repro.web-bundle.v2`; CAM layer key `"cam"`.

## Decisions changed by this plan

| # | Prior decision | New decision | Why |
|---|---|---|---|
| 1 | 4 cases, first of fold 3 | **All 1289 fold-3 val cases** | User's call. Enables AUROC anchor + evidence-based exemplar selection. |
| 2 | Device `mps`, fallback `cpu` | **`cuda`, fallback `cpu`**; multi-GPU shard across 3 devices | Hardware. `mps` path retained but no longer default. |
| 3 | Occlusion batch 1 ("batching brings no benefit") | **Batch 32, tuned in Phase 1** | That finding was an MPS quirk (batch 8 was *worse* at 0.785 s/pos). CUDA batches scale. |
| 4 | Case 3 centre-cropped to fit | **Pad to a conforming shape; never crop** | The crop was rejected-alternative-by-cost: padding to (64,160,160) was "3.4x slower". Free at 48 GB. Deletes deviation #1. |
| 5 | Occluder 16^3 voxels (32 x 10.4 x 10.4 mm, 3x elongated in z) | **Physically isotropic 8 x 24 x 24** | The doc offers this at "1.9x cost" — now affordable. Removes a caption apology. |
| 6 | No AUROC reproduction | **Reproduce fold-3 AUROC** | Was dropped only because 4 cases cannot produce it. |
| 7 | IG steps 16 (reduced from legacy 50) | **32, confirmed by convergence check in Phase 1** | 16 was a laptop concession. Verify rather than assume. |

## Blockers and unknowns — read before scheduling

**B1. The imaging data is gated and absent (blocking).**
`FLARE-MedFM/FLARE-AutoMSC` is `gated: "manual"`; access is granted by the
organisers. Nothing is in `~/.cache/huggingface` on this machine and there is no
`data/luna25/`. Full val needs **1289 image+mask pairs, ~1.8 GB** (vs 5.6 MB
before). Either an existing granted token is reusable here, or this waits on the
organisers. *No downstream phase can start without this.*

**B2. Shape non-conformance is an unquantified population risk (blocking the
cost model).**
The FPN's `ConvTranspose3d` doubles exactly while encoder stride-2 convs round
up, so misaligned inputs raise `RuntimeError: size of tensor a (N) must match
tensor b (N+1)`. Measured crashes: (64,105,105), (64,113,113), (64,135,135),
(64,136,136), (64,144,144). Among the *four* prior cases, three needed padding
and one needed cropping — i.e. **4/4 were non-conforming**. The fingerprint
records 782 distinct spacings, so resampled in-plane sizes vary widely across
1289 cases. Phase 1 must derive the conforming-shape rule analytically and
verify it over all 1289 before any batch run. If padding must vary per case,
uniform batching is compromised and Phase 3's cost model changes.

**B3. `pre-jsc-luna25-swap` tag missing.** `git tag -l` is empty. Recreated in
Phase 0.

**B4. CUDA numerical equivalence unverified.** Prior deviation #4 checked
MPS vs CPU (`4.8e-07`). CUDA is a different backend and **TF32 matmuls are on by
default on Ada**, which will exceed that tolerance. Phase 1 pins TF32 off for
the attribution path and re-establishes the equivalence figure.

**B5. Class imbalance shapes every aggregate.** Fold-3 val is 1289 cases with
**104 Malignant (8.1%)**. Means over all cases are dominated by Benign.
All aggregates report per-class, and AUROC is the primary discrimination number.

## Phases

Each phase gates the next. Every check raises.

### Phase 0 — environment, provenance, rollback (no data needed)

- Recreate tag `pre-jsc-luna25-swap` at `982cc86`.
- Clone JSC at the pinned commit to `third_party/JSC` (gitignored). Apply the
  two known local patches: `[tool.setuptools.packages.find] include =
  ["nnunetv2*"]`, and install `torchmetrics` (absent from JSC's deps but
  imported at `nnUNetCLSTrainer.py:73`).
- `uv pip install -e third_party/JSC`. Record: **`uv sync` prunes it**;
  reinstall after any sync.
- Set `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` env vars (the fork
  reads env vars and only warns when unset; do *not* edit `paths.py`).
- Fetch the checkpoint; verify sha256 `33ad8109...5968f` and 443,676,001 bytes.
- Verify `load_state_dict(strict=True)` reports all keys matched, and that the
  621 tensors / 329 unique params alias as documented.

**Gate:** checkpoint loads on CUDA, `conv_block` output is
`(1, 640, 16, 16, 16)`.

### Phase 1 — shape law, numerics, and a measured cost model (resolves B2, B4)

This phase exists because the prior plan's cost model does not survive the
scope change, and must be rebuilt on measurement.

- **Derive the conforming-shape law** from the stride chain
  (`strides [[1,1,1],[1,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]`), then verify it
  reproduces all five known crashes and all four known passes.
- Build `artifacts/jsc/case_index.csv` for **all 1289 fold-3 val cases**:
  case_id, label, original shape/spacing, resampled shape, required pad/crop,
  conforming target shape. Cross-check on-disk NIfTI shape/spacing against
  `dataset_fingerprint.json` (whose arrays align index-wise with the *sorted*
  case-ID list).
- **Padding only. If any case cannot be made conforming by padding, stop and
  report it** — do not silently crop. Cropping discards tissue and was the
  prior design's declared deviation #1.
- Pin TF32 off; re-measure CPU vs CUDA logit agreement on a known case and
  record the figure.
- Tune occlusion batch size (8/16/32/64) and confirm IG convergence at 16 vs
  32 vs 64 steps.

**Gate:** all 1289 cases have a verified conforming shape; batch size and IG
steps chosen from data; CUDA/CPU agreement recorded. **Publish the real
projected wall-clock here — the ~72 min figure in the prior design is void.**

### Phase 2 — pipeline on a stratified pilot (~40 cases)

- Implement M1 loader / M2 data / M3 attribution / M4 scoring per the prior
  design, with the Phase-1 shape law and CUDA device.
- Run all seven methods end-to-end on a stratified pilot: ~20 Malignant, ~20
  Benign, spanning the mask-size range (the four known masks span 241–5225
  voxels, 0.023%–0.498% of volume).
- Self-tests: uniform heatmap scores `enrichment == 1.0` within tolerance;
  every map matches patch shape, no NaN, not all-zero; masks non-empty.

**Gate:** seven methods produce valid maps on every pilot case; metrics
self-consistent; extrapolated full-val cost within budget.

### Phase 3 — full fold-3 val run (1289 cases)

- Shard across the 3 GPUs; detached with `nohup setsid … & disown` so it
  survives session close; per-case incremental CSV writes so a crash loses one
  case, not the run; watchdog for stalls.
- Emit per-case: logit, predicted probability, true label, seg Dice vs GT mask,
  and for each of the seven methods `enrichment`, `mass_in_gt`,
  `inside_outside_ratio`, `pointing_acc`, `cam_seg_agreement`.
- **Verify on disk, not by exit code** — count output rows and spot-check
  files. A clean exit is not evidence of written output.

**Gate:** 1289/1289 rows present and non-corrupt; zero tracebacks.

### Phase 4 — AUROC anchor (resolves goal 2)

- Compute fold-3 val AUROC from the 1289 logits; compare against README `0.884`
  and embedded `val_auc[-1] = 0.8875`.
- **A materially lower AUROC means the pipeline is wrong, not the paper.**
  Treat as a blocking correctness failure: suspect preprocessing drift, padding
  effects on InstanceNorm/GAP statistics, or TF32.
- Report per-class counts alongside (B5).

**Gate:** AUROC within a stated tolerance of the published anchors, or a
diagnosed and documented explanation.

### Phase 5 — exemplar selection (the success/failure ask)

Selection rule fixed **before** looking at heatmaps, and published with the
figures:

- **Success:** correctly and confidently classified (Malignant p high /
  Benign p low), high seg Dice, and high attribution `enrichment`.
- **Failure:** split into two distinct kinds, because they mean different
  things —
  - *classification failure* — confidently wrong logit;
  - *attribution failure* — correctly classified but `enrichment` near or below
    1.0, i.e. the model is right while the explanation does not localise the
    lesion. This is the scientifically interesting cell and only a full-val
    sweep can find it.
- Sample exemplars from both classes; **never present a hand-picked case
  without the distribution behind it.** Every exemplar figure cites its
  percentile in the 1289-case distribution.
- Keep the longitudinal pair `100012_1_19990102` / `100012_1_20000102` (same
  patient, nodule grows 521.9 -> 3125.3 mm^3, 5.99x) as a named case study, with
  the caveat that the two scans are not independent samples.

**Gate:** selection rule stated in the doc and reproducible from the CSV by
script — no manual curation.

### Phase 6 — figures, web bundle, deck

- Distribution figures first (per-method `enrichment` over 1289, split by
  class), then exemplar grids: CT / GT mask / predicted mask / seven heatmaps.
- Benign-vs-Malignant pairing replaces `class_discriminability.png` (with one
  logit, per-case class flipping is degenerate — "class 0" Grad-CAM is the
  negation of "class 1").
- Bump `web/DATA_CONTRACT.md` to `gradcam-repro.web-bundle.v2`; move the seven
  `tests/test_web_export_*.py` with it.
- Axial slices need no aspect correction (y-x is isotropic at 0.664 mm); the
  2.0 mm z spacing affects only CAM tap resolution.

## Declared deviations after this plan

Down from five to three — two were pure compute compromises and this hardware
retires them.

1. **Guided backprop treats LeakyReLU as ReLU** (unchanged; no standard guided
   rule exists for LeakyReLU).
2. **Occlusion runs `cls_only`**, skipping the decoder — exact for the
   classification logit, since the decoder feeds only `seg_output`.
3. **Attribution runs on CUDA with TF32 disabled**; equivalence figure from
   Phase 1.

Retired: the case-3 centre-crop (now padded), and the elongated occluder (now
physically isotropic).

## Out of scope

- Training any backbone, including ResEncM
- Folds 0, 1, 2, 4 — fold 3 only, matching the checkpoint
- The full 6132-case dataset; only fold-3 val is fetched
- Reproducing the published 5-fold aggregate metrics
