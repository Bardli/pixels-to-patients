# Measured findings: wiring gradcam3d_viz onto the JSC LUNA25 model

Date: 2026-07-29
Status: living record — Phase 0 complete, first attribution run green
Companion plan removed in the 2026-07-30 repository cleanup; this file is the
surviving record.

Everything below was measured on this machine (3x RTX 6000 Ada, CUDA 13.0,
`torch 2.13.0+cu130`), not carried over from the 2026-07-28 laptop session.
Where a measurement contradicts an earlier document, the contradiction is
stated explicitly rather than silently overwriting it.

## 1. Assets — all verified, nothing gated in practice

The 2026-07-28 design listed gated access as a blocking risk (B1). It is not.

The HF token at `~/.cache/huggingface/token` belongs to user **BardF**, who holds
`write` on the **FLARE-MedFM org itself** plus `canReadGatedRepos: true`. The
dataset gate was never an obstacle.

| Artifact | Verified |
|---|---|
| `checkpoint_best.pth` | 443,676,001 bytes, sha256 `33ad8109…5968f` — exact match |
| `splits_final.json` | sha256 `cf4cdd44…` — exact match |
| `cls_data.csv` | sha256 `4fbaf9b3…` — exact match |
| `dataset.json` | sha256 `3a5bb526…` — exact match |
| Imaging (fold-3 val) | **2578/2578 files, 1770.1 MB**, 1289 image+label pairs, 0 failures |

Downloader: [`scripts/download_luna25_fold3.py`](../../../scripts/download_luna25_fold3.py).
Resumable, validates every file is real gzip before accepting it (a gated 401
returns a short HTML body that would otherwise sit on disk masquerading as
data), and re-scans the disk at the end rather than trusting exit status.
Per-file sha256 manifest at `artifacts/jsc/fold3_val_manifest.csv`.

## 2. The `dataset.json` gotcha runs the opposite way from the 2026-07-28 design

That document says the *gated* `dataset.json` carries the wrong dataset name and
instructs using the checkpoint's embedded copy instead. Measured, the embedded
copy is the one that is wrong, and the two files are identical anyway:

| Source | `name` / `dataset_name` |
|---|---|
| Gated `dataset.json` | `Dataset009_LUNA25` |
| Checkpoint embedded `dataset_json` | `Dataset009_LUNA25` |
| Checkpoint embedded `plans` | `Dataset009_LUNA25` |
| **Downloaded `plans.json`** | **`Dataset005_LUNA25`** |

`embedded == gated` evaluates to `True` with no differing keys, so the original
rule is harmless but vacuous. The real hazard is the reverse one: the embedded
plans say `Dataset009` while the folder on disk is `Dataset005`, and nnU-Net
resolves paths through that field.

**Consequence:** use the *downloaded* `plans.json`. This is what
`artifacts/jsc/model/…/plans.json` now holds, and it is why the official
inference script works without a path override.

Two smaller corrections to the same document's checkpoint facts:
`fold` is `None` at top level (the `3` lives in `init_args`), and the weights
resolve to **145** unique parameters, not the documented 329 — aliasing is
heavier than recorded. Neither affects loading.

## 3. Split integrity — clean at both case and patient level

| Check | Result |
|---|---|
| fold-3 `train ∩ val` | **0** |
| `train ∪ val` | 6132 = `numTraining` |
| Duplicates within val / train | 0 / 0 |
| **Patient-level** overlap (`PATIENT_x_DATE` → first field) | **0** (421 val patients vs 1687 train patients) |
| All five folds `train ∩ val` | 0 for every fold |
| Downloaded files vs fold-3 val list | set-equal; 0 extra, 0 missing, 0 in train |

Patient-level disjointness matters more than case-level here: LUNA25 contains
repeat scans of the same patient (e.g. `100012_1_19990102` and
`100012_1_20000102`), so a split that kept cases distinct but let a patient
straddle both sides would inflate AUROC. This split does not.

Class balance, fold-3 val: **104 Malignant / 1185 Benign = 8.1%**
(whole dataset 543 / 5589 = 10.3 : 1). Every val case has a label.

## 4. Sliding windows — SUPERSEDED, every case is single-window

> **Corrected 2026-07-29, later the same day (see §10).** The census below is
> wrong, and the error was mine: it was computed by *simulating* the plans
> resample in my own script, not by reading the files. Measured on disk, all
> 1289 fold-3 val cases are exactly `(64,128,128)` — one single distinct shape,
> equal to the patch size, with zero image/label shape mismatches. Since §10
> skips resampling entirely, **every case is single-window** and no
> window-aggregation layer is needed. The rest of this section is retained
> because the design note at the end still applies if resampling is ever
> reintroduced.

This was an open unknown (B2) and the answer appeared to invalidate an
assumption made earlier in the session, namely that LUNA25 crops are all
smaller than the `[64,128,128]` patch and therefore single-window.

Census over all 1289 fold-3 val cases (`artifacts/jsc/window_census.csv`),
generated from nnU-Net's resample arithmetic and
`compute_steps_for_sliding_window` logic — **note this models a resample that we
do not actually perform**:

| Windows | Cases |
|---|---|
| 1 | 569 (44%) |
| 2 | 103 |
| 4 | 467 |
| 8 | 148 |
| 12 | 2 |

- **720 cases (56%) are multi-window**, max 12
- 785 need zero-padding on some axis; 720 *exceed* the patch on some axis
- resampled `y` spans **90..181** — the large tail does not fit the patch

`gradcam3d_viz` has **no** sliding-window support (`grep patch|slicer|sliding|
pad_nd|window` returns nothing). It assumes one forward pass over the whole
volume. So the 569 single-window cases work as-is and the other 720 need a
window-aggregation layer that does not exist yet.

Design note for that layer: aggregate the **logit first, then backward** (the
official script averages per-window class logits at
`segcls_ensemble_infer.py:303`), rather than doing per-window CAM and stitching.
The quantity we want to explain is the case-level prediction, and per-window
CAMs carry independent normalisation scales that make them non-comparable
across a seam. Memory is not a constraint: one window's tap activation is
`640x16x16x16` fp32 ≈ 10 MB, so even 12 windows is trivial on 48 GB.

## 5. Three bugs in the official inference script

Found by reading all 462 lines of `segcls_ensemble_infer.py`. None are ours, but
all three affect how we use it.

**5.1 `.train()` before inference (L345) — breaks reproducibility.**
The official script calls `self.network.train()` inside `inference()`, with the
comment "messing with state dict names". The classification head contains
`Dropout(0.5)`, so this randomises every forward pass: the same case scored
twice gives different logits and different heatmaps. Attribution maps must be
deterministic, so **we use `eval()`**.

Measured mitigation detail: the backbone is `InstanceNorm3d` (no running stats
tracked — only 9 running-stat tensors exist in the whole checkpoint), so
`.train()` is inert for normalisation. Dropout is the entire problem.

`eval()` and gradient availability are orthogonal — `eval()` only switches
Dropout/BatchNorm behaviour and does not affect the autograd graph.

**5.2 `class_logits` buffer is allocated with the wrong shape (L282).**
`torch.zeros((self.cls_class_num))` gives shape `(2,)`, but the network emits
`(1,)` per patch because `cls_head_output = cls_class_num if cls_class_num > 2
else 1` → `1`. Broadcasting hides it: `class_logits += class_logits_patch` writes
the same value into both slots, so the emitted `probs` column is a length-2 list
with identical entries. Harmless for AUROC (take either), misleading as a column.

**5.3 `--fold` default is a tuple but `type=str` (L403).**
Default `(0,1,2,3,4)` means a 5-fold ensemble; we only have fold 3, so
`--fold 3` must be passed explicitly or the script looks for fold directories
that do not exist. Output filename is consequently `fold3_results.csv`, not the
`results.csv` the README promises.

Also: `--model_path` must point at the **trainer** directory (the script reads
`model_path/{dataset.json,plans.json}` at L157-158 and weights at
`model_path/fold_{f}/…` at L168), not at a fold directory.

## 6. Why hooks are required — the official script cannot supply attribution

The official script's outputs are the segmentation mask and the classification
probability, i.e. terminal forward products. Grad-CAM and its relatives need
`∂y_c/∂A_k`: the gradient of the target logit with respect to an intermediate
activation. That requires a real backward pass.

The distinction that matters: backward-for-attribution reads `.grad` and never
calls `optimizer.step()`, so no weight changes. "Inference has no gradients" is a
property of the *code* (`inference_mode`), not of the model — the model is
differentiable the moment the graph is enabled.

The blocker is structural: the whole call path is gradient-free at three levels —
`inference()` (L326) is `@torch.inference_mode()` *and* opens
`with torch.no_grad()`, `_internal_predict_sliding_window_return_logits` (L261)
is `@torch.inference_mode()`, and `_internal_maybe_mirror_and_predict` (L237) is
`@torch.inference_mode()`. `inference_mode` is stricter than `no_grad`: tensors
created inside it cannot participate in autograd later at all, so the forward
pass must be *re-run* under `enable_grad()`, not patched after the fact.

Of the seven methods, only Occlusion is pure-forward. The other six need hooks.

## 7. What `gradcam3d_viz` already solves

The vendored 1534-line `scripts/gradcam3d_viz.py` is more mature than the M3 module the plan
proposed writing, and it retires one declared deviation outright.

**7.1 A correct guided rule for LeakyReLU — deviation retired.**
The 2026-07-28 design declared "guided backprop treats LeakyReLU as ReLU" as an
approximation to be disclosed in figure captions. Unnecessary:
`_GuidedLeakyReLUFunc` (L201-213) implements a real custom autograd Function —
forward is standard LeakyReLU, backward gates by `(grad>0)*(x>0)`. **Deviation
#1 of the prior design is deleted, not disclosed.**

**7.2 The inplace hazard is sidestepped structurally.**
`guided_backprop_context` (L245-270) swaps whole modules rather than clearing
`inplace` flags, and iterates `_modules` rather than `__dict__` — the docstring
notes this is because some models name the activation `nonlin` rather than
`relu`. nnU-Net uses exactly that name, so this trap was already hit and fixed
upstream.

**7.3 Single-logit handling exists** (L298-301), matching the design's
observation that with one logit the class-0 CAM is the negation of class-1.

**7.4 Model-agnostic injection.** `get_stages_fn` + `extract_logits_fn` are all
that JSC needs:
```python
get_stages_fn     = lambda m: [m.conv_block]   # CAM tap
extract_logits_fn = lambda out: out[1]         # (seg, cls) -> cls
```

**7.5 Also present:** all seven methods, `align_gt_seg` for real-mask alignment,
and `_classify_outcome` (L358) producing TP/FP/TN/FN badges — the last is
directly what Phase 5's success/failure exemplar selection needs.

## 8. Bug in `gradcam3d_viz`: `target_class=-1` collapses to class 0 for single-logit models

At L298-301:
```python
if logits.shape[1] == 1:
    cls = 0 if (user_target_class == -1 or user_target_class == 0) else 1
    score = logits[0, 0] if cls == 0 else -logits[0, 0]
```
`target_class = -1` documents as "use the predicted class", but in the
single-logit branch `-1` is hard-mapped to class **0**, ignoring the prediction
entirely.

Observed: `100012_1_19990102` has GT=1, pred=1, outcome TP — and
`target_class = 0` in its metadata. Because the class-0 attribution is the
negation of class-1, **every malignant case's heatmap comes out sign-flipped**,
highlighting evidence *for benign*. Multi-class models never enter this branch,
so it is specific to single-logit heads like JSC's.

Status: **to be fixed** — `-1` must follow `pred_label`.

## 9. First green run

Wiring: [`scripts/jsc_gradcam.py`](../../../scripts/jsc_gradcam.py). Reuses the
official `SimplePredictor` for preprocessing (plans `CTNormalization` + resample;
hand-rolled normalisation stays forbidden) and hands `(C,D,H,W)` tensors to
`run_gradcam`.

```
[jsc] network ready on cuda, TF32 off, TTA off, eval()
[1/2] 100012_1_19990102  input (1,64,128,128)  GT=1 Pred=1 -> TP   36 figures
[2/2] 100289_4_20010102  input (1,64,128,128)  GT=0 Pred=0 -> TN   18 figures
```

Both single-window cases, both classified correctly, 54 figures written. TTA off
and TF32 off per §5.1 and B4.

## 10. No resampling — LUNA25 ships pre-cropped to patch size

Found by questioning the rendered GT rather than the numbers: the mask contour
looked like a thin vertical sliver when a 760-voxel nodule should be roughly
spherical.

**Measured on disk, all 1289 fold-3 val cases:** exactly one distinct shape,
`(64,128,128)`, equal to the model's patch size, with **zero** image/label shape
mismatches and matching spacings per case. There are 335 distinct spacings in the
metadata (the same voxel grid representing different physical sizes), but the
crops are already patch-shaped.

Running the plans resample therefore does active harm:

```
image: (64,128,128) -> resample to plans spacing -> (64,113,113)
                    -> zero-pad back to (64,128,128) at slice(7,120)   [7-voxel offset]
mask:  (64,128,128) -> align_gt_seg's own separate resample path
                    -> lands in a DIFFERENT frame
```

Measured consequences on `100012_1_19990102`:

| | voxels | centroid (z,y,x) | bbox |
|---|---|---|---|
| On-disk mask | 760 | (31.4, 65.5, 65.0) | z[29..34] y[56..75] x[58..72] |
| `align_gt_seg` output | 579 | (30.1, 65.7, 64.0) | z[22..39] y[58..73] **x[62..66]** |

**IoU 0.1955.** The nodule was smeared from 6 z-slices to 18 and squeezed from
15 x-columns to 5, losing 181 voxels — and the image lost resolution too
(0.586 mm native resampled to 0.664 mm).

**Fix:** skip resampling; apply only plans `CTNormalization` (verified line-by-line
against `nnunetv2/preprocessing/normalization/default_normalization_schemes.py`:
clip to `[percentile_00_5, percentile_99_5]`, subtract `mean`, divide by
`max(std, 1e-8)`), and read the on-disk mask verbatim. Two hard guards raise
rather than degrade: shape must equal the patch, and label shape must equal image
shape.

Result: **IoU 1.0000** on both test cases, voxel counts, centroids and bboxes
identical to the on-disk masks. Z-slice ranges recovered from the smeared 18/9 to
the true 6/3.

This is also why §4 is superseded and why the deviation ledger gains an entry.

## 11. Web bundle swapped to v2 (2026-07-29)

The site does **not** consume matplotlib composites. It reads raw pixel-data
PNGs (one pixel per voxel) plus a parallel `.json` payload per slice that drives
hover read-outs (`web/js/bench.js:31-32`). So the 189 headerless figures from §9
were the wrong artifact; the deliverable is an exporter, not a file copy.
Written as [`scripts/jsc_web_export.py`](../../../scripts/jsc_web_export.py),
schema `gradcam-repro.web-bundle.v2`.

### Exemplars — selected on measured criteria, all four outcomes

Screened 160 cases (80 malignant / 80 benign) forward-only, then picked:

| Example | Outcome | logit | seg Dice | GT voxels |
|---|---|---|---|---|
| `203125_1_19990102` | TP | +3.52 | 0.950 | 22686 |
| `207347_1_20010102` | TP | +2.50 | 0.934 | 5539 |
| `100438_5_20010102` | TN | −6.10 | 0.907 | 58 |
| `100471_3_20000102` | **FP** | +1.46 | 0.647 | 9108 |
| `134082_1_19990102` | **FN** | −0.94 | 0.960 | 659 |

### `enrichment` confirmed necessary, not theoretical

The 2026-07-28 design predicted raw `mass_in_gt` would collapse on real masks.
Measured across the five exemplars it lands at **0.0074–0.0305**, while
`enrichment` spreads 1.16–5.10 and stays comparable across lesion sizes. The
compare page now ranks by `enrichment`, not mass.

| Method | enrichment | pointing |
|---|---|---|
| gradcam | **5.10** | 0.20 |
| occlusion | 5.05 | 0.20 |
| layercam | 4.63 | 0.20 |
| integrated_gradcam | 4.15 | 0.40 |
| notgradcam | 4.08 | 0.40 |
| integrated_gradients | 2.66 | 0.00 |
| guided_gradcam | 1.16 | **1.00** |

`guided_gradcam` is a genuine metric-disagreement case: it hits the peak every
time (`pointing 1.00`) while scoring 1.16 enrichment — barely better than a
uniform map. That pairing replaced an invented quiz number on the compare page.

### Two site claims were false under the real model and were re-measured

Not rewritten from assumption — measured first, per
[[feedback_verify_premise_before_fix]].

1. **"About 3 of 16 channels fire; most are blank."** The real tap has 640
   channels and **639 fire**; across the five exemplars `n_silent` is 0 or 1.
   The sparsity is real but lives at the *voxel* level — 52–72% of activations
   are exactly zero depending on the case. R3.4 still holds, but its
   explanation inverts: averaging hundreds of *busy* channels is what dilutes
   the class signal. The baseline page now says so and shows all 640 channels
   (top 6 as slices, the rest as mean chips via `all_channels`).
2. **"Integrated Gradients had the highest mass-in-GT (33.5%)."** Impossible
   against real masks. Replaced with the measured `guided_gradcam` case above.

### Interface contract worth knowing

`activations.json` keeps the layer ids `stage1/stage2/stage3` because
`web/js/scene3d.js` hardcodes them, but they now tap `encoder.stages[2]`,
`conv_block`, and `encoder.stages[5]`. Each entry carries `real_name`, which is
what the UI labels — the ids are plumbing, not claims.

### Prose, code snippets, and tutorial all swapped

Site-wide: zero remaining references to `Task06`, `MSD`, `ToyCNN`, `RealCtCNN`,
`real_ct`, `lung_0NN`, `stage2`-as-a-claim, `32³`, or `tumour`. Code snippets on
the Grad-CAM, notGradCAM, and Occlusion pages now show the real
`_compute_truegradcam` / `JSCAdapter` / `_as_two_column` / `_pick_target_class` /
`_compute_occlusion` sources, including the two sign fixes. `reproduce.html`
documents the real four-step pipeline (gated download → published checkpoint
layout → attribute + score) with no training step, and lists all four declared
deviations.

## Deviation ledger, current state

Down to three from the original five, and one more is retired here.

| # | Deviation | Status |
|---|---|---|
| ~~1~~ | ~~Guided backprop treats LeakyReLU as ReLU~~ | **retired** — real guided rule exists (§7.1) |
| ~~2~~ | ~~Case 3 centre-cropped~~ | **retired** — GPU makes padding free |
| ~~3~~ | ~~Elongated 16³ occluder~~ | **retired** — physically isotropic affordable |
| 4 | Occlusion runs `cls_only`, skipping the decoder | stands (exact for the cls logit) |
| 5 | CUDA with TF32 disabled | stands; equivalence figure still to measure |
| 6 | **TTA (mirroring) disabled for attribution** | **new** — gradients would live in flipped frames; our logits will not match the official script exactly |
| 7 | **No geometric resampling** | **new** (§10) — LUNA25 is pre-cropped to the patch size, so resampling misaligns image and mask (IoU 0.196) and loses resolution. Means our probabilities will not match the official script, which does resample. |

## Open items

1. ~~Fix §8 (`target_class` single-logit bug)~~ — **done.** Fixed, plus a second
   bug it was masking: MONAI indexes `logits[:, class_idx]`, which is out of
   bounds for class 1 on a single-logit head. Both resolved via
   `_as_two_column`, applied at six call sites.
2. ~~Run all seven methods~~ — **done.** All seven green on every exemplar.
3. ~~Build the sliding-window aggregation layer~~ — **cancelled, premise was
   wrong.** The 569/720 split came from my own script *simulating* the plans
   resample, not from disk. Measured on disk, all 1289 fold-3 val cases are
   exactly `(64,128,128)` — one single distinct shape, equal to the patch size.
   Since §10 skips resampling, every case is single-window.
4. ~~GT alignment~~ — **done.** `align_gt_seg` gave IoU **0.1955** against the
   on-disk mask (nodule smeared from 6 z-slices to 18, squeezed from 15
   x-columns to 5). Root cause: LUNA25 ships pre-cropped to patch size, so the
   plans resample rescaled the image to `(64,113,113)` and zero-padded back at a
   7-voxel offset while the mask took a separate path. Resampling is now skipped
   entirely and the mask is read verbatim: **IoU 1.0000**.
5. ~~Occlusion sign~~ — **done.** MONAI returns the occluded class *score*, not
   the score *drop*, so high meant unimportant. Negated; in-lesion ratio went
   0.29 → 3.55.
6. `Encoder stages: 1` — only `conv_block` is tapped for attribution. LayerCAM
   is designed for multi-layer aggregation, so consider also tapping the encoder
   stages. (The web bundle's `activations.json` already exports three taps, but
   that is for visualisation, not attribution.)
7. AUROC anchor: decide whether to run the official script unmodified as a
   baseline (which would also quantify the §5.1 Dropout non-determinism) or only
   our `eval()` path against README `0.884` / embedded `val_auc[-1] 0.8875`.
8. Full 1289-case run. Note one new variable: we skip resampling while the
   official script performs it, so the two paths' probabilities will not match
   exactly. Worth comparing both on ~50 cases before committing to a full sweep.
9. `brain_vol` / `brain_mask_threshold` in `gradcam3d_viz` are neuroimaging
   leftovers and the 0.05 threshold is not obviously right for CT-normalised
   lung. **Deferred by decision** — the effect is that normalisation can be
   dominated by chest wall on some cases; the agreed mitigation is exemplar
   selection rather than changing the normaliser, to stay close to the official
   preprocessing.
