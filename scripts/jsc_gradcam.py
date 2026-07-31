"""Run the 7-method attribution suite on the JSC LUNA25 fold-3 model.

Wiring only: nnU-Net's own preprocessing (via JSC's SimplePredictor) feeds
`gradcam3d_viz.run_gradcam`, which owns the attribution methods.

Deviations from the official inference script, all deliberate:
  * eval() not train()  -- the official script calls .train() at L345, which
    activates Dropout(0.5) in the classification head and makes every run
    non-reproducible. Attribution maps must be deterministic.
  * TTA off             -- mirroring averages 8 flipped branches; gradients
                           would live in flipped coordinate frames.
  * TF32 off            -- keeps CUDA numerically comparable to CPU.

Current limitation: single-sliding-window cases only (569 / 1289 of fold-3 val).
gradcam3d_viz has no sliding-window support; the other 720 cases need a
window-aggregation layer that does not exist yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "JSC"))

# gradcam3d_viz lives under a filename with a space; import it by path.
import importlib.util

_VIZ = REPO / "scripts" / "gradcam3d_viz.py"
_spec = importlib.util.spec_from_file_location("gradcam3d_viz", _VIZ)
gradcam3d_viz = importlib.util.module_from_spec(_spec)
sys.modules["gradcam3d_viz"] = gradcam3d_viz
_spec.loader.exec_module(gradcam3d_viz)
run_gradcam = gradcam3d_viz.run_gradcam
GradcamConfig = gradcam3d_viz.GradcamConfig

from segcls_ensemble_infer import SimplePredictor  # noqa: E402
from batchgenerators.utilities.file_and_folder_operations import join  # noqa: E402
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO  # noqa: E402

# Canonical layout per docs/jsc-luna25-sources.md: plans.json, dataset.json and
# dataset_fingerprint.json sit here, the weights in fold_<N>/. That is exactly
# the shape the official predictor expects for --model_path, so no extra
# trainer-named directory is needed.
#
# Use these two files, NOT the checkpoint's embedded copies and NOT the gated
# dataset repo's dataset.json: of the four sources only the weights repo's
# plans.json and dataset.json name the dataset Dataset005_LUNA25. The embedded
# dataset_json and the gated dataset.json both say Dataset009_LUNA25, and
# nnU-Net resolves paths through that field.
MODEL_DIR = "artifacts/jsc"


class JSCAdapter(torch.nn.Module):
    """Presents the JSC network as forward(x) -> (seg, cls_logits).

    run_gradcam drives forward/backward itself, so this only needs to be a
    plain nn.Module with no inference_mode anywhere in its call path.
    """

    def __init__(self, network: torch.nn.Module):
        super().__init__()
        self.network = network
        # CAM tap: output of conv_block, (1, 640, z/4, y/8, x/8)
        self.conv_block = network.conv_block

    def forward(self, x):
        return self.network(x)


def build_predictor(device: str, checkpoint: Path | None = None) -> SimplePredictor:
    """Load the published fold-3 network.

    `checkpoint` points at the .pth itself (e.g. artifacts/jsc/fold_3/
    checkpoint_best.pth); its grandparent is the model directory and its parent
    name supplies the fold, so a caller can swap folds without a second flag.
    """
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        model_dir = checkpoint.parent.parent
        fold = checkpoint.parent.name.removeprefix("fold_")
        ckpt_name = checkpoint.name
    else:
        model_dir, fold, ckpt_name = REPO / MODEL_DIR, "3", "checkpoint_best.pth"
    for required in ("plans.json", "dataset.json"):
        if not (model_dir / required).is_file():
            raise FileNotFoundError(
                f"{model_dir / required} missing — the model directory must hold "
                "plans.json and dataset.json alongside fold_<N>/. See "
                "docs/jsc-luna25-sources.md."
            )

    dev = torch.device(device, 0) if device != "cpu" else torch.device("cpu")
    pred = SimplePredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,          # TTA off
        perform_everything_on_device=(device != "cpu"),
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    pred.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(fold,), checkpoint_name=ckpt_name
    )
    # The official script never loads weights in initialize_...; it loads them
    # per-fold inside inference(). We load explicitly and keep eval().
    pred.network.load_state_dict(pred.list_of_parameters[0])
    pred.network.to(dev)
    pred.network.eval()
    return pred


def preprocess_case(pred: SimplePredictor, case_id: str, data_root: Path):
    """Intensity-normalise only. Returns (image_tensor, gt_mask, props).

    LUNA25 ships every case already cropped to exactly the model's patch size
    (verified: all 1289 fold-3 val cases are (64,128,128) on disk, one single
    distinct shape, with image and label shapes and spacings matching).

    Resampling is therefore deliberately SKIPPED. Running the plans resample
    would rescale (64,128,128) -> (64,113,113) and then zero-pad back to
    (64,128,128) at an offset of 7 voxels per in-plane edge, which both loses
    resolution (0.586 mm native -> 0.664 mm target) and puts the image in a
    different frame from the label. Aligning the mask through a separate
    resample path gave IoU 0.1955 against the on-disk mask -- the nodule was
    smeared from 6 z-slices to 18 and squeezed from 15 x-columns to 5.

    Because no geometric transform is applied, the on-disk mask is already
    registered to the image and is read verbatim.
    """
    import SimpleITK as sitk

    img_path = data_root / "imagesTr" / f"{case_id}_0000.nii.gz"
    seg_path = data_root / "labelsTr" / f"{case_id}.nii.gz"

    img_itk = sitk.ReadImage(str(img_path))
    seg_itk = sitk.ReadImage(str(seg_path))
    image = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    gt = (sitk.GetArrayFromImage(seg_itk) > 0).astype(np.uint8)

    patch = tuple(pred.configuration_manager.patch_size)
    if image.shape != patch:
        raise RuntimeError(
            f"{case_id}: on-disk shape {image.shape} != patch {patch}. This "
            "pipeline assumes pre-cropped LUNA25 cases; a case needing geometric "
            "resampling must not be handled by the no-resample path."
        )
    if gt.shape != image.shape:
        raise RuntimeError(f"{case_id}: label shape {gt.shape} != image {image.shape}")

    # plans CTNormalization, applied verbatim (nnU-Net's CTNormalization):
    #   clip to [percentile_00_5, percentile_99_5], then (x - mean) / std
    fg = pred.plans_manager.foreground_intensity_properties_per_channel["0"]
    image = np.clip(image, fg["percentile_00_5"], fg["percentile_99_5"])
    image = (image - fg["mean"]) / max(fg["std"], 1e-8)

    props = {
        "sitk_stuff": {
            "spacing": img_itk.GetSpacing(),
            "origin": img_itk.GetOrigin(),
            "direction": img_itk.GetDirection(),
        },
        "spacing": list(img_itk.GetSpacing())[::-1],
    }
    return torch.from_numpy(image[None]).float(), gt, props


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--data-root", type=Path, default=REPO / "data/luna25")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/jsc/gradcam_out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="path to fold_<N>/checkpoint_best.pth; defaults to "
                         "artifacts/jsc/fold_3/checkpoint_best.pth")
    ap.add_argument("--methods", nargs="+", default=["notgradcam", "truegradcam"])
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pred = build_predictor(args.device, args.checkpoint)
    net = JSCAdapter(pred.network).eval()
    print(f"[jsc] network ready on {args.device}, TF32 off, TTA off, eval()")

    meta = REPO / "artifacts/jsc/luna25_meta"
    labels = {r["identifier"]: int(r["label"])
              for r in csv.DictReader(open(meta / "cls_data.csv"))}

    patients = []
    for c in args.cases:
        # No resample, so the on-disk mask needs no alignment step (see
        # preprocess_case). run_gradcam wants (C, D, H, W) and batches it itself.
        x, gt_seg, props = preprocess_case(pred, c, args.data_root)
        patients.append({
            "patient_id": c,
            "input": x,
            "gt_label": labels[c],
            "gt_seg": gt_seg,
        })
        print(f"[jsc] {c}: input {tuple(x.shape)} label={labels[c]} "
              f"gt_seg={gt_seg.shape} nodule_voxels={int(gt_seg.sum())}")

    cfg = GradcamConfig(
        get_stages_fn=lambda m: [m.conv_block],       # CAM tap
        extract_logits_fn=lambda out: out[1],         # (seg, cls) -> cls
        class_names={0: "Benign", 1: "Malignant"},
        methods=args.methods,
        out_dir=str(args.out),
        occ_mask_size=16,
        ig_n_steps=32,
    )
    run_gradcam(net, patients, cfg)
    print(f"[jsc] done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
