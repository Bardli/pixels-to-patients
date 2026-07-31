"""Export a web bundle (schema v2) from the JSC / LUNA25 attribution pipeline.

The site consumes raw pixel-data PNGs (one pixel per voxel) plus a parallel
JSON payload per slice -- not matplotlib composites. So this writes the bundle
directly from the attribution arrays rather than copying rendered figures.

Layout and field semantics follow web/DATA_CONTRACT.md, bumped to
`gradcam-repro.web-bundle.v2`. Differences from v1, all forced by the model
swap (toy 2-class CNN -> JSC single-logit joint seg+cls):

  * `logits` is length 1, not 2. `pred_label` is `logit > 0`, not argmax.
  * `model_graph.json` describes the real PlainConvUNet + FPN path and gains a
    `seg_head` node; `cam_tap` is `conv_block`.
  * examples gain `pred_mask_slice.png` -- the model's OWN predicted nodule
    mask, which the toy CNN could not produce.
  * metrics gain `enrichment` (mass_in_gt / gt_volume_fraction) as the headline
    number, because real nodule masks occupy 0.02-0.5% of the volume and raw
    `mass_in_gt` collapses to ~0.00x and stops being comparable across cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "JSC"))

import importlib.util

_spec = importlib.util.spec_from_file_location("jsc_gradcam", REPO / "scripts" / "jsc_gradcam.py")
jsc_gradcam = importlib.util.module_from_spec(_spec)
sys.modules["jsc_gradcam"] = jsc_gradcam
_spec.loader.exec_module(jsc_gradcam)

gradcam3d_viz = jsc_gradcam.gradcam3d_viz

SCHEMA = "gradcam-repro.web-bundle.v2"

# Site-facing method keys -> internal gradcam3d_viz names. The site calls true
# Grad-CAM "gradcam"; the viz module calls it "truegradcam".
METHOD_KEYS = [
    ("notgradcam", "notgradcam"),
    ("gradcam", "truegradcam"),
    ("guided_gradcam", "guided_gradcam"),
    ("layercam", "layercam"),
    ("occlusion", "occlusion"),
    ("integrated_gradients", "integrated_gradients"),
    ("integrated_gradcam", "integrated_gradcam"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": False}


def slice_payload(sl: np.ndarray, prenormalised: bool) -> dict:
    """Quantise a 2D slice to the contract's payload shape."""
    vmin, vmax = float(sl.min()), float(sl.max())
    if prenormalised:
        norm = np.clip(sl, 0.0, 1.0)
    else:
        norm = (sl - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(sl)
    q = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return {"shape": [int(sl.shape[0]), int(sl.shape[1])],
            "values": [int(v) for v in q.ravel()],
            "vmin": vmin, "vmax": vmax}


def gray_png(sl: np.ndarray, path: Path) -> None:
    from PIL import Image
    vmin, vmax = float(sl.min()), float(sl.max())
    n = (sl - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(sl)
    Image.fromarray((n * 255).astype(np.uint8), mode="L").save(path)


def overlay_png(ct: np.ndarray, heat: np.ndarray, path: Path, cmap: str = "turbo") -> None:
    """CT in grayscale blended with a colormapped heatmap."""
    import matplotlib as mpl
    from PIL import Image
    vmin, vmax = float(ct.min()), float(ct.max())
    base = (ct - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(ct)
    rgb_base = np.stack([base] * 3, axis=-1)
    h = np.clip(heat, 0.0, 1.0)
    rgb_heat = mpl.colormaps[cmap](h)[..., :3]
    alpha = (h * 0.65)[..., None]
    out = rgb_base * (1 - alpha) + rgb_heat * alpha
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path)


def mask_png(ct: np.ndarray, mask: np.ndarray, path: Path, colour=(0.0, 1.0, 0.2)) -> None:
    from PIL import Image
    vmin, vmax = float(ct.min()), float(ct.max())
    base = (ct - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(ct)
    out = np.stack([base] * 3, axis=-1)
    m = mask > 0
    for i, c in enumerate(colour):
        out[..., i] = np.where(m, 0.35 * out[..., i] + 0.65 * c, out[..., i])
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path)


def metrics_for(heat3d: np.ndarray, gt3d: np.ndarray) -> dict:
    """mass_in_gt / inside_outside_ratio / pointing_acc / enrichment.

    `enrichment` normalises mass by the GT's volume fraction, so a uniform
    heatmap scores exactly 1.0 regardless of lesion size. Real LUNA25 nodules
    span 0.02-0.5% of the volume, which makes raw mass_in_gt incomparable
    across cases.
    """
    h = np.clip(heat3d.astype(np.float64), 0, None)
    gt = gt3d > 0
    total = h.sum()
    n_gt, n_all = int(gt.sum()), int(gt.size)
    if n_gt == 0:
        return {"mass_in_gt": 0.0, "inside_outside_ratio": 0.0,
                "pointing_acc": 0.0, "enrichment": 0.0}
    mass = float(h[gt].sum() / total) if total > 0 else 0.0
    inside = float(h[gt].mean())
    outside = float(h[~gt].mean()) if (~gt).any() else 0.0
    ratio = inside / outside if outside > 1e-12 else 0.0
    peak = np.unravel_index(int(np.argmax(h)), h.shape)
    pointing = 1.0 if gt[peak] else 0.0
    frac = n_gt / n_all
    return {"mass_in_gt": mass, "inside_outside_ratio": ratio,
            "pointing_acc": pointing, "enrichment": mass / frac if frac > 0 else 0.0}


def normalise01(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


# Three real tap points along the JSC forward path, exported under the layer
# ids the frontend's stepper expects (stage1/stage2/stage3). The names are an
# interface contract with web/js/scene3d.js; `real_name` carries the truth and
# is what the UI labels.
ACT_TAPS = [
    ("stage1", "encoder.stages[2]", lambda n: n.encoder.stages[2]),
    ("stage2", "conv_block (CAM tap)", lambda n: n.conv_block),
    ("stage3", "encoder.stages[5]", lambda n: n.encoder.stages[5]),
]
MAX_CHANNELS = 6


def activations_payload(net, x_batch) -> list[dict]:
    """Capture the three tapped activation tensors and summarise each.

    Per the site's teaching point (requirements R3.4) the channel grid must show
    which channels are silent and why, so `n_silent` is reported alongside the
    displayed top channels -- post-ReLU, a filter that does not respond to this
    patch outputs all zeros and contributes nothing to the mean.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for layer_id, _real, getter in ACT_TAPS:
        mod = getter(net.network if hasattr(net, "network") else net)
        handles.append(mod.register_forward_hook(
            lambda _m, _i, o, _k=layer_id: captured.__setitem__(
                _k, (o[0] if isinstance(o, (tuple, list)) else o).detach())))
    with torch.no_grad():
        net(x_batch)
    for h in handles:
        h.remove()

    out = []
    for layer_id, real_name, _g in ACT_TAPS:
        a = captured[layer_id][0].float().cpu().numpy()   # (C, D, H, W)
        C, D = a.shape[0], a.shape[1]
        flat = a.reshape(C, -1)
        ch_mean, ch_max = flat.mean(axis=1), flat.max(axis=1)
        n_silent = int((ch_max <= 1e-6).sum())
        # z index rescaled from the input grid into this layer's own depth
        fz = min(D - 1, max(0, int(round((x_batch.shape[2] // 2) * D / x_batch.shape[2]))))
        order = np.argsort(-ch_mean)[:min(MAX_CHANNELS, C)]
        channels = [{
            "index": int(i),
            "slice": slice_payload(a[i, fz], False),
            "mean": float(ch_mean[i]),
            "max": float(ch_max[i]),
        } for i in order]
        counts, bins = np.histogram(a, bins=20)
        # Every channel's mean/max, ordered by mean. Lets the baseline page show
        # the WHOLE set being averaged (R3.4) without shipping 640 slice
        # payloads: ~640 small objects instead of ~640 x 256 ints.
        order_all = np.argsort(-ch_mean)
        all_channels = [{"index": int(i), "mean": float(ch_mean[i]), "max": float(ch_max[i])}
                        for i in order_all]
        out.append({
            "layer": layer_id,
            "real_name": real_name,
            "feature_shape": [int(v) for v in a.shape],
            "feature_z": fz,
            "channels": channels,
            "all_channels": all_channels,
            "n_silent": n_silent,
            "n_channels": int(C),
            "frac_zero": float((a == 0).mean()),
            "histogram": {"bins": [float(b) for b in bins],
                          "counts": [int(c) for c in counts]},
        })
    return out


def model_graph(net, input_shape) -> dict:
    """Describe the real JSC forward path."""
    def pcount(m, exclude=()):
        """Parameter count, optionally excluding shared submodules.

        nnU-Net's UNetDecoder keeps a reference to the encoder it decodes
        (dynamic_network_architectures/building_blocks/unet_decoder.py:46,
        `self.encoder = encoder`), and nn.Module.parameters() recurses, so a
        bare count of the decoder silently includes the whole encoder. That
        made seg_head report 31,113,098 instead of 17,106,986 and inflated the
        site's block total to 69,374,891 against a real 55,368,779. Exclude by
        parameter identity so the fix does not depend on the attribute name.
        """
        skip = {id(q) for e in exclude for q in e.parameters()}
        return int(sum(q.numel() for q in m.parameters() if id(q) not in skip))

    inner = net.network
    enc, dec = inner.encoder, inner.decoder
    fpn = inner.feature_fusion_block
    cb, clf = inner.conv_block, inner.classifier
    c, d, h, w = input_shape
    nodes = [
        {"id": "input", "name": "CT patch (CTNormalization)", "type": "input",
         "out_shape": [c, d, h, w], "param_count": 0, "cam_tap": False},
        {"id": "encoder", "name": "PlainConvUNet encoder (6 stages, 32->320)",
         "type": "conv", "out_shape": [320, d // 4, h // 32, w // 32],
         "param_count": pcount(enc), "cam_tap": False},
        {"id": "fpn", "name": "FPN fusion of skips[-3:] (256,320,320 -> 320)",
         "type": "conv", "out_shape": [320, d // 4, h // 8, w // 8],
         "param_count": pcount(fpn), "cam_tap": False},
        {"id": "conv_block", "name": "Conv block 320->640 (CAM tap)", "type": "conv",
         "out_shape": [640, d // 4, h // 8, w // 8],
         "param_count": pcount(cb), "cam_tap": True},
        {"id": "pool", "name": "Global avg pool", "type": "pool",
         "out_shape": [640, 1, 1, 1], "param_count": 0, "cam_tap": False},
        {"id": "classifier", "name": "Linear 640->320->1", "type": "linear",
         "out_shape": [1], "param_count": pcount(clf), "cam_tap": False},
        {"id": "logits", "name": "Malignancy logit (single, sigmoid)", "type": "output",
         "out_shape": [1], "param_count": 0, "cam_tap": False},
        {"id": "seg_head", "name": "Decoder -> nodule segmentation", "type": "output",
         "out_shape": [2, d, h, w], "param_count": pcount(dec, exclude=(enc,)),
         "cam_tap": False},
    ]
    return {"schema": SCHEMA, "input_shape": [c, d, h, w], "cam_tap": "conv_block",
            "nodes": nodes,
            "notes": ("Joint segmentation + classification. The classification "
                      "branch ignores the decoder output; the decoder is shown "
                      "because its predicted mask cross-checks attribution.")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--data-root", type=Path, default=REPO / "data/luna25")
    ap.add_argument("--out", type=Path, default=REPO / "web/public/data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="path to fold_<N>/checkpoint_best.pth; defaults to "
                         "artifacts/jsc/fold_3/checkpoint_best.pth")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pred = jsc_gradcam.build_predictor(args.device, args.checkpoint)
    net = jsc_gradcam.JSCAdapter(pred.network).eval()
    device = pred.device

    import csv
    labels = {r["identifier"]: int(r["label"]) for r in
              csv.DictReader(open(REPO / "artifacts/jsc/luna25_meta/cls_data.csv"))}

    site_keys = [k for k, _ in METHOD_KEYS]
    examples: list[str] = []
    per_example: dict[str, dict] = {k: {} for k in site_keys}
    out_root = args.out
    (out_root / "examples").mkdir(parents=True, exist_ok=True)

    for idx, case in enumerate(args.cases):
        x, gt, _props = jsc_gradcam.preprocess_case(pred, case, args.data_root)
        x_batch = x[None].to(device)

        with torch.no_grad():
            seg_out, cls_out = net(x_batch)
            logit = float(cls_out.reshape(-1)[0])
            pred_mask = torch.argmax(seg_out, dim=1)[0].cpu().numpy().astype(np.uint8)
        pred_label = 1 if logit > 0 else 0
        true_label = labels[case]

        # Key slice: the GT nodule's centre of mass, so the figure shows lesion.
        zs = np.where(gt.sum(axis=(1, 2)) > 0)[0]
        z = int(round(float(zs.mean()))) if zs.size else gt.shape[0] // 2

        ex_id = f"{idx:02d}_{case}"
        ex_dir = out_root / "examples" / ex_id
        (ex_dir / "attributions").mkdir(parents=True, exist_ok=True)
        examples.append(ex_id)

        ct_vol = x[0].numpy()
        ct_sl = ct_vol[z]
        gray_png(ct_sl, ex_dir / "ct_slice.png")
        (ex_dir / "ct_slice.json").write_text(json.dumps(slice_payload(ct_sl, False)))
        mask_png(ct_sl, gt[z], ex_dir / "mask_slice.png")
        mask_png(ct_sl, pred_mask[z], ex_dir / "pred_mask_slice.png", colour=(1.0, 0.55, 0.0))

        acts = activations_payload(net, x_batch)
        (ex_dir / "activations.json").write_text(json.dumps(acts))

        stages = [net.conv_block]
        stage_names = ["stage0"]
        target_shape = tuple(ct_vol.shape)
        target_cls, _ = gradcam3d_viz._pick_target_class(
            gradcam3d_viz._as_two_column(cls_out), -1)

        cfg = gradcam3d_viz.GradcamConfig(
            get_stages_fn=lambda m: [m.conv_block],
            extract_logits_fn=lambda out: out[1],
            class_names={0: "Benign", 1: "Malignant"},
            methods=[internal for _s, internal in METHOD_KEYS],
            out_dir=str(REPO / "artifacts/jsc/_web_tmp"),
            occ_mask_size=16, ig_n_steps=32,
        )

        raws: dict[str, np.ndarray] = {}
        raws["notgradcam"] = gradcam3d_viz._compute_notgradcam(
            net, x_batch, stages, stage_names, target_shape, case)["stage0"]
        tg = gradcam3d_viz._compute_truegradcam(
            net, x_batch, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, case)
        raws["truegradcam"] = tg["stage0"]
        raws["guided_gradcam"] = gradcam3d_viz._compute_guided_gradcam(
            net, x_batch, tg, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, case)["stage0"]
        raws["layercam"] = gradcam3d_viz._compute_layercam(
            net, x_batch, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, case)["stage0"]
        raws["occlusion"] = gradcam3d_viz._compute_occlusion(
            net, x_batch, target_shape, target_cls, cfg.extract_logits_fn, cfg, case)
        raws["integrated_gradients"] = gradcam3d_viz._compute_integrated_gradients(
            net, x_batch, target_shape, target_cls, cfg.extract_logits_fn, cfg, case)
        raws["integrated_gradcam"] = gradcam3d_viz._compute_integrated_gradcam(
            net, x_batch, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, cfg, case)["stage0"]

        ex_metrics = {}
        for site_key, internal in METHOD_KEYS:
            heat = normalise01(raws[internal])
            m = metrics_for(heat, gt)
            ex_metrics[site_key] = m
            per_example[site_key][ex_id] = m
            sl = heat[z]
            overlay_png(ct_sl, sl, ex_dir / "attributions" / f"{site_key}.png")
            (ex_dir / "attributions" / f"{site_key}.json").write_text(
                json.dumps(slice_payload(sl, True)))

        meta = {
            "example_id": ex_id, "case_id": case,
            "true_label": true_label, "pred_label": pred_label,
            "logits": [logit],
            "prob_malignant": float(torch.sigmoid(torch.tensor(logit))),
            "z_slice": z,
            "input_shape": list(ct_vol.shape),
            "methods": site_keys,
            "gt_voxels": int(gt.sum()),
            "gt_volume_fraction": float(gt.sum() / gt.size),
            "pred_mask_voxels": int((pred_mask > 0).sum()),
            "seg_dice": float(2 * ((pred_mask > 0) & (gt > 0)).sum() /
                              max((pred_mask > 0).sum() + (gt > 0).sum(), 1)),
            "outcome": ("TP" if true_label == 1 and pred_label == 1 else
                        "TN" if true_label == 0 and pred_label == 0 else
                        "FP" if pred_label == 1 else "FN"),
            "metrics": ex_metrics,
        }
        (ex_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[web] {ex_id}: logit={logit:+.3f} pred={pred_label} true={true_label} "
              f"{meta['outcome']} z={z} dice={meta['seg_dice']:.3f} gt={meta['gt_voxels']}v")

    metric_keys = ["mass_in_gt", "inside_outside_ratio", "pointing_acc", "enrichment"]
    aggregate = {
        k: ({mk: float(np.mean([per_example[k][e][mk] for e in examples]))
             for mk in metric_keys} if examples else {mk: 0.0 for mk in metric_keys})
        for k in site_keys
    }
    (out_root / "benchmark.json").write_text(json.dumps({
        "schema": SCHEMA, "methods": site_keys, "examples": examples,
        "per_example": per_example, "aggregate": aggregate,
    }))

    ckpt = Path(args.checkpoint) if args.checkpoint else REPO / "artifacts/jsc/fold_3/checkpoint_best.pth"
    lock = REPO / "uv.lock"
    (out_root / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "uv run python scripts/jsc_web_export.py",
        "model": {
            "name": "JSC (bowang-lab) nnUNetCLSTrainerMTL, PlainConvUNet 3d_fullres",
            "repo": "https://github.com/bowang-lab/JSC",
            "commit": "49511ef01c414014afb7e7a3265d820544bf93cc",
            "dataset": "FLARE-AutoMSC Dataset005_LUNA25",
            "fold": 3,
            "published_val_auc": 0.8875,
        },
        "checkpoint": {"path": str(ckpt.relative_to(REPO)), "sha256": sha256(ckpt)},
        "uv_lock_sha256": sha256(lock) if lock.exists() else None,
        "git": git_info(),
        "methods": site_keys,
        "examples": examples,
        "num_examples": len(examples),
        "deviations": [
            "TTA (mirroring) disabled: gradients would live in flipped frames.",
            "eval() not train(): the official script's .train() activates "
            "Dropout(0.5) and makes every run non-reproducible.",
            "No geometric resampling: LUNA25 ships pre-cropped to the "
            "(64,128,128) patch size, so resampling would misalign image and mask.",
            "TF32 disabled on CUDA.",
        ],
        "policy": "Every displayed figure and number is generated by this exporter and hashed here.",
    }, indent=2))

    (out_root / "model_graph.json").write_text(
        json.dumps(model_graph(net, [1, *ct_vol.shape]), indent=2))

    print(f"\n[web] bundle -> {out_root}")
    print(f"[web] {len(examples)} examples x {len(site_keys)} methods, schema {SCHEMA}")
    print("\n=== aggregate enrichment (uniform heatmap = 1.0) ===")
    for k in site_keys:
        a = aggregate[k]
        print(f"  {k:22s} enrich={a['enrichment']:7.2f} ratio={a['inside_outside_ratio']:6.2f} "
              f"mass={a['mass_in_gt']:.4f} point={a['pointing_acc']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
