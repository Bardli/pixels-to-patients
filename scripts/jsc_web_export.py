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


def occlusion_terms(occluded_prob: np.ndarray, prob_intact: float) -> dict:
    """Turn the occluded-probability field into the three panels the page steps
    through: the intact score, the drop, and the occluded score.

    MONAI owns the sliding-window loop, so the per-cube probabilities are not
    computed by us -- but they are exactly what its map holds once you stop
    treating it as a saliency map. Occluding a region can only lower the score,
    so H(p) = P(c | x) - P(c | x occluded at p). Derived here rather than in the
    browser so the numbers sit under the manifest's hash.

    `prob_intact` must come from gradcam3d_viz.occlusion_intact_probability, not
    from meta.json's prob_malignant: MONAI softmaxes our two-column [-z, +z]
    form, so its scale is sigmoid(2z) while prob_malignant is sigmoid(z).
    """
    if not 0.0 <= prob_intact <= 1.0:
        raise ValueError(f"prob_intact {prob_intact} is not a probability")
    lo, hi = float(np.min(occluded_prob)), float(np.max(occluded_prob))
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(
            f"occluded field spans [{lo:.4f}, {hi:.4f}], which is not in "
            "probability units -- MONAI's activate=True was likely disabled"
        )
    drop = prob_intact - occluded_prob
    over = float(np.max(drop)) - prob_intact
    if over > 1e-6:
        raise ValueError(
            f"drop max {float(np.max(drop)):.4f} exceeds the intact probability "
            f"{prob_intact:.4f} by {over:.4f}; the maps are not on one scale"
        )
    return {"intact": float(prob_intact),
            "drop": drop.astype(np.float32),
            "occluded": occluded_prob.astype(np.float32)}


TAP_NAME = "conv_block"
TAP_SHAPE = [640, 16, 16, 16]
# Cap for decomposition slices. The CAM family is already 16x16; the
# input-resolution methods (Integrated Gradients, occlusion) are 128x128, and
# four path checkpoints at that size cost 235 KB per case per method -- an order
# of magnitude over the bundle's budget for something that renders into a 288 px
# canvas with image-rendering: pixelated. Block-mean to at most 64x64, which is
# the 4096 values the panel is meant to expose one cell at a time.
DECOMP_MAX_SIDE = 64


def _shrink(sl: np.ndarray, max_side: int = DECOMP_MAX_SIDE) -> np.ndarray:
    """Block-mean a 2D slice down to at most max_side, by an integer factor."""
    h, w = sl.shape
    f = max(1, int(np.ceil(max(h, w) / max_side)))
    if f == 1:
        return sl
    hh, ww = (h // f) * f, (w // f) * f
    return sl[:hh, :ww].reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))


def decomp_payload(terms: dict, z_input: int) -> dict:
    """One method's computation terms as quantised 2D slices.

    Terms arrive at whichever resolution their method works in -- the CAM family
    at the 16^3 tap, Integrated Gradients and occlusion at the 64x128x128 input
    -- so the slice index is rescaled per term instead of assumed. `signed`
    tells the renderer to centre a diverging scale at zero; without it a large
    negative would paint like a large positive, which is the very failure the
    Grad-CAM page is trying to teach.
    """
    payload: dict = {"tap": TAP_NAME, "tap_shape": TAP_SHAPE, "terms": {}}
    for key in ("channel", "alpha", "alpha_negative", "grad_is_constant",
                "intact", "steps"):
        if key in terms:
            v = terms[key]
            payload[key] = v.item() if hasattr(v, "item") else v

    def add(name: str, vol: np.ndarray) -> None:
        fz = min(vol.shape[0] - 1, int(round(z_input * vol.shape[0] / 64)))
        signed = bool(vol.min() < 0.0)
        p = slice_payload(_shrink(vol[fz]), False)
        p["signed"] = signed
        p["feature_z"] = fz
        p["depth"] = int(vol.shape[0])
        payload["terms"][name] = p

    for name, vol in terms.items():
        if isinstance(vol, np.ndarray) and vol.ndim == 3:
            add(name, vol)

    # Per-channel vectors (alpha across 640 channels, or notGradCAM's channel
    # means). These are the honest substitute for a gradient panel that would
    # otherwise be one flat colour: under a globally pooled head the gradient
    # carries no spatial structure, so the class signal lives in this spread.
    for name in ("alpha_all", "channel_means"):
        if name in terms:
            v = np.asarray(terms[name], dtype=np.float32)
            payload[name] = {"values": [round(float(u), 8) for u in v],
                             "vmin": float(v.min()), "vmax": float(v.max()),
                             "signed": bool(v.min() < 0.0)}

    if "path" in terms:
        payload["path"] = []
        for s in terms["path"]:
            vol = np.asarray(s["map"], dtype=np.float32)
            fz = min(vol.shape[0] - 1, int(round(z_input * vol.shape[0] / 64)))
            signed = bool(vol.min() < 0.0)
            entry = slice_payload(_shrink(vol[fz]), False)
            entry["signed"] = signed
            entry["frac"] = float(s["frac"])
            payload["path"].append(entry)
    return payload


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
        decomps: dict[str, dict] = {}     # site_key -> terms dict

        ng = gradcam3d_viz._compute_notgradcam(
            net, x_batch, stages, stage_names, target_shape, case, return_terms=True)
        raws["notgradcam"] = ng["raw"]["stage0"]
        decomps["notgradcam"] = ng["terms"]["stage0"]

        # The exported Grad-CAM map stays MONAI's; the hand-written version
        # supplies terms only. Parity between them is gated at the tap by
        # tests/test_gradcam_parity.py.
        tg = gradcam3d_viz._compute_truegradcam(
            net, x_batch, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, case)
        raws["truegradcam"] = tg["stage0"]
        decomps["gradcam"] = gradcam3d_viz._compute_gradcam_decomposed(
            net, x_batch, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, case)["terms"]["stage0"]

        raws["guided_gradcam"] = gradcam3d_viz._compute_guided_gradcam(
            net, x_batch, tg, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, case)["stage0"]

        lc = gradcam3d_viz._compute_layercam(
            net, x_batch, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, case, return_terms=True)
        raws["layercam"] = lc["raw"]["stage0"]
        decomps["layercam"] = lc["terms"]["stage0"]

        oc = gradcam3d_viz._compute_occlusion(
            net, x_batch, target_shape, target_cls, cfg.extract_logits_fn, cfg,
            case, return_terms=True)
        raws["occlusion"] = oc["raw"]
        p_intact = gradcam3d_viz.occlusion_intact_probability(
            gradcam3d_viz._as_two_column(cls_out), target_cls)
        ot = occlusion_terms(oc["terms"]["input_resolution"]["occluded_prob"], p_intact)
        decomps["occlusion"] = {"channel": None, **ot}

        ig = gradcam3d_viz._compute_integrated_gradients(
            net, x_batch, target_shape, target_cls, cfg.extract_logits_fn, cfg,
            case, return_terms=True)
        raws["integrated_gradients"] = ig["raw"]
        decomps["integrated_gradients"] = ig["terms"]["input_resolution"]

        igc = gradcam3d_viz._compute_integrated_gradcam(
            net, x_batch, stages, stage_names, target_shape, target_cls,
            cfg.extract_logits_fn, cfg, case, return_terms=True)
        raws["integrated_gradcam"] = igc["raw"]["stage0"]
        decomps["integrated_gradcam"] = igc["terms"]["stage0"]

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

            terms = decomps.get(site_key)
            if terms:
                (ex_dir / "attributions" / f"{site_key}.decomp.json").write_text(
                    json.dumps(decomp_payload(terms, z)))

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
