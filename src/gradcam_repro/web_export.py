from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .cli import sha256_file
from .attribution import METHODS
from .evaluate import score_single_sample
from .model import RealCtCNN
from .real_ct import RealCtDataset, load_real_ct_model
from .train import resolve_device, set_seed
from .visualize import DEFAULT_METHODS, feature_z, overlay_heatmap, sample_z, volume_slice

WEB_BUNDLE_SCHEMA = "gradcam-repro.web-bundle.v1"

_NODE_META = {
    "stage1": ("Stage 1 conv block", "conv"),
    "stage2": ("Stage 2 conv block (CAM tap)", "conv"),
    "stage3": ("Stage 3 conv", "conv"),
}


def _param_count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def build_model_graph(model: RealCtCNN, input_shape: tuple[int, int, int]) -> dict:
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 1, *input_shape, device=device)
    with torch.no_grad():
        logits, features = model(dummy, return_features=True)

    def out_shape(t: torch.Tensor) -> list[int]:
        return list(t.shape[1:])

    nodes: list[dict] = [
        {"id": "input", "name": "CT patch", "type": "input",
         "out_shape": [1, *input_shape], "param_count": 0, "cam_tap": False},
    ]
    for stage_id in ("stage1", "stage2", "stage3"):
        name, kind = _NODE_META[stage_id]
        nodes.append({
            "id": stage_id, "name": name, "type": kind,
            "out_shape": out_shape(features[stage_id]),
            "param_count": _param_count(getattr(model, stage_id)),
            "cam_tap": stage_id == "stage2",
        })
    nodes.append({"id": "pool", "name": "Global avg pool", "type": "pool",
                  "out_shape": [features["stage3"].shape[1], 1, 1, 1],
                  "param_count": 0, "cam_tap": False})
    nodes.append({"id": "classifier", "name": "Linear classifier", "type": "linear",
                  "out_shape": [int(logits.shape[1])],
                  "param_count": _param_count(model.classifier), "cam_tap": False})
    nodes.append({"id": "logits", "name": "Class logits", "type": "output",
                  "out_shape": [int(logits.shape[1])], "param_count": 0, "cam_tap": False})
    return {
        "schema": WEB_BUNDLE_SCHEMA,
        "input_shape": [1, *input_shape],
        "cam_tap": "stage2",
        "nodes": nodes,
    }


def slice_to_payload(slice2d: torch.Tensor, already_normalized: bool = False) -> dict:
    data = slice2d.detach().float().cpu()
    vmin = float(data.min().item())
    vmax = float(data.max().item())
    if already_normalized:
        scaled = data.clamp(0.0, 1.0)
    else:
        scaled = (data - vmin) / (vmax - vmin + 1e-8)
    quantised = (scaled * 255.0).round().clamp(0, 255).to(torch.uint8)
    return {
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "values": [int(v) for v in quantised.flatten().tolist()],
        "vmin": vmin,
        "vmax": vmax,
    }


def _histogram(tensor: torch.Tensor, hist_bins: int) -> dict:
    flat = tensor.detach().float().cpu().flatten()
    counts = torch.histc(flat, bins=hist_bins, min=float(flat.min()), max=float(flat.max()))
    lo, hi = float(flat.min()), float(flat.max())
    edges = torch.linspace(lo, hi, hist_bins + 1)
    return {"bins": [float(v) for v in edges.tolist()], "counts": [int(v) for v in counts.tolist()]}


def build_activation_summary(
    features: dict[str, torch.Tensor],
    sample: dict,
    max_channels: int = 6,
    hist_bins: int = 20,
) -> list[dict]:
    summary: list[dict] = []
    for layer in ("stage1", "stage2", "stage3"):
        activation = features[layer]  # (1, C, D, H, W)
        fz = feature_z(sample, activation)
        channel_means = activation.flatten(2).mean(dim=2).squeeze(0)  # (C,)
        top = torch.topk(channel_means, k=min(max_channels, activation.shape[1])).indices
        channels = []
        for ch in top.tolist():
            channel_slice = volume_slice(activation[0, ch], fz)  # (H, W)
            channels.append({
                "index": int(ch),
                "slice": slice_to_payload(channel_slice),
                "mean": float(activation[0, ch].mean().item()),
                "max": float(activation[0, ch].max().item()),
            })
        summary.append({
            "layer": layer,
            "feature_shape": list(activation.shape[1:]),
            "feature_z": int(fz),
            "channels": channels,
            "histogram": _histogram(activation, hist_bins),
        })
    return summary


def _save_gray_png(slice2d: torch.Tensor, path: Path) -> None:
    data = slice2d.detach().float().cpu()
    norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    arr = (norm.numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _save_rgb_png(rgb: torch.Tensor, path: Path) -> None:
    arr = (rgb.detach().float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_mask_png(mask_slice: torch.Tensor, ct_slice: torch.Tensor, path: Path) -> None:
    ct = ct_slice.detach().float().cpu()
    ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
    gray = ct.unsqueeze(-1).repeat(1, 1, 3)
    green = torch.zeros_like(gray)
    green[..., 1] = 1.0
    m = mask_slice.detach().float().cpu().unsqueeze(-1)
    blended = torch.where(m > 0, 0.5 * gray + 0.5 * green, gray)
    _save_rgb_png(blended, path)


def export_example(
    model: RealCtCNN,
    sample: dict,
    methods: list[str],
    device: torch.device,
    out_dir: Path,
    example_id: str,
    max_channels: int = 6,
) -> dict:
    root = out_dir / example_id
    (root / "attributions").mkdir(parents=True, exist_ok=True)
    z = sample_z(sample)
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    label = sample["label"].view(1).to(device)

    model.eval()
    with torch.no_grad():
        logits, features = model(image, return_features=True)
    pred_label = int(logits.argmax(dim=1).item())

    _save_gray_png(volume_slice(sample["image"], z), root / "ct_slice.png")
    (root / "ct_slice.json").write_text(json.dumps(slice_to_payload(volume_slice(sample["image"], z))))
    _save_mask_png(volume_slice(sample["mask"], z), volume_slice(sample["image"], z), root / "mask_slice.png")
    (root / "activations.json").write_text(json.dumps(build_activation_summary(features, sample, max_channels=max_channels)))

    metrics: dict[str, dict[str, float]] = {}
    for method in methods:
        heatmap = METHODS[method](model, image, label)  # (1,1,D,H,W) normalised
        metrics[method] = score_single_sample(heatmap, mask)
        overlay = overlay_heatmap(sample["image"], heatmap.cpu(), z)  # (H,W,3)
        _save_rgb_png(overlay, root / "attributions" / f"{method}.png")
        heat_slice = heatmap[0, 0, z]
        (root / "attributions" / f"{method}.json").write_text(
            json.dumps(slice_to_payload(heat_slice, already_normalized=True))
        )

    return {
        "example_id": example_id,
        "case_id": str(sample["case_id"]),
        "true_label": int(sample["label"].item()),
        "pred_label": pred_label,
        "logits": [float(v) for v in logits.squeeze(0).tolist()],
        "z_slice": int(z),
        "input_shape": list(sample["image"].shape[1:]),
        "methods": methods,
        "metrics": metrics,
    }


_METRIC_KEYS = ("mass_in_gt", "inside_outside_ratio", "pointing_acc")


def build_benchmark(example_metas: list[dict], methods: list[str]) -> dict:
    example_ids = [m["example_id"] for m in example_metas]
    per_example: dict[str, dict[str, dict[str, float]]] = {method: {} for method in methods}
    for meta in example_metas:
        for method in methods:
            per_example[method][meta["example_id"]] = meta["metrics"][method]
    aggregate: dict[str, dict[str, float]] = {}
    for method in methods:
        rows = list(per_example[method].values())
        aggregate[method] = {
            key: (sum(r[key] for r in rows) / len(rows)) if rows else 0.0
            for key in _METRIC_KEYS
        }
    return {
        "schema": WEB_BUNDLE_SCHEMA,
        "methods": methods,
        "examples": example_ids,
        "per_example": per_example,
        "aggregate": aggregate,
    }


def _git_provenance() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": False}


def build_manifest(
    checkpoint: Path,
    data_cache: Path,
    uv_lock: Path,
    example_ids: list[str],
    methods: list[str],
    generated_at: str,
) -> dict:
    return {
        "schema": WEB_BUNDLE_SCHEMA,
        "generated_at": generated_at,
        "generated_by": "uv run gradcam-repro web-export",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "data_cache": {"path": str(data_cache), "sha256": sha256_file(data_cache)},
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock.exists() else None,
        "git": _git_provenance(),
        "methods": methods,
        "examples": example_ids,
        "num_examples": len(example_ids),
        "policy": "Every displayed figure and number is generated by web-export and hashed here.",
    }


def web_export(
    cache: Path,
    checkpoint: Path,
    out_dir: Path,
    split: str = "test",
    num_examples: int = 3,
    methods: list[str] | None = None,
    device_name: str = "auto",
    seed: int = 13,
    max_channels: int = 6,
    generated_at: str = "",
    uv_lock: Path = Path("uv.lock"),
) -> dict:
    methods = methods or DEFAULT_METHODS
    set_seed(seed)
    device = resolve_device(device_name)
    model = load_real_ct_model(checkpoint, device)
    dataset = RealCtDataset(cache, split=split, limit=num_examples, positive_only=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = out_dir / "examples"
    shutil.rmtree(examples_dir, ignore_errors=True)
    input_shape = tuple(int(s) for s in dataset[0]["image"].shape[1:])

    (out_dir / "model_graph.json").write_text(json.dumps(build_model_graph(model, input_shape)))

    example_metas: list[dict] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        example_id = f"{idx:02d}_{sample['case_id']}"
        meta = export_example(model, sample, methods, device, examples_dir, example_id, max_channels=max_channels)
        (examples_dir / example_id / "meta.json").write_text(json.dumps(meta))
        example_metas.append(meta)

    (out_dir / "benchmark.json").write_text(json.dumps(build_benchmark(example_metas, methods)))
    manifest = build_manifest(
        checkpoint, cache, uv_lock,
        [m["example_id"] for m in example_metas], methods, generated_at,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "bundle": str(out_dir),
        "num_examples": len(example_metas),
        "methods": methods,
        "manifest": str(out_dir / "manifest.json"),
    }
