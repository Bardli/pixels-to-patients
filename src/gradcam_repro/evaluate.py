from __future__ import annotations

import json
from pathlib import Path

import torch

from .attribution import METHODS
from .data import CrossHalfDataset
from .model import ToyCNN
from .visualize import DEFAULT_METHODS


def score_single_sample(heatmap: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    inv_mask = 1.0 - mask
    mass = heatmap.sum().clamp_min(1e-8)
    inside_mass = (heatmap * mask).sum()
    inside_mean = inside_mass / mask.sum().clamp_min(1)
    outside_mean = (heatmap * inv_mask).sum() / inv_mask.sum().clamp_min(1)
    peak_idx = int(heatmap.flatten().argmax().item())
    pointing_hit = float(mask.flatten()[peak_idx].item() > 0)
    return {
        "mass_in_gt": float((inside_mass / mass).item()),
        "inside_outside_ratio": float((inside_mean / outside_mean.clamp_min(1e-8)).item()),
        "pointing_acc": pointing_hit,
    }


def score_attributions(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    device: torch.device,
    methods: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    methods = methods or DEFAULT_METHODS
    totals = {
        method: {"mass_in_gt": 0.0, "inside_outside_ratio": 0.0, "pointing_acc": 0.0}
        for method in methods
    }
    for sample in dataset:
        image = sample["image"].unsqueeze(0).to(device)
        label = sample["label"].view(1).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        for method in methods:
            heatmap = METHODS[method](model, image, label)
            per = score_single_sample(heatmap, mask)
            for key, value in per.items():
                totals[method][key] += value
    n = len(dataset)
    return {
        method: {metric: value / n for metric, value in metrics.items()}
        for method, metrics in totals.items()
    }


def save_scores(scores: dict[str, dict[str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2))
