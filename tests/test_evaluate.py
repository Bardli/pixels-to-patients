from __future__ import annotations

import torch

from gradcam_repro.evaluate import score_single_sample


def test_score_single_sample_perfect_localization() -> None:
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[:, :, 1:3, 1:3, 1:3] = 1.0
    heatmap = mask.clone()  # all mass inside GT, peak inside
    scores = score_single_sample(heatmap, mask)
    assert scores["mass_in_gt"] == 1.0
    assert scores["pointing_acc"] == 1.0
    assert scores["inside_outside_ratio"] > 1.0


def test_score_single_sample_miss() -> None:
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[:, :, 0, 0, 0] = 1.0
    heatmap = torch.zeros(1, 1, 4, 4, 4)
    heatmap[:, :, 3, 3, 3] = 1.0  # peak outside GT
    scores = score_single_sample(heatmap, mask)
    assert scores["mass_in_gt"] == 0.0
    assert scores["pointing_acc"] == 0.0


def test_score_single_sample_empty_mask() -> None:
    mask = torch.zeros(1, 1, 4, 4, 4)
    heatmap = torch.rand(1, 1, 4, 4, 4)
    scores = score_single_sample(heatmap, mask)
    assert scores["mass_in_gt"] == 0.0
    assert scores["inside_outside_ratio"] == 0.0
    assert scores["pointing_acc"] == 0.0
