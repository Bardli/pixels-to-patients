from __future__ import annotations

import torch

from gradcam_repro.web_export import build_activation_summary


def test_build_activation_summary(real_ct_model, fake_sample) -> None:
    x = fake_sample["image"].unsqueeze(0)
    _, features = real_ct_model(x, return_features=True)
    summary = build_activation_summary(features, fake_sample, max_channels=4)
    layers = [entry["layer"] for entry in summary]
    assert layers == ["stage1", "stage2", "stage3"]
    stage2 = summary[1]
    assert stage2["feature_shape"][0] == 16
    assert len(stage2["channels"]) == 4
    assert stage2["channels"][0]["slice"]["shape"] == stage2["feature_shape"][2:]
    assert len(stage2["histogram"]["counts"]) == 20
    assert len(stage2["histogram"]["bins"]) == 21
