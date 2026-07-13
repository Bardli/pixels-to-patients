from __future__ import annotations

from pathlib import Path

import torch

from gradcam_repro.web_export import export_example


def test_export_example_writes_files(real_ct_model, fake_sample, tmp_path: Path) -> None:
    methods = ["gradcam", "layercam"]
    device = torch.device("cpu")
    meta = export_example(real_ct_model, fake_sample, methods, device, tmp_path, "ex0")
    root = tmp_path / "ex0"
    assert (root / "ct_slice.png").exists()
    assert (root / "ct_slice.json").exists()
    assert (root / "mask_slice.png").exists()
    assert (root / "activations.json").exists()
    for m in methods:
        assert (root / "attributions" / f"{m}.png").exists()
        assert (root / "attributions" / f"{m}.json").exists()
    assert meta["example_id"] == "ex0"
    assert meta["case_id"] == "fake_0"
    assert set(meta["metrics"].keys()) == set(methods)
    assert set(meta["metrics"]["gradcam"].keys()) == {"mass_in_gt", "inside_outside_ratio", "pointing_acc"}
    assert meta["pred_label"] in (0, 1)
    assert len(meta["logits"]) == 2
