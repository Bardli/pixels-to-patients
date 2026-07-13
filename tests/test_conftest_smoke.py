from __future__ import annotations

from pathlib import Path

from gradcam_repro.real_ct import RealCtDataset


def test_fake_cache_loads(fake_cache: Path) -> None:
    ds = RealCtDataset(fake_cache, split="test")
    assert len(ds) == 3
    item = ds[0]
    assert item["image"].shape == (1, 16, 16, 16)
    assert item["mask"].shape == (1, 16, 16, 16)


def test_model_forward_features(real_ct_model, fake_sample) -> None:
    x = fake_sample["image"].unsqueeze(0)
    logits, features = real_ct_model(x, return_features=True)
    assert logits.shape == (1, 2)
    assert features["stage2"].shape[1] == 16
