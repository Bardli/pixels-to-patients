from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gradcam_repro.model import RealCtCNN


@pytest.fixture
def real_ct_model() -> RealCtCNN:
    torch.manual_seed(0)
    model = RealCtCNN()
    model.eval()
    return model


def _cube_mask(size: int, half: int = 3) -> torch.Tensor:
    mask = torch.zeros(1, size, size, size)
    c = size // 2
    mask[:, c - half : c + half, c - half : c + half, c - half : c + half] = 1.0
    return mask


@pytest.fixture
def fake_sample() -> dict[str, torch.Tensor | str]:
    torch.manual_seed(1)
    size = 16
    return {
        "image": torch.rand(1, size, size, size),
        "mask": _cube_mask(size),
        "label": torch.tensor(1, dtype=torch.long),
        "center": torch.tensor([size // 2, size // 2, size // 2], dtype=torch.long),
        "case_id": "fake_0",
    }


@pytest.fixture
def fake_cache(tmp_path: Path) -> Path:
    torch.manual_seed(2)
    size = 16
    n = 3
    images = torch.rand(n, 1, size, size, size)
    masks = torch.stack([_cube_mask(size), torch.zeros(1, size, size, size), _cube_mask(size)])
    labels = torch.tensor([1, 0, 1], dtype=torch.long)
    centers = torch.tensor([[size // 2] * 3] * n, dtype=torch.long)
    payload = {
        "schema": "gradcam-repro.real-ct-msd-lung.v2",
        "source": {"dataset": "synthetic-test"},
        "config": {},
        "images": images,
        "masks": masks,
        "labels": labels,
        "centers": centers,
        "case_ids": [f"fake_{i}" for i in range(n)],
        "source_images": ["x"] * n,
        "source_masks": ["x"] * n,
        "splits": {"train": [0], "val": [1], "test": [0, 1, 2]},
    }
    path = tmp_path / "fake_cache.pt"
    torch.save(payload, path)
    return path
