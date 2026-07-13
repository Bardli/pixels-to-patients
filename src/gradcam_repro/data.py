from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToyDataConfig:
    volume_size: int = 24
    cross_size: int = 7
    cross_thickness: int = 1
    gt_window: int = 7
    distractors: int = 4
    box_size_min: int = 3
    box_size_max: int = 6
    noise_std: float = 0.10
    target_intensity: float = 1.0
    distractor_intensity: float = 0.35


class CrossHalfDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic 3D task from the slide deck.

    A bright 1-voxel-wide 3-axis cross determines the label: class 0 if it is
    in the left half, class 1 if it is in the right half. Dim 3D box distractors
    create red herrings.
    """

    def __init__(
        self,
        n_samples: int,
        seed: int,
        config: ToyDataConfig | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.config = config or ToyDataConfig()
        self.images, self.labels, self.masks, self.centers = self._generate(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "mask": self.masks[idx],
            "center": self.centers[idx],
        }

    def _generate(
        self, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        rng = np.random.default_rng(seed)
        size = cfg.volume_size
        images = np.zeros((self.n_samples, 1, size, size, size), dtype=np.float32)
        labels = np.zeros((self.n_samples,), dtype=np.int64)
        masks = np.zeros_like(images)
        centers = np.zeros((self.n_samples, 3), dtype=np.int64)

        margin = cfg.gt_window // 2 + 1
        left_x = np.arange(margin, size // 2 - margin)
        right_x = np.arange(size // 2 + margin, size - margin)
        valid_y = np.arange(margin, size - margin)
        valid_z = np.arange(margin, size - margin)

        for i in range(self.n_samples):
            label = int(rng.integers(0, 2))
            cx = int(rng.choice(left_x if label == 0 else right_x))
            cy = int(rng.choice(valid_y))
            cz = int(rng.choice(valid_z))
            image = rng.normal(0.0, cfg.noise_std, (size, size, size)).astype(np.float32)

            self._draw_box_distractors(image, rng, cfg, target_center=(cz, cy, cx))
            self._draw_cross3d(
                image,
                cz,
                cy,
                cx,
                cfg.target_intensity,
                cfg.cross_size,
                cfg.cross_thickness,
            )
            mask = self._draw_gt_mask(size, cz, cy, cx, cfg.gt_window)

            image = np.clip(image, -0.35, 1.0)
            image = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)

            images[i, 0] = image
            labels[i] = label
            masks[i, 0] = mask
            centers[i] = (cz, cy, cx)

        return (
            torch.from_numpy(images),
            torch.from_numpy(labels),
            torch.from_numpy(masks),
            torch.from_numpy(centers),
        )

    @staticmethod
    def _draw_cross3d(
        image: np.ndarray,
        cz: int,
        cy: int,
        cx: int,
        intensity: float,
        cross_size: int,
        thickness: int,
    ) -> None:
        radius = cross_size // 2
        half_thick = thickness // 2
        z0, z1 = cz - half_thick, cz + half_thick + 1
        y0, y1 = cy - half_thick, cy + half_thick + 1
        x0, x1 = cx - half_thick, cx + half_thick + 1
        image[z0:z1, y0:y1, cx - radius : cx + radius + 1] += intensity
        image[z0:z1, cy - radius : cy + radius + 1, x0:x1] += intensity
        image[cz - radius : cz + radius + 1, y0:y1, x0:x1] += intensity

    @staticmethod
    def _draw_gt_mask(image_size: int, cz: int, cy: int, cx: int, window: int) -> np.ndarray:
        radius = window // 2
        mask = np.zeros((image_size, image_size, image_size), dtype=np.float32)
        z0, z1 = max(0, cz - radius), min(image_size, cz + radius + 1)
        x0, x1 = max(0, cx - radius), min(image_size, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(image_size, cy + radius + 1)
        mask[z0:z1, y0:y1, x0:x1] = 1.0
        return mask

    @staticmethod
    def _draw_box_distractors(
        image: np.ndarray,
        rng: np.random.Generator,
        cfg: ToyDataConfig,
        target_center: tuple[int, int, int],
    ) -> None:
        cz, cy, cx = target_center
        target_radius = cfg.gt_window // 2 + 1
        target_box = (
            cz - target_radius,
            cz + target_radius + 1,
            cy - target_radius,
            cy + target_radius + 1,
            cx - target_radius,
            cx + target_radius + 1,
        )

        def overlaps_target(box: tuple[int, int, int, int, int, int]) -> bool:
            z0, z1, y0, y1, x0, x1 = box
            tz0, tz1, ty0, ty1, tx0, tx1 = target_box
            return z0 < tz1 and z1 > tz0 and y0 < ty1 and y1 > ty0 and x0 < tx1 and x1 > tx0

        for distractor_idx in range(cfg.distractors):
            placed_box: tuple[int, int, int, int, int, int] | None = None
            for _attempt in range(50):
                depth = int(rng.integers(cfg.box_size_min, cfg.box_size_max + 1))
                height = int(rng.integers(cfg.box_size_min, cfg.box_size_max + 1))
                width = int(rng.integers(cfg.box_size_min, cfg.box_size_max + 1))
                if distractor_idx == 0:
                    z0 = max(1, min(cfg.volume_size - depth - 1, cz - depth // 2))
                else:
                    z0 = int(rng.integers(1, cfg.volume_size - depth - 1))
                y0 = int(rng.integers(1, cfg.volume_size - height - 1))
                x0 = int(rng.integers(1, cfg.volume_size - width - 1))
                box = (
                    z0,
                    z0 + depth,
                    y0,
                    y0 + height,
                    x0,
                    x0 + width,
                )
                if not overlaps_target(box):
                    placed_box = box
                    break
            if placed_box is None:
                continue
            z0, z1, y0, y1, x0, x1 = placed_box
            image[z0:z1, y0:y1, x0:x1] += cfg.distractor_intensity * 0.55
            image[z0:z1, y0, x0:x1] += cfg.distractor_intensity * 0.45
            image[z0:z1, y1 - 1, x0:x1] += cfg.distractor_intensity * 0.45
            image[z0:z1, y0:y1, x0] += cfg.distractor_intensity * 0.45
            image[z0:z1, y0:y1, x1 - 1] += cfg.distractor_intensity * 0.45
            image[z0, y0:y1, x0:x1] += cfg.distractor_intensity * 0.45
            image[z1 - 1, y0:y1, x0:x1] += cfg.distractor_intensity * 0.45


def make_splits(
    train_size: int = 1200,
    val_size: int = 200,
    test_size: int = 120,
    seed: int = 7,
) -> tuple[CrossHalfDataset, CrossHalfDataset, CrossHalfDataset]:
    return (
        CrossHalfDataset(train_size, seed=seed),
        CrossHalfDataset(val_size, seed=seed + 1),
        CrossHalfDataset(test_size, seed=seed + 2),
    )
