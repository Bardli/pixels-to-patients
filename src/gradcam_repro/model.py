from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ToyCNN(nn.Module):
    """Three-stage CNN matching the architecture sketched in the PPT.

    Shapes: input 1x24x24x24 -> stage1 8x12x12x12 ->
    stage2 16x6x6x6 -> stage3 32x6x6x6 -> spatial evidence head -> 2 logits.

    The task needs absolute left/right position. Instead of a learned 8x8
    collapse kernel, the head explicitly mean-pools evidence over the left and right
    halves. That keeps the classifier faithful to the synthetic rule while
    avoiding Grad-CAM edge artifacts from position-specific classifier weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),
        )
        self.stage3_conv = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=1),
            nn.ReLU(inplace=False),
        )
        self.evidence_head = nn.Conv3d(32, 1, kernel_size=1)
        self.logit_scale = nn.Parameter(torch.tensor(4.0))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features: dict[str, torch.Tensor] = {}
        x = self.stage1(x)
        features["stage1"] = x
        x = self.stage2(x)
        features["stage2"] = x
        x = self.stage3_conv(x)
        features["stage3"] = x
        evidence = F.softplus(self.evidence_head(x))
        width = evidence.shape[-1]
        left = evidence[:, :, :, :, : width // 2].mean(dim=(1, 2, 3, 4))
        right = evidence[:, :, :, :, width // 2 :].mean(dim=(1, 2, 3, 4))
        logits = self.logit_scale.clamp_min(0.1) * torch.stack((left, right), dim=1)
        if return_features:
            return logits, features
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RealCtCNN(nn.Module):
    """Small generic 3D CNN for real CT classification examples.

    It keeps the same `stage2` tap point used by the PPT attribution methods,
    but uses a standard global-pooling classifier for the real CT patch task.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features: dict[str, torch.Tensor] = {}
        x = self.stage1(x)
        features["stage1"] = x
        x = self.stage2(x)
        features["stage2"] = x
        x = self.stage3(x)
        features["stage3"] = x
        pooled = self.pool(x).flatten(1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, features
        return logits
