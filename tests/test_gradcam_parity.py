"""Grad-CAM by hand must equal MONAI's Grad-CAM.

MONAI returns only the finished map, so the decomposition needs a hand
implementation. That is only safe if the hand version is numerically the same
map -- otherwise the stepper's final ReLU panel would disagree with the
heat-map displayed right above it on the same page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gradcam3d_viz as gv


class TinyNet(nn.Module):
    """Two conv stages then a single logit -- JSC's shape, small enough for CPU."""

    def __init__(self):
        super().__init__()
        self.stage = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        a = self.stage(x)
        return self.head(a.mean(dim=(2, 3, 4)))


def _monai_tap_map(net, x, tap_module, extract_logits_fn, class_idx):
    """MONAI's CAM at tap resolution, before any upsampling.

    `GradCAM.__call__` upsamples to the input size; `compute_map` does not. The
    comparison has to happen here, because our `_upsample_to` uses
    scipy.ndimage.zoom while MONAI upsamples with F.interpolate trilinear.
    Measured on the real network, that interpolation difference alone is 1.2e-01
    in normalised units -- large enough to swamp a parity check that only looks
    at the final 128x128 map, which would then be testing the interpolator
    rather than the CAM arithmetic.
    """
    from monai.visualize import GradCAM

    wrapped = gv._MonaiGradCamWrapper(net, tap_module, extract_logits_fn)
    wrapped = wrapped.to(x.device).eval()
    name = gv._get_module_name(wrapped, tap_module)
    cam = GradCAM(nn_module=wrapped, target_layers=[name])
    return cam.compute_map(x, class_idx=class_idx)[0, 0].detach().cpu().numpy()


def test_hand_gradcam_matches_monai_at_the_tap():
    torch.manual_seed(0)
    net = TinyNet().eval()
    x = torch.randn(1, 1, 8, 8, 8)

    hand = gv._compute_gradcam_decomposed(
        net, x, [net.stage], ["stage0"], (8, 8, 8), 1, lambda o: o,
        "test")["terms"]["stage0"]["relu"]
    monai = _monai_tap_map(net, x, net.stage, lambda o: o, 1)

    n = lambda a: (a - a.min()) / (np.ptp(a) or 1.0)
    assert hand.shape == monai.shape
    assert np.abs(n(hand) - n(monai)).max() < 1e-4, (
        f"max abs diff {np.abs(n(hand) - n(monai)).max():.2e} -- the hand "
        "implementation is not the same map as MONAI's"
    )


def test_gradient_is_spatially_constant_under_a_pooled_head():
    """Guards the fact the stepper's captions rest on.

    With global average pooling between the tap and the classifier,
    d y / d A[k,z,y,x] = (1/N) d y / d pooled_k, so the gradient carries no
    spatial structure at all -- it is one value per channel. If a future model
    breaks this (a head with spatial structure), the gradient panel becomes a
    real texture and the caption claiming otherwise turns into a lie.
    """
    torch.manual_seed(0)
    net = TinyNet().eval()          # stage -> mean over voxels -> Linear
    x = torch.randn(1, 1, 8, 8, 8)
    t = gv._compute_gradcam_decomposed(
        net, x, [net.stage], ["stage0"], (8, 8, 8), 1, lambda o: o,
        "test")["terms"]["stage0"]

    assert t["grad_is_constant"] is True
    assert np.ptp(t["gradient"]) == 0.0, "gradient varies in space under a pooled head"
    # weighted is then activation * scalar, which is why it must not be shown as
    # if it were a separate spatial pattern.
    a, w = t["activation"], t["weighted"]
    if abs(t["alpha"]) > 0:
        np.testing.assert_allclose(w, a * t["alpha"], rtol=1e-5, atol=1e-7)


def test_alpha_vector_is_exported_for_every_channel():
    """The class signal lives in the spread of alpha, so ship all of it."""
    torch.manual_seed(0)
    net = TinyNet().eval()
    x = torch.randn(1, 1, 8, 8, 8)
    t = gv._compute_gradcam_decomposed(
        net, x, [net.stage], ["stage0"], (8, 8, 8), 1, lambda o: o,
        "test")["terms"]["stage0"]

    assert t["alpha_all"].shape == (4,), t["alpha_all"].shape
    assert t["alpha"] == pytest.approx(float(t["alpha_all"][t["channel"]]))
    assert t["alpha_negative"] == int((t["alpha_all"] < 0).sum())


def test_terms_have_tap_shape_and_expected_signs():
    torch.manual_seed(0)
    net = TinyNet().eval()
    x = torch.randn(1, 1, 8, 8, 8)
    out = gv._compute_gradcam_decomposed(
        net, x, [net.stage], ["stage0"], (8, 8, 8), 1, lambda o: o, "test")
    t = out["terms"]["stage0"]

    for key in ("activation", "gradient", "weighted", "summed", "relu"):
        assert t[key].shape == (8, 8, 8), f"{key} has shape {t[key].shape}"
        assert np.isfinite(t[key]).all(), f"{key} has non-finite values"

    assert (t["activation"] >= 0).all(), "post-ReLU activations cannot be negative"
    assert (t["relu"] >= 0).all(), "the ReLU term cannot be negative"
    assert 0 <= t["channel"] < 4
    # The gradient is the one signed term; that sign is the whole reason the
    # gradient panel needs a diverging colour scale.
    assert t["gradient"].min() < 0 or t["gradient"].max() > 0


def test_relu_term_is_the_map_that_gets_upsampled():
    """The last step the reader sees must be the map the page scores.

    If these diverge, the stepper tells a story that ends somewhere other than
    the figure beside it.
    """
    torch.manual_seed(0)
    net = TinyNet().eval()
    x = torch.randn(1, 1, 8, 8, 8)
    out = gv._compute_gradcam_decomposed(
        net, x, [net.stage], ["stage0"], (8, 8, 8), 1, lambda o: o, "test")
    # Tap and target shape are both 8^3 here, so upsampling is the identity and
    # the relu term must equal raw exactly.
    np.testing.assert_allclose(
        out["terms"]["stage0"]["relu"], out["raw"]["stage0"], atol=1e-6)
