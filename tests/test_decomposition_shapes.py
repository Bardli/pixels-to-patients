"""Every method that can expose its terms must expose them in the same shape,
so the front end has one contract instead of seven.
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
    def __init__(self):
        super().__init__()
        self.stage = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.stage(x).mean(dim=(2, 3, 4)))


TAP = (8, 8, 8)


def _net_and_x():
    torch.manual_seed(0)
    return TinyNet().eval(), torch.randn(1, 1, 8, 8, 8)


@pytest.mark.parametrize("fn_name,needs_cls", [
    ("_compute_notgradcam", False),
    ("_compute_layercam", True),
    ("_compute_gradcam_decomposed", True),
])
def test_terms_shape_contract(fn_name, needs_cls):
    net, x = _net_and_x()
    fn = getattr(gv, fn_name)
    if fn_name == "_compute_gradcam_decomposed":
        out = fn(net, x, [net.stage], ["stage0"], TAP, 1, lambda o: o, "test")
    elif needs_cls:
        out = fn(net, x, [net.stage], ["stage0"], TAP, 1, lambda o: o, "test",
                 return_terms=True)
    else:
        out = fn(net, x, [net.stage], ["stage0"], TAP, "test", return_terms=True)

    assert set(out) == {"raw", "terms"}
    t = out["terms"]["stage0"]
    assert "channel" in t and isinstance(t["channel"], int)
    for k, v in t.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            assert v.shape == TAP, f"{fn_name}.{k} shape {v.shape}"
            assert np.isfinite(v).all(), f"{fn_name}.{k} non-finite"


@pytest.mark.parametrize("fn_name", ["_compute_notgradcam", "_compute_layercam"])
def test_return_terms_false_keeps_the_old_return_value(fn_name):
    """The flag must be additive: existing callers get exactly what they got."""
    net, x = _net_and_x()
    fn = getattr(gv, fn_name)
    if fn_name == "_compute_notgradcam":
        plain = fn(net, x, [net.stage], ["stage0"], TAP, "test")
    else:
        plain = fn(net, x, [net.stage], ["stage0"], TAP, 1, lambda o: o, "test")
    assert isinstance(plain, dict) and set(plain) == {"stage0"}
    assert plain["stage0"].shape == TAP


def test_terms_do_not_change_the_exported_map():
    """Asking for terms must not perturb the map the site scores."""
    net, x = _net_and_x()
    plain = gv._compute_notgradcam(net, x, [net.stage], ["stage0"], TAP, "test")
    net2, x2 = _net_and_x()
    withterms = gv._compute_notgradcam(
        net2, x2, [net2.stage], ["stage0"], TAP, "test", return_terms=True)
    np.testing.assert_allclose(plain["stage0"], withterms["raw"]["stage0"], atol=0)


def test_notgradcam_has_no_gradient_term():
    """The missing gradient is the baseline page's argument; do not invent one."""
    net, x = _net_and_x()
    t = gv._compute_notgradcam(
        net, x, [net.stage], ["stage0"], TAP, "test", return_terms=True)["terms"]["stage0"]
    assert "gradient" not in t
    assert "channel_means" in t and t["channel_means"].shape == (4,)


def test_layercam_reports_whether_the_gradient_was_constant():
    """LayerCAM claims per-voxel weighting; under a pooled head there is none,
    and the page must be able to say so from data rather than assert a
    refinement that is not present."""
    net, x = _net_and_x()
    t = gv._compute_layercam(
        net, x, [net.stage], ["stage0"], TAP, 1, lambda o: o, "test",
        return_terms=True)["terms"]["stage0"]
    assert t["grad_is_constant"] is True
    assert t["alpha_all"].shape == (4,)
    assert t["hadamard"].shape == TAP


class Cfg:
    """Minimal stand-in for GradcamConfig, with the real attribute names."""
    ig_n_steps = 8
    ig_batch_size = 2
    ig_smooth_sigma = 0.0
    igc_n_steps = 8


@pytest.mark.parametrize("fn_name,extra", [
    ("_compute_integrated_gradients", ()),
    ("_compute_integrated_gradcam", ("stages",)),
])
def test_path_checkpoints_are_monotone_in_frac(fn_name, extra):
    """The saturation story only reads if the partial sums are in path order."""
    net, x = _net_and_x()
    fn = getattr(gv, fn_name)
    if extra:
        out = fn(net, x, [net.stage], ["stage0"], TAP, 1, lambda o: o, Cfg(),
                 "test", return_terms=True)
        terms = out["terms"]["stage0"]
    else:
        out = fn(net, x, TAP, 1, lambda o: o, Cfg(), "test", return_terms=True)
        terms = out["terms"]["input_resolution"]

    path = terms["path"]
    assert len(path) == 4, f"expected 4 checkpoints, got {len(path)}"
    fracs = [p["frac"] for p in path]
    assert fracs == sorted(fracs), f"path out of order: {fracs}"
    assert fracs[-1] == pytest.approx(1.0)
    for p in path:
        assert np.isfinite(p["map"]).all()
        assert p["map"].ndim == 3


def test_path_last_checkpoint_tracks_the_final_map():
    """The last panel a reader sees must be the finished integral, not a stage
    partway along it."""
    net, x = _net_and_x()
    out = gv._compute_integrated_gradcam(
        net, x, [net.stage], ["stage0"], TAP, 1, lambda o: o, Cfg(), "test",
        return_terms=True)
    last = out["terms"]["stage0"]["path"][-1]["map"]
    # raw is upsampled to target_shape; here tap == target so they must agree.
    np.testing.assert_allclose(last, out["raw"]["stage0"], rtol=1e-5, atol=1e-7)


def test_layercam_equals_the_positive_alpha_sum_under_a_pooled_head():
    """The LayerCAM page now states an identity, so pin it.

    The tap ends in a ReLU (A >= 0) and a globally pooled head makes the
    gradient one constant a_k per channel, so
        sum_k ReLU(a_k A^k) == sum_{a_k > 0} a_k A^k
    exactly. That is why LayerCAM drops negative-weight channels outright here
    instead of refining anything, and the `discarded` term is the part Grad-CAM
    would have cancelled against the positives instead.
    """
    net, x = _net_and_x()
    out = gv._compute_layercam(
        net, x, [net.stage], ["stage0"], TAP, 1, lambda o: o, "test",
        return_terms=True)
    t = out["terms"]["stage0"]
    assert t["grad_is_constant"] is True, "identity only holds for a pooled head"

    import torch as _t
    acts = {}
    h = net.stage.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o.detach()))
    with _t.no_grad():
        net(x)
    h.remove()
    A = acts["a"][0]
    assert float(A.min()) >= 0.0, "tap must be post-ReLU for the identity to hold"

    alpha = _t.as_tensor(t["alpha_all"])
    pos = (alpha.clamp(min=0)[:, None, None, None] * A).sum(0).numpy()
    np.testing.assert_allclose(t["relu"], pos, rtol=1e-5, atol=1e-7)

    neg = (alpha.clamp(max=0)[:, None, None, None] * A).sum(0).numpy()
    np.testing.assert_allclose(t["discarded"], neg, rtol=1e-5, atol=1e-7)
    assert t["discarded"].min() <= 0.0, "discarded evidence is non-positive"
