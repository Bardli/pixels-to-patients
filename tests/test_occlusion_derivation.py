"""Occlusion's intermediate is recoverable, so recover it rather than shipping a
page that stops at the result -- and recover it on the right scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "gradcam3d_viz", REPO / "scripts" / "gradcam3d_viz.py")
gv = importlib.util.module_from_spec(_spec)
sys.modules["gradcam3d_viz"] = gv
_spec.loader.exec_module(gv)


def _exporter():
    """Import jsc_web_export without its argparse main running."""
    spec = importlib.util.spec_from_file_location(
        "jsc_web_export", REPO / "scripts" / "jsc_web_export.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jsc_web_export"] = mod
    argv = sys.argv
    sys.argv = ["jsc_web_export", "--cases", "none"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


ex = _exporter()


def test_occluded_probability_is_intact_minus_drop():
    occluded = np.array([[0.97, 0.47], [0.77, 0.07]], dtype=np.float32)
    out = ex.occlusion_terms(occluded, prob_intact=0.97)
    assert out["intact"] == pytest.approx(0.97)
    np.testing.assert_allclose(out["drop"], 0.97 - occluded, atol=1e-6)
    np.testing.assert_allclose(out["occluded"], occluded, atol=1e-6)
    assert (out["drop"] >= -1e-6).all(), "occluding cannot raise the score"


def test_rejects_a_field_that_is_not_in_probability_units():
    """A field outside [0,1] means activate=True was lost -- fail loudly rather
    than render a negative probability."""
    with pytest.raises(ValueError, match="not in probability units"):
        ex.occlusion_terms(np.array([[-3.2]], dtype=np.float32), prob_intact=0.97)


def test_rejects_a_non_probability_intact_value():
    with pytest.raises(ValueError, match="not a probability"):
        ex.occlusion_terms(np.array([[0.5]], dtype=np.float32), prob_intact=3.5)


def test_intact_probability_uses_monais_softmax_scale_not_sigmoid():
    """The scale trap: MONAI softmaxes our two-column [-z, +z], so its intact
    probability is sigmoid(2z). meta.json's prob_malignant is sigmoid(z). On the
    lead case those are 0.999116 and 0.971106 -- subtracting the map from the
    wrong one silently corrupts every drop value.
    """
    z = 3.5148
    two_col = torch.tensor([[-z, z]])
    got = gv.occlusion_intact_probability(two_col, 1)
    assert got == pytest.approx(float(torch.sigmoid(torch.tensor(2 * z))), abs=1e-6)
    assert got != pytest.approx(float(torch.sigmoid(torch.tensor(z))), abs=1e-3)
    assert got == pytest.approx(0.999116, abs=1e-5)


def test_intact_probability_matches_the_target_class():
    z = 2.0
    two_col = torch.tensor([[-z, z]])
    p1 = gv.occlusion_intact_probability(two_col, 1)
    p0 = gv.occlusion_intact_probability(two_col, 0)
    assert p1 + p0 == pytest.approx(1.0, abs=1e-6)
    assert p1 > p0
