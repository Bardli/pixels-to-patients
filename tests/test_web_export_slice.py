from __future__ import annotations

import torch

from gradcam_repro.web_export import slice_to_payload


def test_slice_to_payload_normalizes() -> None:
    s = torch.tensor([[0.0, 5.0], [10.0, 2.5]])
    payload = slice_to_payload(s)
    assert payload["shape"] == [2, 2]
    assert len(payload["values"]) == 4
    assert min(payload["values"]) == 0
    assert max(payload["values"]) == 255
    assert payload["vmin"] == 0.0
    assert payload["vmax"] == 10.0


def test_slice_to_payload_already_normalized() -> None:
    s = torch.zeros(3, 3)
    payload = slice_to_payload(s, already_normalized=True)
    assert payload["values"] == [0] * 9
    assert payload["vmax"] == 0.0
