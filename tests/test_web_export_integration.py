from __future__ import annotations

import json
from pathlib import Path

from gradcam_repro.web_export import web_export


def test_web_export_end_to_end(real_ct_model, fake_cache: Path, tmp_path: Path, monkeypatch) -> None:
    # Patch model loader so we don't need a real trained checkpoint.
    import gradcam_repro.web_export as we

    monkeypatch.setattr(we, "load_real_ct_model", lambda checkpoint, device: real_ct_model)
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"stub")
    out = tmp_path / "bundle"
    result = web_export(
        cache=fake_cache, checkpoint=ckpt, out_dir=out, split="test",
        num_examples=2, methods=["gradcam", "layercam"], device_name="cpu",
        seed=13, generated_at="2026-07-09T00:00:00Z", uv_lock=tmp_path / "uv.lock",
    )
    assert (out / "manifest.json").exists()
    assert (out / "model_graph.json").exists()
    assert (out / "benchmark.json").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["num_examples"] == 2
    bench = json.loads((out / "benchmark.json").read_text())
    assert len(bench["examples"]) == 2
    assert result["num_examples"] == 2
    for ex in manifest["examples"]:
        assert (out / "examples" / ex / "ct_slice.png").exists()
