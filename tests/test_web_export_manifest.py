from __future__ import annotations

from pathlib import Path

from gradcam_repro.web_export import build_manifest


def test_build_manifest_hashes(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"checkpoint-bytes")
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"cache-bytes")
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"lock-bytes")
    manifest = build_manifest(ckpt, cache, lock, ["a", "b"], ["gradcam"], "2026-07-09T00:00:00Z")
    assert manifest["schema"] == "gradcam-repro.web-bundle.v1"
    assert manifest["generated_at"] == "2026-07-09T00:00:00Z"
    assert len(manifest["checkpoint"]["sha256"]) == 64
    assert manifest["num_examples"] == 2
    assert manifest["methods"] == ["gradcam"]
    assert "dirty" in manifest["git"]
