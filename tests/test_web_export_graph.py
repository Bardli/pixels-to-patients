from __future__ import annotations

from gradcam_repro.web_export import build_model_graph


def test_build_model_graph_shapes(real_ct_model) -> None:
    graph = build_model_graph(real_ct_model, (16, 16, 16))
    assert graph["schema"] == "gradcam-repro.web-bundle.v1"
    assert graph["cam_tap"] == "stage2"
    ids = [n["id"] for n in graph["nodes"]]
    assert ids == ["input", "stage1", "stage2", "stage3", "pool", "classifier", "logits"]
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["input"]["out_shape"] == [1, 16, 16, 16]
    assert by_id["stage1"]["out_shape"] == [8, 8, 8, 8]
    assert by_id["stage2"]["out_shape"] == [16, 4, 4, 4]
    assert by_id["stage2"]["cam_tap"] is True
    assert by_id["classifier"]["out_shape"] == [2]
    assert by_id["stage1"]["param_count"] > 0
