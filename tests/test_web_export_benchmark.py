from __future__ import annotations

from gradcam_repro.web_export import build_benchmark


def test_build_benchmark_aggregates() -> None:
    metas = [
        {"example_id": "a", "metrics": {"gradcam": {"mass_in_gt": 0.2, "inside_outside_ratio": 2.0, "pointing_acc": 1.0}}},
        {"example_id": "b", "metrics": {"gradcam": {"mass_in_gt": 0.4, "inside_outside_ratio": 4.0, "pointing_acc": 0.0}}},
    ]
    bench = build_benchmark(metas, ["gradcam"])
    assert bench["examples"] == ["a", "b"]
    assert bench["per_example"]["gradcam"]["a"]["mass_in_gt"] == 0.2
    assert abs(bench["aggregate"]["gradcam"]["mass_in_gt"] - 0.3) < 1e-9
    assert abs(bench["aggregate"]["gradcam"]["pointing_acc"] - 0.5) < 1e-9
