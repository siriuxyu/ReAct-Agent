import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_runtime_microbenchmarks_report_control_plane_costs():
    from benchmark.runtime_micro_benchmark import run_runtime_microbenchmarks

    payload = run_runtime_microbenchmarks(iterations=50)
    names = {item["name"] for item in payload["benchmarks"]}

    assert payload["iterations"] == 50
    assert payload["manifest"]["benchmark"] == "runtime_micro"
    assert {"runtime_workspace_build", "tool_policy_evaluate", "model_route_explain"} <= names
    for item in payload["benchmarks"]:
        assert item["mean_us"] > 0
        assert item["p95_us"] >= item["mean_us"] or item["p95_us"] > 0
