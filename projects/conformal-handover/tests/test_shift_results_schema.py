import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_DIR / "figures"


def load_json(name: str):
    path = FIGURES_DIR / name
    assert path.exists(), f"missing artifact: {path}"
    return json.loads(path.read_text())


def test_shift_results_schema():
    data = load_json("shift-results-v6.json")
    assert "synthetic_shift" in data
    assert "irish_shift" in data
    assert "cost_tracking" in data
    assert "metadata" in data

    synthetic = data["synthetic_shift"]
    assert "aggregated" in synthetic
    assert "paired_deltas" in synthetic
    assert "aci_gamma_ablation" in synthetic
    assert "trigger_quantile_ablation" in synthetic

    shifts = ["iid", "speed-shift", "measurement-noise-shift", "shadow-shift", "regime-switch"]
    methods = ["3db", "top1", "top3", "static-cp", "aci", "daci", "triggered-aci", "weighted-cp"]
    for shift in shifts:
        assert shift in synthetic["aggregated"]
        for method in methods:
            assert method in synthetic["aggregated"][shift]

    assert synthetic["aci_gamma_ablation"]
    assert synthetic["trigger_quantile_ablation"]
    assert "irish_seed" in data["metadata"]


def test_irish_results_schema():
    data = load_json("irish-shift-results-v6.json")
    assert "results" in data
    assert "bootstrap_ci" in data
    assert "speed_bins" in data
    assert "paired_deltas" in data
    assert "metadata" in data

    methods = ["top1", "top3", "static-cp", "aci", "daci", "triggered-aci", "weighted-cp"]
    for method in methods:
        assert method in data["results"]
        assert method in data["bootstrap_ci"]
    assert "seed" in data["metadata"]


def test_metric_sanity_ranges():
    shift = load_json("shift-results-v6.json")
    irish = shift["irish_shift"]
    shadow = shift["synthetic_shift"]["aggregated"]["shadow-shift"]
    regime = shift["synthetic_shift"]["aggregated"]["regime-switch"]

    for name, vals in shadow.items():
        if "coverage_mean" in vals:
            assert 0.0 <= vals["coverage_mean"] <= 1.0, name
    for name, vals in regime.items():
        if "coverage_mean" in vals:
            assert 0.0 <= vals["coverage_mean"] <= 1.0, name

    assert shadow["aci"]["coverage_mean"] >= shadow["static-cp"]["coverage_mean"]
    assert regime["aci"]["coverage_mean"] >= regime["static-cp"]["coverage_mean"]
    assert irish["results"]["aci"]["coverage"] >= irish["results"]["static-cp"]["coverage"]


def test_expected_figures_exist():
    required = [
        "shift-coverage-v6.pdf",
        "regime-switch-rolling-v6.pdf",
        "aci-gamma-ablation-v6.pdf",
        "trigger-quantile-ablation-v6.pdf",
        "hard-shift-pareto-v6.pdf",
        "irish-shift-rolling-v6.pdf",
        "irish-speed-bins-v6.pdf",
    ]
    for name in required:
        assert (FIGURES_DIR / name).exists(), name
