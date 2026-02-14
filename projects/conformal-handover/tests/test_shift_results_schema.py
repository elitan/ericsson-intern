import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_DIR / "figures"
REPORT_TEX = PROJECT_DIR / "report.tex"


def load_json(name: str):
    path = FIGURES_DIR / name
    assert path.exists(), f"missing artifact: {path}"
    return json.loads(path.read_text())


def fmt_dot(value: float):
    return f".{int(round(value * 1000)):03d}"


def fmt_cov_ci(value: float, low: float, high: float):
    mean = fmt_dot(value)
    lo = fmt_dot(low)
    hi = fmt_dot(high)
    return f"${mean}\\,[{lo},{hi}]$"


def test_shift_results_schema():
    data = load_json("shift-results-v6.json")
    assert "synthetic_shift" in data
    assert "irish_shift" in data
    assert "cost_tracking" in data
    assert "metadata" in data

    metadata = data["metadata"]
    for key in [
        "git_commit",
        "created_at_iso_utc",
        "daci_gamma_low_grid",
        "daci_gamma_high_grid",
        "daci_ema_beta_grid",
        "raps_k_reg",
        "raps_lam",
        "run_ensemble_baseline",
        "ensemble_members",
        "irish_seed",
    ]:
        assert key in metadata

    synthetic = data["synthetic_shift"]
    for key in [
        "aggregated",
        "paired_deltas",
        "aci_gamma_ablation",
        "trigger_quantile_ablation",
        "daci_robustness",
        "conditional_coverage",
        "ensemble_baseline",
    ]:
        assert key in synthetic

    shifts = ["iid", "speed-shift", "measurement-noise-shift", "shadow-shift", "regime-switch"]
    methods = ["3db", "top1", "top3", "static-cp", "raps-cp", "aci", "daci", "triggered-aci", "weighted-cp", "ensemble-cp"]
    for shift in shifts:
        assert shift in synthetic["aggregated"]
        for method in methods:
            assert method in synthetic["aggregated"][shift]

    assert synthetic["aci_gamma_ablation"]
    assert synthetic["trigger_quantile_ablation"]
    assert synthetic["daci_robustness"]["shadow-shift"]
    assert synthetic["daci_robustness"]["regime-switch"]


def test_irish_results_schema():
    data = load_json("irish-shift-results-v6.json")
    for key in [
        "results",
        "bootstrap_ci",
        "speed_bins",
        "paired_deltas",
        "rolling",
        "daci_robustness",
        "conditional_coverage",
        "additional_splits",
        "ensemble_baseline",
        "metadata",
    ]:
        assert key in data

    methods = ["top1", "top3", "static-cp", "raps-cp", "aci", "daci", "triggered-aci", "weighted-cp", "ensemble-cp"]
    for method in methods:
        assert method in data["results"]
        assert method in data["bootstrap_ci"]

    for key in [
        "seed",
        "git_commit",
        "created_at_iso_utc",
        "daci_gamma_low_grid",
        "daci_gamma_high_grid",
        "daci_ema_beta_grid",
        "raps_k_reg",
        "raps_lam",
        "split_protocol",
        "extra_split_protocol",
        "run_ensemble_baseline",
        "ensemble_members",
    ]:
        assert key in data["metadata"]
    if data["metadata"]["extra_split_protocol"] != "none":
        assert data["metadata"]["extra_split_protocol"] in data["additional_splits"]


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
        "daci-robustness-v6.pdf",
        "conditional-coverage-speed-v6.pdf",
        "conditional-coverage-confidence-v6.pdf",
        "ensemble-vs-cp-v6.pdf",
        "irish-shift-rolling-v6.pdf",
        "irish-speed-bins-v6.pdf",
        "related-work-matrix-v6.csv",
    ]
    for name in required:
        assert (FIGURES_DIR / name).exists(), name


def test_run_paper_script_exists():
    assert (PROJECT_DIR / "run-paper-v6.py").exists()


def test_report_number_consistency():
    shift = load_json("shift-results-v6.json")
    text = REPORT_TEX.read_text()

    shadow = shift["synthetic_shift"]["aggregated"]["shadow-shift"]
    regime = shift["synthetic_shift"]["aggregated"]["regime-switch"]
    irish = shift["irish_shift"]["results"]
    irish_boot = shift["irish_shift"]["bootstrap_ci"]
    irish_extra = shift["irish_shift"]["additional_splits"]["trace-holdout"]

    shadow_static = fmt_cov_ci(
        shadow["static-cp"]["coverage_mean"],
        shadow["static-cp"]["coverage_ci95_low"],
        shadow["static-cp"]["coverage_ci95_high"],
    )
    shadow_raps = fmt_cov_ci(
        shadow["raps-cp"]["coverage_mean"],
        shadow["raps-cp"]["coverage_ci95_low"],
        shadow["raps-cp"]["coverage_ci95_high"],
    )
    regime_static = fmt_cov_ci(
        regime["static-cp"]["coverage_mean"],
        regime["static-cp"]["coverage_ci95_low"],
        regime["static-cp"]["coverage_ci95_high"],
    )
    regime_daci_mean = fmt_dot(regime["daci"]["coverage_mean"])
    regime_daci_low = fmt_dot(regime["daci"]["coverage_ci95_low"])
    regime_daci_high = fmt_dot(regime["daci"]["coverage_ci95_high"])

    assert f"& Static CP & {shadow_static}" in text
    assert f"& RAPS CP & {shadow_raps}" in text
    assert f"& Static CP & {regime_static}" in text
    assert f"& DACI & $\\mathbf{{{regime_daci_mean}}}\\,[{regime_daci_low},{regime_daci_high}]$" in text

    irish_static = fmt_cov_ci(
        irish["static-cp"]["coverage"],
        irish_boot["static-cp"]["coverage_ci95_low"],
        irish_boot["static-cp"]["coverage_ci95_high"],
    )
    irish_raps = fmt_cov_ci(
        irish["raps-cp"]["coverage"],
        irish_boot["raps-cp"]["coverage_ci95_low"],
        irish_boot["raps-cp"]["coverage_ci95_high"],
    )
    irish_daci_mean = fmt_dot(irish["daci"]["coverage"])
    irish_daci_low = fmt_dot(irish_boot["daci"]["coverage_ci95_low"])
    irish_daci_high = fmt_dot(irish_boot["daci"]["coverage_ci95_high"])

    assert f"Static CP & {irish_static}" in text
    assert f"RAPS CP & {irish_raps}" in text
    assert f"DACI & $\\mathbf{{{irish_daci_mean}}}\\,[{irish_daci_low},{irish_daci_high}]$" in text

    trace_static_mean = fmt_dot(irish_extra["results"]["static-cp"]["coverage"])
    trace_daci_mean = fmt_dot(irish_extra["results"]["daci"]["coverage"])
    assert f"Static CP & ${trace_static_mean}" in text
    assert f"DACI & $\\mathbf{{{trace_daci_mean}}}" in text
