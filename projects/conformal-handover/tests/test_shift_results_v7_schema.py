import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_DIR / "figures"
REPORT_TEX = PROJECT_DIR / "report.tex"


def load_json(name: str):
    path = FIGURES_DIR / name
    assert path.exists(), f"missing artifact: {path}"
    return json.loads(path.read_text())


def test_v7_schema():
    data = load_json("shift-results-v7.json")
    for key in ["metadata", "synthetic", "irish", "elapsed_seconds"]:
        assert key in data

    metadata = data["metadata"]
    for key in [
        "git_commit",
        "created_at_iso_utc",
        "delay_grid",
        "budget_grid",
        "budget_target",
        "budget_gamma",
        "budget_target_grid",
        "budget_gamma_grid",
        "replay_config",
        "irish_horizon",
        "irish_split_seeds",
    ]:
        assert key in metadata

    synthetic = data["synthetic"]
    for key in ["aggregated", "delayed_feedback", "recalibration", "worst_slice", "paired_deltas"]:
        assert key in synthetic

    irish = data["irish"]
    assert "protocols" in irish
    for protocol in ["speed-split", "trace-holdout", "chronological-holdout"]:
        assert protocol in irish["protocols"]
        for key in [
            "split_seeds",
            "runs",
            "bootstrap_ci_runs",
            "split",
            "results",
            "paired_deltas",
            "budget_sweep",
            "delayed_feedback",
            "recalibration",
            "worst_slice",
            "worst_slice_summary",
            "frontier_methods",
        ]:
            assert key in irish["protocols"][protocol]
        for key in ["configs", "frontier_configs", "best_overall", "best_under_daci_load"]:
            assert key in irish["protocols"][protocol]["budget_sweep"]
        sample_method = irish["protocols"][protocol]["worst_slice_summary"]["confidence"]["daci"]
        for key in [
            "worst_decile_mode",
            "min_coverage",
            "min_coverage_ci95_low",
            "min_coverage_ci95_high",
            "mean_coverage",
            "mean_coverage_ci95_low",
            "mean_coverage_ci95_high",
        ]:
            assert key in sample_method


def test_v7_figures_exist():
    required = [
        "controller-shift-summary-v7.pdf",
        "controller-frontier-v7.pdf",
        "delayed-feedback-v7.pdf",
        "recalibration-budget-v7.pdf",
        "irish-worst-slice-v7.pdf",
    ]
    for name in required:
        assert (FIGURES_DIR / name).exists(), name


def test_report_mentions_v7_artifacts():
    text = REPORT_TEX.read_text()
    for pattern in [
        "controller-frontier-v7.pdf",
        "delayed-feedback-v7.pdf",
        "recalibration-budget-v7.pdf",
        "irish-worst-slice-v7.pdf",
    ]:
        assert pattern in text
