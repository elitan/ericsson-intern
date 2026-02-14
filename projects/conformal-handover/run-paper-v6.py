import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def get_git_commit(project_dir: Path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_dir, text=True).strip()
    except Exception:
        return "unknown"


def run_cmd(cmd: list[str], cwd: Path):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1011")
    parser.add_argument("--n-traj", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-json", type=str, default="figures/shift-results-v6.json")
    parser.add_argument("--budget-cap-usd", type=float, default=12.0)
    parser.add_argument("--cost-soft-stop-usd", type=float, default=10.5)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--raps-k-reg", type=int, default=1)
    parser.add_argument("--raps-lam", type=float, default=0.01)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    run_shift_cmd = [
        python_exe,
        "run-shift-reliability-v6.py",
        "--mode",
        "all",
        "--seeds",
        args.seeds,
        "--n-traj",
        str(args.n_traj),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--aci-gamma-grid",
        "0.002,0.005,0.01,0.02,0.05",
        "--trigger-quantile-grid",
        "0.5,0.6,0.7,0.8,0.9",
        "--daci-gamma-low-grid",
        "0.002,0.005,0.01",
        "--daci-gamma-high-grid",
        "0.01,0.02,0.05",
        "--daci-ema-beta-grid",
        "0.9,0.95,0.98",
        "--raps-k-reg",
        str(args.raps_k_reg),
        "--raps-lam",
        str(args.raps_lam),
        "--irish-split-protocol",
        "speed-split",
        "--irish-extra-split-protocol",
        "trace-holdout",
        "--run-ensemble-baseline",
        "--ensemble-members",
        str(args.ensemble_members),
        "--budget-cap-usd",
        str(args.budget_cap_usd),
        "--cost-soft-stop-usd",
        str(args.cost_soft_stop_usd),
        "--output-json",
        args.output_json,
    ]

    run_cmd(run_shift_cmd, project_dir)
    run_cmd(["uv", "run", "--with", "pytest", "pytest", "-q", "tests/test_shift_results_schema.py"], project_dir)
    run_cmd(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], project_dir)

    metadata = {
        "created_at_iso_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": get_git_commit(project_dir),
        "argv": sys.argv,
        "run_shift_cmd": run_shift_cmd,
        "output_json": args.output_json,
    }
    out_path = project_dir / "figures" / "run-paper-v6-metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
