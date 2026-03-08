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
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--n-traj", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-files", type=int, default=40)
    parser.add_argument("--irish-horizon", type=int, default=10)
    parser.add_argument("--irish-split-seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--output-json", type=str, default="figures/shift-results-v7.json")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    run_shift_cmd = [
        python_exe,
        "run_shift_reliability_v7.py",
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
        "--max-files",
        str(args.max_files),
        "--irish-horizon",
        str(args.irish_horizon),
        "--irish-split-seeds",
        args.irish_split_seeds,
        "--output-json",
        args.output_json,
    ]

    run_cmd(run_shift_cmd, project_dir)
    run_cmd(["uv", "run", "--with", "pytest", "pytest", "-q", "tests/test_v7_controller.py", "tests/test_shift_results_v7_schema.py"], project_dir)
    run_cmd(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], project_dir)

    metadata = {
        "created_at_iso_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": get_git_commit(project_dir),
        "argv": sys.argv,
        "run_shift_cmd": run_shift_cmd,
        "output_json": args.output_json,
    }
    out_path = project_dir / "figures" / "run-paper-v7-metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
