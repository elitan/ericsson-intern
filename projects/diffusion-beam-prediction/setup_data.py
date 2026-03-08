#!/usr/bin/env python3
"""Check Downloads folder for DeepSense data and set up data directory."""

import os
import shutil
import zipfile
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads"
DATA_DIR = Path(__file__).parent / "data"
SCENARIOS = [31, 32, 33, 34]


def find_scenario_files():
    """Find scenario files in Downloads."""
    found = {}
    for scenario in SCENARIOS:
        patterns = [
            f"scenario{scenario}",
            f"scenario_{scenario}",
            f"Scenario{scenario}",
            f"Scenario_{scenario}",
        ]
        for pattern in patterns:
            for path in DOWNLOADS.glob(f"*{pattern}*"):
                if path.is_dir() or path.suffix in [".zip", ".tar", ".gz"]:
                    found[scenario] = path
                    break
            if scenario in found:
                break
    return found


def extract_if_needed(path: Path, dest: Path):
    """Extract archive if needed."""
    if path.suffix == ".zip":
        print(f"Extracting {path.name}...")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(dest.parent)
        extracted = dest.parent / path.stem
        if extracted.exists() and extracted != dest:
            shutil.move(str(extracted), str(dest))
    elif path.is_dir():
        print(f"Copying {path.name}...")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(path), str(dest))


def inspect_scenario(scenario_dir: Path):
    """Inspect a scenario directory structure."""
    print(f"\n  Structure of {scenario_dir.name}:")

    csv_files = list(scenario_dir.glob("*.csv"))
    print(f"    CSV files: {[f.name for f in csv_files]}")

    if csv_files:
        import pandas as pd
        df = pd.read_csv(csv_files[0], nrows=5)
        print(f"    Columns: {list(df.columns)[:10]}...")
        print(f"    Rows: {len(pd.read_csv(csv_files[0]))}")

    for subdir in sorted(scenario_dir.iterdir()):
        if subdir.is_dir():
            files = list(subdir.iterdir())[:3]
            print(f"    {subdir.name}/: {len(list(subdir.iterdir()))} files")
            if files:
                print(f"      Examples: {[f.name for f in files]}")


def main():
    print("=" * 60)
    print("DeepSense 6G Data Setup")
    print("=" * 60)

    print(f"\nChecking Downloads folder: {DOWNLOADS}")
    found = find_scenario_files()

    if not found:
        print("\nNo scenario files found yet in Downloads.")
        print("Looking for patterns: scenario31, scenario_31, etc.")
        print("\nFiles in Downloads:")
        for f in sorted(DOWNLOADS.iterdir())[:20]:
            print(f"  {f.name}")
        return

    print(f"\nFound {len(found)} scenarios:")
    for scenario, path in found.items():
        print(f"  Scenario {scenario}: {path}")

    DATA_DIR.mkdir(exist_ok=True)

    for scenario, src_path in found.items():
        dest = DATA_DIR / f"scenario{scenario}"
        if dest.exists():
            print(f"\nScenario {scenario} already exists in data/")
        else:
            extract_if_needed(src_path, dest)

        if dest.exists():
            inspect_scenario(dest)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
