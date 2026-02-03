"""
Irish 5G Dataset preprocessing for handover prediction.

Dataset: https://github.com/uccmisl/5Gdataset
Contains: RSRP, RSRQ, CellID, Speed, Location from real 5G network.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_driving_traces(data_dir: Path, max_files: Optional[int] = None) -> pd.DataFrame:
    """Load all driving traces from the Irish 5G dataset."""
    traces = []
    trace_id = 0

    driving_dirs = [
        data_dir / "Download" / "Driving",
        data_dir / "Netflix" / "Driving" / "Season3-StrangerThings",
        data_dir / "Netflix" / "Driving" / "animated-RickandMorty",
        data_dir / "Amazon_Prime" / "Driving" / "Season3-TheExpanse",
        data_dir / "Amazon_Prime" / "Driving" / "animated-AdventureTime",
    ]

    files_loaded = 0
    for driving_dir in driving_dirs:
        if not driving_dir.exists():
            continue
        for csv_file in sorted(driving_dir.glob("*.csv")):
            if max_files and files_loaded >= max_files:
                break
            try:
                df = pd.read_csv(csv_file)
                df["trace_id"] = trace_id
                df["source_file"] = csv_file.name
                traces.append(df)
                trace_id += 1
                files_loaded += 1
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        if max_files and files_loaded >= max_files:
            break

    if not traces:
        raise ValueError(f"No traces found in {data_dir}")

    return pd.concat(traces, ignore_index=True)


def preprocess_for_handover(df: pd.DataFrame) -> dict:
    """
    Preprocess Irish 5G data for handover prediction.

    Returns dict with:
        - features: RSRP, RSRQ, Speed, etc.
        - current_cell: current serving cell
        - next_cell: cell at next timestamp (target for prediction)
        - handover_occurred: binary indicator
    """
    df = df.copy()

    df["RSRP"] = pd.to_numeric(df["RSRP"], errors="coerce")
    df["RSRQ"] = pd.to_numeric(df["RSRQ"], errors="coerce")
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["NRxRSRP"] = pd.to_numeric(df["NRxRSRP"], errors="coerce")
    df["NRxRSRQ"] = pd.to_numeric(df["NRxRSRQ"], errors="coerce")

    df = df.dropna(subset=["RSRP", "CellID", "Speed"])

    cell_encoder = {cell: i for i, cell in enumerate(df["CellID"].unique())}
    df["cell_idx"] = df["CellID"].map(cell_encoder)

    all_features = []
    all_current = []
    all_next = []
    all_trace_ids = []

    for trace_id in df["trace_id"].unique():
        trace = df[df["trace_id"] == trace_id].sort_values("Timestamp").reset_index(drop=True)

        if len(trace) < 10:
            continue

        for i in range(len(trace) - 1):
            row = trace.iloc[i]
            next_row = trace.iloc[i + 1]

            features = {
                "rsrp": row["RSRP"],
                "rsrq": row["RSRQ"] if pd.notna(row["RSRQ"]) else -10,
                "speed": row["Speed"],
                "nr_rsrp": row["NRxRSRP"] if pd.notna(row["NRxRSRP"]) else -100,
            }

            all_features.append(features)
            all_current.append(row["cell_idx"])
            all_next.append(next_row["cell_idx"])
            all_trace_ids.append(trace_id)

    features_df = pd.DataFrame(all_features)

    features_df = features_df.fillna(features_df.median())

    for col in features_df.columns:
        mean = features_df[col].mean()
        std = features_df[col].std()
        if std > 0:
            features_df[col] = (features_df[col] - mean) / std

    return {
        "features": features_df.values,
        "current_cell": np.array(all_current),
        "next_cell": np.array(all_next),
        "trace_id": np.array(all_trace_ids),
        "n_cells": len(cell_encoder),
        "cell_encoder": cell_encoder,
    }


def get_handover_stats(data: dict) -> dict:
    """Compute handover statistics."""
    current = data["current_cell"]
    next_cell = data["next_cell"]

    handovers = current != next_cell
    n_handovers = handovers.sum()
    n_samples = len(current)

    return {
        "n_samples": n_samples,
        "n_cells": data["n_cells"],
        "n_handovers": n_handovers,
        "handover_rate": n_handovers / n_samples,
        "n_traces": len(np.unique(data["trace_id"])),
    }


if __name__ == "__main__":
    data_dir = Path("data/irish_5g/5G-production-dataset")

    print("Loading Irish 5G driving traces...")
    df = load_driving_traces(data_dir, max_files=50)
    print(f"Loaded {len(df)} samples from {df['trace_id'].nunique()} traces")

    print("\nPreprocessing for handover prediction...")
    data = preprocess_for_handover(df)

    stats = get_handover_stats(data)
    print(f"\nDataset statistics:")
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Unique cells: {stats['n_cells']}")
    print(f"  Traces: {stats['n_traces']}")
    print(f"  Handovers: {stats['n_handovers']}")
    print(f"  Handover rate: {stats['handover_rate']:.2%}")

    print(f"\nFeature shape: {data['features'].shape}")
