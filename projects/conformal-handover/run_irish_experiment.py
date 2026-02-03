"""
Experiment on Irish 5G real-world dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src.handover.irish_data import load_driving_traces, preprocess_for_handover, get_handover_stats
from src.handover.conformal import calibrate_threshold, predict_sets, evaluate_cp


class IrishHandoverMLP(nn.Module):
    def __init__(self, input_dim: int, n_cells: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_cells),
        )

    def forward(self, x):
        return self.net(x)


def train_model(features, labels, train_idx, val_idx, n_cells, n_epochs=30):
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = IrishHandoverMLP(features.shape[1], n_cells)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} - Val Acc: {correct/total:.4f}")

    return model


def get_probs(model, features, idx):
    model.eval()
    X = torch.tensor(features[idx], dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
    return probs.numpy()


def main():
    print("="*60)
    print("Irish 5G Real-World Experiment")
    print("="*60)

    data_dir = Path("data/irish_5g/5G-production-dataset")

    print("\nLoading data...")
    df = load_driving_traces(data_dir, max_files=50)
    data = preprocess_for_handover(df)
    stats = get_handover_stats(data)

    print(f"Samples: {stats['n_samples']}")
    print(f"Cells: {stats['n_cells']}")
    print(f"Handovers: {stats['n_handovers']} ({stats['handover_rate']:.2%})")

    trace_ids = data["trace_id"]
    unique_traces = np.unique(trace_ids)
    n_traces = len(unique_traces)

    np.random.seed(42)
    np.random.shuffle(unique_traces)

    train_traces = set(unique_traces[:int(0.6 * n_traces)])
    cal_traces = set(unique_traces[int(0.6 * n_traces):int(0.8 * n_traces)])
    test_traces = set(unique_traces[int(0.8 * n_traces):])

    train_idx = np.array([i for i, t in enumerate(trace_ids) if t in train_traces])
    cal_idx = np.array([i for i, t in enumerate(trace_ids) if t in cal_traces])
    test_idx = np.array([i for i, t in enumerate(trace_ids) if t in test_traces])

    print(f"\nSplits: Train={len(train_idx)}, Cal={len(cal_idx)}, Test={len(test_idx)}")

    print("\nTraining model...")
    model = train_model(
        data["features"], data["next_cell"],
        train_idx, cal_idx, data["n_cells"], n_epochs=30
    )

    cal_probs = get_probs(model, data["features"], cal_idx)
    test_probs = get_probs(model, data["features"], test_idx)
    cal_labels = data["next_cell"][cal_idx]
    test_labels = data["next_cell"][test_idx]

    top1_acc = (test_probs.argmax(axis=1) == test_labels).mean()
    print(f"\nTop-1 Accuracy: {top1_acc:.4f}")

    for k in [1, 3, 5, 10]:
        topk = np.argsort(test_probs, axis=1)[:, -k:]
        cov = np.mean([test_labels[i] in topk[i] for i in range(len(test_labels))])
        print(f"Top-{k}: {cov:.4f}")

    print("\n--- Conformal Prediction ---")
    for alpha in [0.05, 0.10, 0.20]:
        threshold = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
        pred_sets = predict_sets(test_probs, threshold)
        results = evaluate_cp(pred_sets, test_labels)
        print(f"CP (α={alpha}): coverage={results['coverage']:.4f}, "
              f"size={results['avg_set_size']:.2f}")

    print("\n--- Key Insight ---")
    if top1_acc > 0.95:
        print("High accuracy (>95%) - CP likely adds minimal value")
        print("(Similar to O1_28 scenario in beam prediction)")
    elif top1_acc > 0.80:
        print("Moderate accuracy - CP provides useful uncertainty quantification")
    else:
        print("Lower accuracy - CP essential for reliability guarantees")


if __name__ == "__main__":
    main()
