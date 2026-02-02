import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from beampred.config import N_NARROW_BEAMS, N_WIDE_BEAMS
from beampred.codebook import get_narrow_codebook, get_wide_codebook


def hierarchical_search(channels):
    narrow_cb = get_narrow_codebook()
    wide_cb = get_wide_codebook()

    wide_gains = np.abs(channels @ wide_cb.conj().T) ** 2
    best_wide = np.argmax(wide_gains, axis=1)

    ratio = N_NARROW_BEAMS // N_WIDE_BEAMS
    narrow_gains = np.abs(channels @ narrow_cb.conj().T) ** 2

    starts = best_wide * ratio
    candidate_indices = starts[:, None] + np.arange(ratio)[None, :]
    candidate_indices = np.clip(candidate_indices, 0, N_NARROW_BEAMS - 1)

    candidate_gains = np.take_along_axis(narrow_gains, candidate_indices, axis=1)
    best_narrow = candidate_indices[np.arange(len(channels)), np.argmax(candidate_gains, axis=1)]

    overhead = N_WIDE_BEAMS + ratio
    return best_narrow, overhead


def train_logistic_regression(train_features, train_labels):
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        n_jobs=-1,
    )
    clf.fit(train_features, train_labels)
    return clf


def predict_logistic(clf, features, k=1):
    if k == 1:
        return clf.predict(features)
    proba = clf.predict_proba(features)
    return np.argsort(proba, axis=1)[:, -k:][:, ::-1]


class BeamCNN(nn.Module):
    def __init__(self, n_input=N_WIDE_BEAMS, n_output=N_NARROW_BEAMS):
        super().__init__()
        self.n_input = n_input
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_output)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.interpolate(x, size=32, mode="linear", align_corners=False)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
