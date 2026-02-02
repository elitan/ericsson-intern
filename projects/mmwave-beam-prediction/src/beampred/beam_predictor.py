import torch
import torch.nn as nn
import math
from beampred.config import N_WIDE_BEAMS, N_NARROW_BEAMS, HIDDEN_DIMS, DROPOUT


class BeamPredictor(nn.Module):
    def __init__(
        self,
        n_input=N_WIDE_BEAMS,
        n_output=N_NARROW_BEAMS,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ):
        super().__init__()
        layers = []
        prev_dim = n_input
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNetMLP(nn.Module):
    def __init__(self, n_input=N_WIDE_BEAMS, n_output=N_NARROW_BEAMS, hidden=256,
                 n_blocks=3, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, n_output)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BeamTransformer(nn.Module):
    def __init__(self, n_input=N_WIDE_BEAMS, n_output=N_NARROW_BEAMS,
                 d_model=64, nhead=2, n_layers=2, dim_ff=128):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_input, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_output)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
