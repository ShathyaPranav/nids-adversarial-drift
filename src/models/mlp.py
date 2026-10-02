
import torch.nn as nn


class IntrusionMLP(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        hidden1=256,
        hidden2=128,
        hidden3=64,
        dropout=0.3,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),

            nn.Linear(hidden3, n_classes),
        )

    def forward(self, x):
        return self.network(x)
