"""1D-CNN over the fixed feature order of each independent flow."""
import torch.nn as nn


class FlowCNN(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        channels1=32,
        channels2=64,
        kernel_size=3,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv1d(1, channels1, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(channels1, channels2, kernel_size, padding=padding),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels2 * n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        # (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)
