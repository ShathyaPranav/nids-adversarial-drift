import torch
from torch import nn

class FlowLSTM(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.unsqueeze(-1)            # (batch, 83) -> (batch, 83, 1)
        _, (h, _) = self.lstm(x)
        return self.classifier(h[-1])
