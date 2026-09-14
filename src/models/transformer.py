"""Model 4: attention-based (Transformer) classifier over CICFlowMeter flow features.

The 78 flow features are tabular (one row = one flow), not a token sequence, so this
follows the FT-Transformer approach (Gorishniy et al., 2021): each scalar feature is
projected into its own d_model embedding ("feature tokenizer"), a learned [CLS] token
is prepended, and a standard Transformer encoder attends across features before a
classification head reads out the [CLS] token.
"""
import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Projects each of n_features scalars into its own d_model-dim embedding."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight + self.bias


class FlowTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)  # (batch, n_features, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, 1 + n_features, d_model)
        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0])
        return self.head(cls_out)
