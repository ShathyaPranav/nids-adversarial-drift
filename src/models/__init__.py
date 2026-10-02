
"""Model registry."""

from .transformer import FlowTransformer
from .mlp import IntrusionMLP

MODELS = {
    "transformer": FlowTransformer,
    "mlp": IntrusionMLP,
}


def build_model(model_cfg: dict, n_features: int, n_classes: int):
    """Build a registered model from its config."""
    params = dict(model_cfg)
    name = params.pop("name", "transformer")

    if name not in MODELS:
        raise ValueError(
            f"unknown model '{name}', registered: {sorted(MODELS)}"
        )

    return MODELS[name](
        n_features=n_features,
        n_classes=n_classes,
        **params
    )
