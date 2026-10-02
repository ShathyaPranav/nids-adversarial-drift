
"""Model registry. To add an architecture: create src/models/<name>.py with a class whose
constructor is (n_features, n_classes, **hyperparameters) and whose forward maps a float
tensor of shape (batch, n_features) to logits of shape (batch, n_classes); then add it below.
"""
from .transformer import FlowTransformer
from .lstm import FlowLSTM

MODELS = {
    "transformer": FlowTransformer,
    "lstm": FlowLSTM,
}


def build_model(model_cfg: dict, n_features: int, n_classes: int):
    """`model_cfg` is the `model:` section of a config; `name` picks the architecture."""
    params = dict(model_cfg)
    name = params.pop("name", "transformer")
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', registered: {sorted(MODELS)}")
    return MODELS[name](n_features=n_features, n_classes=n_classes, **params)