"""Evaluate a trained checkpoint on the held-out test split.

Usage:
    python src/evaluate.py --checkpoint checkpoints/transformer_best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import FlowDataset  # noqa: E402
from models.transformer import FlowTransformer  # noqa: E402
from utils.metrics import compute_metrics, format_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/transformer_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg, meta = ckpt["config"], ckpt["meta"]
    class_names = meta["classes"]

    test_ds = FlowDataset(f"{cfg['data']['processed_dir']}/test.npz")
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])

    model = FlowTransformer(n_features=meta["n_features"], n_classes=len(class_names), **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    metrics = compute_metrics(y_true, y_pred, class_names)
    print("Test set results:")
    print(format_report(metrics, class_names))

    if metrics["rare_classes"]:
        print(
            f"\n{len(metrics['rare_classes'])} rare class(es) present "
            f"(< {1000} test samples) -- macro-F1 is sensitive to these, check them individually above."
        )


if __name__ == "__main__":
    main()
