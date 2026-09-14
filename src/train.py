"""Train Model 4 (FlowTransformer) on the preprocessed CSE-CIC-IDS2018 splits.

Usage:
    python src/train.py --config configs/transformer.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import FlowDataset, load_meta  # noqa: E402
from models.transformer import FlowTransformer  # noqa: E402
from utils.metrics import compute_metrics, format_report  # noqa: E402


def run_eval(model, loader, device, n_classes) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, all_preds, all_true = 0.0, [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(y.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    return total_loss / len(y_true), y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processed_dir = cfg["data"]["processed_dir"]
    meta = load_meta(processed_dir)
    class_names = meta["classes"]
    n_features = meta["n_features"]

    train_ds = FlowDataset(f"{processed_dir}/train.npz")
    val_ds = FlowDataset(f"{processed_dir}/val.npz")
    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FlowTransformer(n_features=n_features, n_classes=len(class_names), **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1, epochs_without_improvement = -1.0, 0
    checkpoint_path = Path(cfg["train"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
        train_loss = running_loss / len(train_ds)

        val_loss, y_true, y_pred = run_eval(model, val_loader, device, len(class_names))
        val_metrics = compute_metrics(y_true, y_pred, class_names)
        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {"model_state": model.state_dict(), "config": cfg, "meta": meta},
                checkpoint_path,
            )
            print(f"  -> new best (macro_f1={best_val_f1:.4f}), saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["train"]["patience"]:
                print(f"Early stopping (no improvement for {cfg['train']['patience']} epochs)")
                break

    print("\nFinal validation report (best checkpoint):")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, y_true, y_pred = run_eval(model, val_loader, device, len(class_names))
    print(format_report(compute_metrics(y_true, y_pred, class_names), class_names))


if __name__ == "__main__":
    main()
