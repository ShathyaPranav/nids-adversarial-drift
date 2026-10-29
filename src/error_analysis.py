"""Error analysis for one trained model.

Clean test set: confusion matrix and, for every class with F1 below 0.9, what it is mistaken for.
Under attack: PGD evasion rate per attack class, and which mutable features the attack moves most
on the flows that evade.

Usage:
    python src/error_analysis.py --checkpoint checkpoints/transformer_best.pt --tag transformer
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack.attacks import run_attack  # noqa: E402
from attack.constraints import FeatureConstraints  # noqa: E402
from run_attack import load_classifier  # noqa: E402

PER_CLASS_CAP = 300


@torch.no_grad()
def predict(model, X: torch.Tensor, batch: int = 4096) -> np.ndarray:
    return torch.cat([model(X[i:i + batch]).argmax(1) for i in range(0, len(X), batch)]).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--attack-config", default="configs/attack.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_classifier(args.checkpoint, device)
    classes, features = meta["classes"], meta["feature_cols"]
    with open(args.attack_config) as f:
        acfg = yaml.safe_load(f)
    pdir = Path(acfg["processed_dir"])
    benign = classes.index(acfg["benign_class"])
    out = Path("results/error_analysis") / args.tag
    out.mkdir(parents=True, exist_ok=True)

    test = np.load(pdir / "test.npz")
    X = torch.from_numpy(test["X"]).to(device)
    y = test["y"]
    pred = predict(model, X)

    # ---- clean confusion matrix
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion.csv")
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{args.tag}: test confusion matrix (row-normalised)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=200)
    plt.close(fig)

    f1 = f1_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
    lines = [f"# Error analysis: {args.tag}", "", "## Clean test set: classes with F1 < 0.9", ""]
    for i, c in enumerate(classes):
        if f1[i] >= 0.9:
            continue
        wrong = [(classes[j], cm[i, j]) for j in np.argsort(-cm[i]) if j != i and cm[i, j]]
        into = [(classes[j], cm[j, i]) for j in np.argsort(-cm[:, i]) if j != i and cm[j, i]]
        lines.append(f"- {c}: F1 {f1[i]:.3f}, {cm[i, i]}/{cm[i].sum()} correct. "
                     f"Mistaken for: {', '.join(f'{n} ({k})' for n, k in wrong) or 'none'}. "
                     f"Other classes predicted as it: {', '.join(f'{n} ({k})' for n, k in into) or 'none'}.")

    # ---- per-class PGD evasion
    a = acfg["attack"]
    scaler = np.load(pdir / "scaler.npz")
    cons = FeatureConstraints(features, scaler["mean"], scaler["scale"], acfg["constraints"], device)
    rng = np.random.default_rng(0)
    train_X = np.load(pdir / "train.npz")["X"]
    ref = torch.from_numpy(train_X[rng.choice(len(train_X), acfg["drift"]["n_reference"], replace=False)]).to(device)

    rows, deltas = [], []
    for k, c in enumerate(classes):
        if k == benign:
            continue
        idx = np.flatnonzero((y == k) & (pred == k))
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(PER_CLASS_CAP, len(idx)), replace=False)
        z0 = X[idx]
        mask = torch.ones(len(idx), dtype=torch.bool, device=device)
        z = run_attack(model, cons, z0, mask, benign, ref, "pgd", a["eps"], a["step_size"], a["steps"],
                       a["lam"], a["proxy"], a["representation"])
        evaded = predict(model, z) == benign
        rows.append({"class": c, "n_attacked": len(idx), "evasion_rate": float(evaded.mean()),
                     "violations": int(cons.violations(z, z0).sum())})
        if evaded.any():
            deltas.append((z - z0)[torch.from_numpy(evaded).to(device)].cpu().numpy())

    ev = pd.DataFrame(rows).sort_values("evasion_rate", ascending=False)
    ev.to_csv(out / "per_class_evasion.csv", index=False)
    lines += ["", f"## PGD evasion per attack class (up to {PER_CLASS_CAP} correctly classified test flows each)", "",
              "| class | flows attacked | evasion rate | constraint violations |", "|---|---|---|---|"]
    lines += [f"| {r['class']} | {r['n_attacked']} | {r['evasion_rate']:.3f} | {r['violations']} |" for r in ev.to_dict("records")]

    if deltas:
        d = np.concatenate(deltas)
        mutable = cons.mutable.cpu().numpy()
        shift = pd.DataFrame({
            "feature": np.array(features)[mutable],
            "mean_abs_change": np.abs(d[:, mutable]).mean(0),
            "mean_signed_change": d[:, mutable].mean(0),
        }).sort_values("mean_abs_change", ascending=False)
        shift.to_csv(out / "feature_changes.csv", index=False)
        lines += ["", f"## Mutable features moved most on evading flows (n = {len(d)}, standardised units)", "",
                  "| feature | mean abs change | mean signed change |", "|---|---|---|"]
        lines += [f"| {r['feature']} | {r['mean_abs_change']:.3f} | {r['mean_signed_change']:+.3f} |"
                  for r in shift.head(10).to_dict("records")]

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
