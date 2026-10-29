"""Build the cross-model comparison table and figures from the saved results files.

Reads results/eval_<model>.txt (output of evaluate.py) and results/attack_<model>.json (output of
run_attack.py) for every model, and writes results/comparison.md plus figures in results/figures/.

Usage:
    python src/compare_models.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = [("mlp", "MLP"), ("cnn", "1D-CNN"), ("lstm", "LSTM"), ("transformer", "Transformer")]
MODES = [("pgd", "PGD"), ("decoupled", "Decoupled"), ("joint", "Joint")]
TABLE_FRACTION = "0.1"
LARGE_CLASS = 1000
RESULTS = Path("results")
FIG = RESULTS / "figures"
ROW = re.compile(r"^\s+(.+?)\s+P=([\d.]+) R=([\d.]+) F1=([\d.]+) n=(\d+)")


def read_eval(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    per_class = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            per_class[m.group(1)] = {"p": float(m.group(2)), "r": float(m.group(3)),
                                     "f1": float(m.group(4)), "n": int(m.group(5))}
    return {
        "accuracy": float(re.search(r"Accuracy: ([\d.]+)", text).group(1)),
        "macro_f1": float(re.search(r"Macro-F1: ([\d.]+)", text).group(1)),
        "per_class": per_class,
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    evals = {k: read_eval(RESULTS / f"eval_{k}.txt") for k, _ in MODELS}
    attacks = {k: json.loads((RESULTS / f"attack_{k}.json").read_text())["results"] for k, _ in MODELS}
    fractions = sorted(attacks["transformer"], key=float)

    lines = ["# Model comparison", "",
             f"Clean metrics on the test split; attack metrics at attacker fraction {TABLE_FRACTION} "
             "(mean +/- std over 3 seeds x 10 windows).", "",
             "| Model | Accuracy | Macro-F1 (16 classes) | Macro-F1 (classes with >=1,000 test flows) "
             "| Evasion PGD | Evasion decoupled | Evasion joint | Drift detected PGD | Drift detected joint "
             "| Constraint violations |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, name in MODELS:
        e, s = evals[k], attacks[k][TABLE_FRACTION]["summary"]
        big = [v["f1"] for v in e["per_class"].values() if v["n"] >= LARGE_CLASS]
        viol = max(s[m]["violation"]["mean"] for m, _ in MODES)
        cell = lambda m, key: f"{s[m][key]['mean']:.3f} +/- {s[m][key]['std']:.3f}"  # noqa: E731
        lines.append(f"| {name} | {e['accuracy']:.4f} | {e['macro_f1']:.3f} | {np.mean(big):.4f} "
                     f"| {cell('pgd', 'evasion')} | {cell('decoupled', 'evasion')} | {cell('joint', 'evasion')} "
                     f"| {cell('pgd', 'drift')} | {cell('joint', 'drift')} | {viol:.3f} |")

    classes = list(evals["transformer"]["per_class"])
    lines += ["", "## Per-class test F1", "",
              "| Class | Test flows | " + " | ".join(n for _, n in MODELS) + " |",
              "|---|---|" + "---|" * len(MODELS)]
    for c in classes:
        n = evals["transformer"]["per_class"][c]["n"]
        lines.append(f"| {c} | {n} | " + " | ".join(f"{evals[k]['per_class'][c]['f1']:.3f}" for k, _ in MODELS) + " |")

    lines += ["", "## Drift detection rate by attacker fraction", "",
              "| Model | Attack | " + " | ".join(f"{float(f):.2f}" for f in fractions) + " |",
              "|---|---|" + "---|" * len(fractions)]
    for k, name in MODELS:
        for m, mname in MODES:
            lines.append(f"| {name} | {mname} | " + " | ".join(
                f"{attacks[k][f]['summary'][m]['drift']['mean']:.2f}" for f in fractions) + " |")

    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # evasion by model and attack
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    width = 0.25
    x = np.arange(len(MODELS))
    for i, (m, mname) in enumerate(MODES):
        means = [attacks[k][TABLE_FRACTION]["summary"][m]["evasion"]["mean"] for k, _ in MODELS]
        stds = [attacks[k][TABLE_FRACTION]["summary"][m]["evasion"]["std"] for k, _ in MODELS]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=2, label=mname)
    ax.set_xticks(x, [n for _, n in MODELS])
    ax.set_ylabel("Classifier evasion rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title(f"Evasion rate by model and attack (attacker fraction {TABLE_FRACTION})")
    fig.tight_layout()
    fig.savefig(FIG / "evasion_by_model.png", dpi=200)
    plt.close(fig)

    # drift detection vs attacker fraction
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    xs = [float(f) for f in fractions]
    for ax, (m, mname) in zip(axes, MODES):
        for k, name in MODELS:
            ax.plot(xs, [attacks[k][f]["summary"][m]["drift"]["mean"] for f in fractions], marker="o", label=name)
        ax.set_title(mname)
        ax.set_xlabel("Share of caught attack flows perturbed")
        ax.set_xticks(xs)
    axes[0].set_ylabel("Drift detection rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "drift_vs_fraction.png", dpi=200)
    plt.close(fig)

    # per-class F1 heatmap
    f1 = np.array([[evals[k]["per_class"][c]["f1"] for k, _ in MODELS] for c in classes])
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(f1, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODELS)), [n for _, n in MODELS])
    ax.set_yticks(range(len(classes)),
                  [f"{c} (n={evals['transformer']['per_class'][c]['n']})" for c in classes], fontsize=7)
    for i in range(len(classes)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{f1[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, fraction=0.046)
    ax.set_title("Per-class test F1")
    fig.tight_layout()
    fig.savefig(FIG / "per_class_f1.png", dpi=200)
    plt.close(fig)

    print("\n".join(lines))
    print(f"\nSaved results/comparison.md and figures in {FIG}")


if __name__ == "__main__":
    main()
