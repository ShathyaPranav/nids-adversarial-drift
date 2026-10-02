"""Adversarial stress-test: pgd vs decoupled vs joint attack against a trained classifier.

For each seed, windows of test traffic are drawn; a share (attacker_fractions) of the attack
flows the classifier currently catches are perturbed under the feature constraints, and three
numbers are measured:
  Classifier Evasion Rate    perturbed flows now classified as benign
  Drift Detection Rate       windows flagged by the KS drift detector
  Constraint Violation Rate  perturbed flows that break a feature constraint

Usage:
    python src/run_attack.py --config configs/attack.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack.attacks import MODES, run_attack  # noqa: E402
from attack.constraints import FeatureConstraints  # noqa: E402
from attack.drift import KSDriftDetector  # noqa: E402
from models import build_model  # noqa: E402


def load_classifier(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device)
    meta = ckpt["meta"]
    model = build_model(ckpt["config"]["model"], meta["n_features"], len(meta["classes"]))
    # cuDNN refuses input gradients for recurrent layers in eval mode, which the attacks need
    if any(isinstance(m, torch.nn.RNNBase) for m in model.modules()):
        torch.backends.cudnn.enabled = False
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), meta


@torch.no_grad()
def predict(model, z: torch.Tensor):
    probs = torch.softmax(model(z), dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/attack.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_classifier(cfg["checkpoint"], device)
    benign_idx = meta["classes"].index(cfg["benign_class"])

    pdir = Path(cfg["processed_dir"])
    scaler = np.load(pdir / "scaler.npz")
    constraints = FeatureConstraints(meta["feature_cols"], scaler["mean"], scaler["scale"], cfg["constraints"], device)
    print(f"{int(constraints.mutable.sum())} mutable / {len(meta['feature_cols'])} features, "
          f"{len(constraints.triples)} ordered (Min, Mean, Max) families")

    train_X = np.load(pdir / "train.npz")["X"]
    val_X = np.load(pdir / "val.npz")["X"]
    test = np.load(pdir / "test.npz")
    test_X, test_y = test["X"], test["y"]

    a, d, e = cfg["attack"], cfg["drift"], cfg["eval"]
    names = ("clean",) + MODES
    results = {}

    for fraction in a["attacker_fractions"]:
        per_seed = {m: [] for m in names}
        for seed in e["seeds"]:
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)

            det_ref = torch.from_numpy(val_X[rng.choice(len(val_X), d["n_reference"], replace=False)]).to(device)
            _, det_conf = predict(model, det_ref)
            detector = KSDriftDetector(det_ref.cpu().numpy(), det_conf.cpu().numpy(), d["alpha"])
            # the attacker estimates the reference distribution from its own sample, not the detector's
            atk_ref = torch.from_numpy(train_X[rng.choice(len(train_X), d["n_reference"], replace=False)]).to(device)

            rows = {m: [] for m in names}
            for _ in range(e["n_windows"]):
                idx = rng.choice(len(test_X), d["window_size"], replace=False)
                z0 = torch.from_numpy(test_X[idx]).to(device)
                y = torch.from_numpy(test_y[idx]).to(device)
                pred0, _ = predict(model, z0)

                caught = ((y != benign_idx) & (pred0 != benign_idx)).nonzero().squeeze(1)
                n_adv = max(1, int(round(fraction * len(caught))))
                chosen = caught[torch.from_numpy(rng.permutation(len(caught))[:n_adv]).to(device)]
                adv_mask = torch.zeros(len(z0), dtype=torch.bool, device=device)
                adv_mask[chosen] = True

                for mode in names:
                    if mode == "clean":
                        z = z0
                    else:
                        z = run_attack(
                            model, constraints, z0, adv_mask, benign_idx, atk_ref, mode,
                            a["eps"], a["step_size"], a["steps"], a["lam"], a["proxy"], a["representation"],
                        )
                    pred, conf = predict(model, z)
                    det = detector.detect(z.cpu().numpy(), conf.cpu().numpy())
                    rows[mode].append({
                        "evasion": float((pred[adv_mask] == benign_idx).float().mean()),
                        "drift": float(det["drift"]),
                        "rejected_features": det["n_rejected_features"],
                        "confidence_rejected": float(det["confidence_rejected"]),
                        "violation": float(constraints.violations(z[adv_mask], z0[adv_mask]).float().mean()),
                        "mean_abs_change": float((z - z0)[adv_mask][:, constraints.mutable].abs().mean()),
                        "n_perturbed": float(n_adv),
                    })
            for m in names:
                per_seed[m].append({k: float(np.mean([r[k] for r in rows[m]])) for k in rows[m][0]})

        summary = {
            m: {
                k: {"mean": float(np.mean([s[k] for s in per_seed[m]])), "std": float(np.std([s[k] for s in per_seed[m]]))}
                for k in per_seed[m][0]
            }
            for m in names
        }
        results[str(fraction)] = {"summary": summary, "per_seed": per_seed}

        print(f"\nattacker_fraction={fraction} (~{summary['clean']['n_perturbed']['mean']:.0f} perturbed flows per window)")
        print(f"{'mode':10s} {'evasion':>17s} {'drift detected':>17s} {'violations':>17s} {'KS rejects':>11s} {'conf KS':>8s}")
        for m in names:
            s = summary[m]
            print(
                f"{m:10s} "
                f"{s['evasion']['mean']:9.3f} +/- {s['evasion']['std']:.3f} "
                f"{s['drift']['mean']:9.3f} +/- {s['drift']['std']:.3f} "
                f"{s['violation']['mean']:9.3f} +/- {s['violation']['std']:.3f} "
                f"{s['rejected_features']['mean']:11.1f} {s['confidence_rejected']['mean']:8.2f}"
            )

    out = Path(e["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
