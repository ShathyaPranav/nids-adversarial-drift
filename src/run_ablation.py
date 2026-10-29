"""Ablation of the joint attack: lambda weight, drift proxy and drift representation.

Every variant reuses run_attack.py with configs/attack.yaml as the base, changing only the
listed settings, at the attacker fractions where drift detection is neither 0 nor 1.

Usage:
    python src/run_ablation.py --checkpoint checkpoints/transformer_best.pt --tag transformer
"""
import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

FRACTIONS = [0.10, 0.15]

# (name, overrides of the `attack` section, modes to run)
VARIANTS = [
    ("baselines", {}, ["pgd", "decoupled", "decoupled_keep", "joint"]),
    ("lam_1", {"lam": 1.0}, ["joint"]),
    ("lam_100", {"lam": 100.0}, ["joint"]),
    ("lam_1000", {"lam": 1000.0}, ["joint"]),
    ("mmd_lam_10", {"proxy": "mmd", "lam": 10.0}, ["joint"]),
    ("mmd_lam_100", {"proxy": "mmd", "lam": 100.0}, ["joint"]),
    ("repr_features_lam_100", {"representation": "features", "lam": 100.0}, ["joint"]),
    ("repr_confidence_lam_100", {"representation": "confidence", "lam": 100.0}, ["joint"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/attack.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    with open(args.base) as f:
        base = yaml.safe_load(f)

    out_dir = Path("results/ablation") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, overrides, modes in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["checkpoint"] = args.checkpoint
        cfg["attack"].update(overrides)
        cfg["attack"]["attacker_fractions"] = FRACTIONS
        cfg["eval"]["modes"] = modes
        cfg["eval"]["out"] = str(out_dir / f"{name}.json")

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp)
        print(f"== {name}", flush=True)
        subprocess.run([sys.executable, "src/run_attack.py", "--config", tmp.name], check=True)
        Path(tmp.name).unlink()

        with open(cfg["eval"]["out"]) as f:
            results = json.load(f)["results"]
        for frac, r in results.items():
            for mode in modes:
                s = r["summary"][mode]
                rows.append({
                    "variant": name, "fraction": float(frac), "mode": mode,
                    "lam": cfg["attack"]["lam"], "proxy": cfg["attack"]["proxy"],
                    "representation": cfg["attack"]["representation"],
                    **{k: [s[k]["mean"], s[k]["std"]] for k in
                       ("evasion", "drift", "violation", "rejected_features", "confidence_rejected")},
                })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'variant':26s} {'frac':>5s} {'mode':15s} {'evasion':>15s} {'drift detected':>15s} {'KS rej':>7s} {'conf KS':>7s}")
    for r in rows:
        print(f"{r['variant']:26s} {r['fraction']:5.2f} {r['mode']:15s} "
              f"{r['evasion'][0]:6.3f} +/- {r['evasion'][1]:.3f} {r['drift'][0]:6.3f} +/- {r['drift'][1]:.3f} "
              f"{r['rejected_features'][0]:7.1f} {r['confidence_rejected'][0]:7.2f}")
    print(f"\nSaved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
