# Robustness of DL Architectures to Joint Adversarial-Evasion and Concept-Drift Attacks in NIDS

ICT-4442 Deep Learning mini project. We train four architectures on the same network-flow data
and then attack each one with the same constrained adversarial stress test, which tries to slip
attack flows past the classifier and past a statistical drift detector at the same time.

| Model | Owner | Code |
|---|---|---|
| MLP | Sushant Shekhar Sinha | `src/models/mlp.py`, `configs/mlp.yaml` |
| 1D-CNN | Anne George | `src/models/cnn.py`, `configs/cnn.yaml` |
| LSTM | Siddhant Singh | `src/models/lstm.py`, `configs/lstm.yaml` |
| Transformer | Shathya Pranav S R K | `src/models/transformer.py`, `configs/transformer.yaml` |

Data pipeline, shared training/evaluation scripts and the attack framework: Shathya Pranav S R K.

## Dataset

CSE-CIC-IDS2018, using the corrected release by Liu et al. (2022) (`CSECICIDS2018_improved.zip`,
10.4 GB, 10 daily CSVs, about 63.2M flows, 94% benign). `src/data/preprocess.py` streams the zip
and builds the shared split:

- identifier columns dropped (id, Flow ID, IPs, source port, timestamp), leaving 83 flow features
- "Attempted" labels merged into their parent class (FTP-BruteForce only exists as "Attempted")
- NaN/Inf rows and exact duplicates removed
- each class capped at 200,000 flows, smaller classes kept whole: about 1.11M flows, 16 classes
- signed log1p and standardisation fitted on the training split only
- stratified 70/15/15 split with seed 42 (778,917 / 166,911 / 166,911 flows)

Re-running the script reproduces the files byte for byte, and all four models use the same files.
Capping changes the benign to attack ratio compared with real traffic.

## Reproducing

```bash
pip install -r requirements.txt
python src/data/download.py --corrected        # 10.4 GB, parallel and resumable
python src/data/preprocess.py --cap 200000

python src/train.py --config configs/<model>.yaml          # mlp | cnn | lstm | transformer
python src/evaluate.py --checkpoint checkpoints/<model>_best.pt
python src/run_attack.py --config configs/attack_<model>.yaml

python src/compare_models.py                                # table + figures across models
python src/error_analysis.py --checkpoint checkpoints/transformer_best.pt --tag transformer
python src/run_ablation.py --checkpoint checkpoints/transformer_best.pt --tag transformer
```

## Results

Full tables are in [`results/comparison.md`](results/comparison.md), figures in `results/figures/`.
Attack numbers are at an attacker fraction of 0.10 (mean over 3 seeds x 10 windows of 1,000 flows).

| Model | Test accuracy | Macro-F1 (16 classes) | Evasion PGD | Evasion decoupled | Evasion joint |
|---|---|---|---|---|---|
| MLP | 0.9995 | 0.901 | 0.84 | 0.02 | 0.84 |
| 1D-CNN | 0.9994 | 0.889 | 0.24 | 0.00 | 0.25 |
| LSTM | 0.9989 | 0.813 | 0.86 | 0.22 | 0.85 |
| Transformer | 0.9997 | 0.899 | 0.45 | 0.17 | 0.46 |

- Clean accuracy hardly separates the models. Every model has F1 of at least 0.993 on all nine
  classes with 1,000 or more test flows, and the all-class macro-F1 is pulled down by classes with
  8 to 40 test flows (Dropbox Download, Web Attack SQL, FTP-BruteForce).
- Robustness separates them a lot: the 1D-CNN is evaded about 24% of the time, the Transformer
  45%, the MLP and LSTM about 85%.
- The joint attack keeps PGD's evasion rate, while the decoupled attack loses most of it because
  its drift stage undoes the evasion.
- The joint attack does not lower the drift detection rate compared with PGD. Detection mostly
  depends on how many flows in a window are perturbed (almost never flagged at 5%, always at 15%).
- No attack produced a constraint violation, and clean windows were never flagged.
- Ablation on the Transformer (`results/ablation/transformer/`): changing lambda (1 to 1000), the
  proxy (quantile matching or MMD) or the drift representation never made the joint attack harder
  to detect than PGD. A stronger decoupled attack that reverts drift steps which would undo an
  evasion was flagged in 20% of windows at a 10% attacker fraction (PGD 57%, joint 63%) while still
  evading 30% of the time. The detector mostly reacts to flows the attack failed to evade, which are
  left near the decision boundary with slightly lowered classifier confidence.

## Attack framework

`src/attack/` holds the stress test used for every model:

- `constraints.py`: 43 of 83 features are mutable (attacker-side sizes, counts, timing). Destination
  port, protocol, TCP flags and responder-side features are fixed. Budget of 0.5 standardised units
  per feature, no negative values, and Min <= Mean <= Max kept within each feature family.
- `drift.py`: Bonferroni-corrected KS test per feature plus one on classifier confidence, and the
  differentiable proxies the attacker optimises (per-feature quantile matching, or MMD).
- `attacks.py`: `pgd` (classifier only), `decoupled` (classifier, then drift), `joint`
  (classifier loss + lambda x drift proxy) and `decoupled_keep` (decoupled, but drift steps that
  undo an evasion are reverted). All use the same 40-step budget and projection.

## Limitations

- The split is random over flows, so similar flows from one attack session can land in both
  train and test, and it does not model drift over time.
- The LSTM reads the 83 features of one flow as a sequence, not flows in time.
- Each model is one training run.
- The constraints do not enforce exact relations between features (for example total = mean x
  count) or integer counts, so adversarial flows are plausible but not proven realisable.

## Layout

```
configs/            model and attack configs
src/data/           download, preprocessing, Dataset
src/models/         mlp, cnn, lstm, transformer + registry
src/attack/         constraints, drift detector and proxies, attacks
src/train.py        shared training loop (early stopping on val macro-F1)
src/evaluate.py     test-set evaluation
src/run_attack.py   adversarial stress test
src/run_ablation.py lambda / proxy / representation ablation
src/error_analysis.py, src/compare_models.py
results/            evaluation outputs, attack results, comparison, figures
```
