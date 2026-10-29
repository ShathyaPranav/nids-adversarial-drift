# Robustness of DL Architectures to Joint Adversarial-Evasion and Concept-Drift Attacks in NIDS

ICT-4442 Deep Learning Project. See `ICT4442_Synopsis.docx` for the full proposal.

This branch of the repo covers **Model: MLP classifier** (fully connected feed-forward neural network over CICFlowMeter flow features), trained and evaluated on CSE-CIC-IDS2018 under the shared team protocol: same train/val/test split, same clean metrics, same adversarial stress-test.

## Dataset

CSE-CIC-IDS2018, using Liu et al. (2022)'s **corrected release** (`CSECICIDS2018_improved.zip`,
10.4 GB, 10 daily CSVs, ~63.2M flows, 94% benign, 91 columns). The raw AWS release
(`--sample`) is only used for quick pipeline checks; it has a different schema.

`src/data/preprocess.py` streams the zip and builds a capped sample:

* max 200k rows per class (uniform random), all rows kept for smaller classes
* identifier columns dropped (id, Flow ID, IPs, Src Port, Timestamp) → 83 features
* `- Attempted` labels merged into the parent class (`--attempted merge`)
* NaN/Inf and exact-duplicate flows removed
* signed log1p + standardization
* result: ~1.11M rows, 16 classes, stratified 70/15/15 split

Capping changes the benign/attack ratio versus real traffic.

## Setup

```bash
pip install -r requirements.txt
python src/data/download.py --corrected
python src/data/preprocess.py --cap 200000

python src/train.py --config configs/mlp.yaml
python src/evaluate.py --checkpoint checkpoints/mlp_best.pt
python src/run_attack.py --config configs/attack_mlp.yaml
```

## Model Architecture

Input: 83 flow features

```
83
↓
Linear(83 → 256)
BatchNorm
ReLU
Dropout(0.3)

↓
Linear(256 → 128)
BatchNorm
ReLU
Dropout(0.3)

↓
Linear(128 → 64)
ReLU

↓
Linear(64 → 16)
```

Output: 16-class intrusion classification logits.

Training uses the shared repository pipeline with standard CrossEntropyLoss, AdamW optimizer, early stopping, and no class weighting or oversampling.

## Results (MLP, shared pipeline)

### Validation Set

| split | accuracy | macro-F1 |
| ----- | -------- | -------- |
| val   | 0.9996   | 0.9471   |

### Test Set

| split | accuracy | macro-F1 |
| ----- | -------- | -------- |
| test  | 0.9995   | 0.9006   |

All classes with >=1,000 test samples achieve near-perfect performance. Macro-F1 is mainly affected by extremely small classes such as:

* Infiltration - Dropbox Download (17 samples, F1 = 0.105)
* Web Attack - SQL (8 samples, F1 = 0.500)
* Web Attack - Brute Force (40 samples, F1 = 0.854)

With only a handful of samples, these estimates are highly variable and should be interpreted individually rather than relying solely on Macro-F1.

Caveat: the split is random over flows, so near-identical flows from the same attack session can appear in both train and test sets, potentially inflating performance compared with a temporal or per-day split.

## Adversarial Stress-Test

`python src/run_attack.py --config configs/attack_mlp.yaml`

The shared attack pipeline perturbs a proportion of attack flows within each 1,000-flow test window and evaluates:

* `pgd` (classifier objective only)
* `decoupled` (classifier then drift objective)
* `joint` (classifier + drift objective)

Constraints:

* 43 of 83 features are mutable
* destination port, protocol, TCP flags and responder-side features remain fixed
* ε = 0.5 standardized units
* no negative values allowed
* Min ≤ Mean ≤ Max ordering preserved

Mean over 3 seeds × 10 windows:

| perturbed flows / window | attack                  | evasion               | drift detected        | KS rejections     |
| ------------------------ | ----------------------- | --------------------- | --------------------- | ----------------- |
| ~42 (5%)                 | pgd / decoupled / joint | 0.841 / 0.052 / 0.840 | 0.067 / 0.000 / 0.067 | 0.2 / 0.0 / 0.2   |
| ~84 (10%)                | pgd / decoupled / joint | 0.841 / 0.015 / 0.843 | 1.000 / 0.867 / 1.000 | 3.6 / 1.6 / 1.8   |
| ~125 (15%)               | pgd / decoupled / joint | 0.846 / 0.008 / 0.845 | 1.000 / 1.000 / 1.000 | 14.3 / 4.9 / 11.7 |
| ~167 (20%)               | pgd / decoupled / joint | 0.847 / 0.006 / 0.849 | 1.000 / 1.000 / 1.000 | 22.0 / 6.2 / 17.9 |

Constraint violation rate is 0 for all attacks; clean windows are never flagged.

### Key Observations

* PGD and Joint attacks achieve high evasion rates (~84%).
* Decoupled attacks substantially reduce evasion effectiveness.
* Drift detection becomes highly reliable once ≥10% of flows are perturbed.
* No attack generated invalid flows under the implemented constraints.
* Joint optimization maintains evasion performance while producing similar drift-detection behaviour to PGD.

## Layout

```text
src/
  data/
    download.py
    preprocess.py
    dataset.py

  models/
    mlp.py

  attack/
    constraints.py
    drift.py
    attacks.py

  run_attack.py

  utils/
    metrics.py

  train.py
  evaluate.py

configs/
  mlp.yaml
  attack_mlp.yaml
```
