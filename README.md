# Robustness of DL Architectures to Joint Adversarial-Evasion and Concept-Drift Attacks in NIDS

ICT-4442 Deep Learning Project. See [`ICT4442_Synopsis.docx`](ICT4442_Synopsis.docx) for the full proposal.

This branch of the repo covers **Model 4: Transformer classifier** (self-attention over CICFlowMeter
flow features), trained and evaluated on CSE-CIC-IDS2018 under the shared team protocol: same
train/val/test split, same clean metrics, same adversarial stress-test.

## Dataset

CSE-CIC-IDS2018, using Liu et al. (2022)'s **corrected release** (`CSECICIDS2018_improved.zip`,
10.4 GB, 10 daily CSVs, ~63.2M flows, 94% benign, 91 columns). The raw AWS release
(`--sample`) is only used for quick pipeline checks; it has a different schema.

`src/data/preprocess.py` streams the zip and builds a capped sample:
- max 200k rows per class (uniform random), all rows kept for smaller classes
- identifier columns dropped (id, Flow ID, IPs, Src Port, Timestamp) -> 83 features
- `- Attempted` labels merged into the parent class (`--attempted merge`); FTP-BruteForce
  exists only as "Attempted" in this release, so `drop` would remove that class
- NaN/Inf and exact-duplicate flows removed, signed log1p + standardization
- result: ~1.11M rows, 16 classes, stratified 70/15/15 split

Capping changes the benign/attack ratio versus real traffic.

## Setup

```bash
pip install -r requirements.txt
python src/data/download.py --corrected   # ~10.4 GB, parallel + resumable
python src/data/preprocess.py --cap 200000
python src/train.py --config configs/transformer.yaml
python src/evaluate.py --checkpoint checkpoints/transformer_best.pt
```

## Results so far (Model 4, baseline run 1, clean data only)

| split | accuracy | macro-F1 |
|-------|----------|----------|
| val   | 0.9998   | 0.928    |
| test  | 0.9997   | 0.899    |

All classes with >=1,000 test samples have F1 >= 0.998. Macro-F1 is pulled down by the
smallest classes: Infiltration - Dropbox Download (17 test samples, F1 0.11), Web Attack -
SQL (8 samples, F1 0.50), Web Attack - Brute Force (40 samples, F1 0.89). With single-digit
to tens of test samples per class these numbers are noisy, so report them per class.

Caveat: the split is random over flows, so near-identical flows from the same attack session
can land in both train and test. That inflates scores relative to a temporal or per-day split,
which matters for the concept-drift part of the project.

## Layout

```
src/
  data/
    download.py     # fetch raw or corrected CSE-CIC-IDS2018
    preprocess.py    # clean, encode, stratified split (keeps all rare-class rows)
    dataset.py       # PyTorch Dataset/DataLoader
  models/
    transformer.py   # Model 4: self-attention flow classifier
  utils/
    metrics.py        # macro-F1 + per-class rare-class reporting
  train.py
  evaluate.py
configs/
  transformer.yaml
```
