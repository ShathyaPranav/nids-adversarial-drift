# Robustness of DL Architectures to Joint Adversarial-Evasion and Concept-Drift Attacks in NIDS

ICT-4442 Deep Learning Project. See [`ICT4442_Synopsis.docx`](ICT4442_Synopsis.docx) for the full proposal.

This branch of the repo covers **Model 4: Transformer classifier** (self-attention over CICFlowMeter
flow features), trained and evaluated on CSE-CIC-IDS2018 under the shared team protocol: same
train/val/test split, same clean metrics, same adversarial stress-test.

## Dataset

CSE-CIC-IDS2018, using Liu et al. (2022)'s corrected/relabeled release rather than the raw
AWS Open Data original (the raw version has ~7.5% label-corruption issues; see
`src/data/download.py` for both sources).

## Setup

```bash
pip install -r requirements.txt
python src/data/download.py --sample    # small single-day sample for dev/iteration
python src/data/preprocess.py
python src/train.py --config configs/transformer.yaml
python src/evaluate.py --checkpoint checkpoints/transformer_best.pt
```

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
