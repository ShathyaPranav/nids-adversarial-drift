# nids-adversarial-drift

ICT-4442 deep learning mini project with full title "Robustness of DL architectures to joint adversarial-evasion and concept-drift attacks in NIDS"... we train four models on the same network flow data, then attack all of them the same way. the attack tries to sneak attack flows past the classifier and past a drift detector at the same time.

| Model | Who | Code |
|---|---|---|
| MLP | Sushant Shekhar Sinha | `src/models/mlp.py`, `configs/mlp.yaml` |
| 1D-CNN | Anne George | `src/models/cnn.py`, `configs/cnn.yaml` |
| LSTM | Siddhant Singh | `src/models/lstm.py`, `configs/lstm.yaml` |
| Transformer | Shathya Pranav S R K | `src/models/transformer.py`, `configs/transformer.yaml` |

data pipeline, the shared train/eval scripts and the attack framework were done by Shathya.

## dataset

CSE-CIC-IDS2018, but the corrected version from Liu et al. (2022), not the original (`CSECICIDS2018_improved.zip`, 10.4 GB, 10 daily csvs, ~63.2M flows, 94% benign). `src/data/preprocess.py` streams through the zip and builds the split everyone uses:

- drops the id columns (id, Flow ID, IPs, source port, timestamp), so 83 flow features are left
- merges "Attempted" labels into their parent class (FTP-BruteForce only shows up as "Attempted" anyway)
- removes NaN/Inf rows and exact duplicates
- caps each class at 200,000 flows and keeps smaller classes whole... ends up as ~1.11M flows, 16 classes
- signed log1p + standardisation, fitted on train only
- stratified 70/15/15 split, seed 42 (778,917 / 166,911 / 166,911 flows)

if you rerun it you get the same files byte for byte, and all four models train on those exact files. One thing to keep in mind is that, the capping means the benign to attack ratio isn't what real traffic looks like.

## running it

```bash
pip install -r requirements.txt
python src/data/download.py --corrected        # 10.4 GB, parallel and resumable
python src/data/preprocess.py --cap 200000

python src/train.py --config configs/<model>.yaml          # substitute <model> with mlp | cnn | lstm | transformer
python src/evaluate.py --checkpoint checkpoints/<model>_best.pt
python src/run_attack.py --config configs/attack_<model>.yaml

python src/compare_models.py                                # table + figures across models
python src/error_analysis.py --checkpoint checkpoints/transformer_best.pt --tag transformer
python src/run_ablation.py --checkpoint checkpoints/transformer_best.pt --tag transformer
```

## results

full tables are in [`results/comparison.md`](results/comparison.md) and the figures are in `results/figures/`. attack numbers below are at attacker fraction 0.10 (mean over 3 seeds x 10 windows of 1,000 flows).

| model | test accuracy | macro-F1 (16 classes) | evasion PGD | evasion decoupled | evasion joint |
|---|---|---|---|---|---|
| MLP | 0.9995 | 0.901 | 0.84 | 0.02 | 0.84 |
| 1D-CNN | 0.9994 | 0.889 | 0.24 | 0.00 | 0.25 |
| LSTM | 0.9989 | 0.813 | 0.86 | 0.22 | 0.85 |
| Transformer | 0.9997 | 0.899 | 0.45 | 0.17 | 0.46 |

what we found:

- on clean data you basically can't tell the models apart. Every one gets F1 ≥ 0.993 on all nine classes that have 1,000+ test flows. The all-class macro-F1 only drops because of tiny classes with 8 to 40 test flows (Dropbox Download, Web Attack SQL, FTP-BruteForce)
- under attack it's a different story... the 1D-CNN gets evaded ~24% of the time, the Transformer 45%, and the MLP and LSTM around 85%
- the joint attack evades about as well as PGD. The decoupled one loses most of its evasion because the drift stage pretty much undoes it
- joint doesn't get flagged by the drift detector any less than PGD does. Detection mostly comes down to how many flows in a window are perturbed (almost never flagged at 5%, always flagged at 15%)
- zero constraint violations in any attack and clean windows never got flagged
- ablation on the Transformer (`results/ablation/transformer/`): we tried lambda from 1 to 1000, quantile matching vs MMD as the proxy and different drift representations. None of it made joint harder to detect than PGD. What did work was a stronger decoupled attack that reverts drift steps if they'd undo an evasion... that one got flagged in only 20% of windows at a 10% attacker fraction (PGD 57%, joint 63%) and still evaded 30% of the time. turns out the detector mostly picks up the flows the attack *failed* on, since those end up sitting near the decision boundary with slightly lower confidence

## attack framework

everything's in `src/attack/` and it's the same for every model:

- `constraints.py`: 43 of the 83 features can be changed (stuff the attacker controls like sizes, counts, timing). destination port, protocol, TCP flags and anything from the responder side stay fixed. budget is 0.5 standardised units per feature, nothing goes negative, and Min <= Mean <= Max has to hold inside each feature family
- `drift.py`: KS test per feature plus one on classifier confidence, with Bonferroni correction. also has the differentiable proxies the attacker actually optimises (per-feature quantile matching, or MMD)
- `attacks.py`: `pgd` (classifier only), `decoupled` (classifier first, then drift), `joint` (classifier loss + lambda x drift proxy) and `decoupled_keep` (decoupled, but drift steps that would undo an evasion get reverted). all of them get the same 40-step budget and projection

## limitations

- the split is random over flows, so similar flows from one attack session can end up in both train and test. It also doesn't model drift over time
- the LSTM reads the 83 features of a single flow as a sequence... not flows over time
- each model was only trained once
- the constraints don't enforce exact relations between features (like total = mean x count) or integer counts, so the adversarial flows are plausible but we haven't proven they're realisable

## layout

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
