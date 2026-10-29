# Model comparison

Clean metrics on the test split; attack metrics at attacker fraction 0.1 (mean +/- std over 3 seeds x 10 windows).

| Model | Accuracy | Macro-F1 (16 classes) | Macro-F1 (classes with >=1,000 test flows) | Evasion PGD | Evasion decoupled | Evasion joint | Drift detected PGD | Drift detected joint | Constraint violations |
|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.9995 | 0.901 | 0.9994 | 0.841 +/- 0.007 | 0.015 +/- 0.008 | 0.843 +/- 0.009 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 |
| 1D-CNN | 0.9994 | 0.889 | 0.9989 | 0.242 +/- 0.012 | 0.002 +/- 0.002 | 0.245 +/- 0.013 | 0.867 +/- 0.125 | 0.833 +/- 0.170 | 0.000 |
| LSTM | 0.9989 | 0.813 | 0.9983 | 0.859 +/- 0.017 | 0.217 +/- 0.053 | 0.851 +/- 0.018 | 0.367 +/- 0.309 | 0.467 +/- 0.249 | 0.000 |
| Transformer | 0.9997 | 0.899 | 0.9997 | 0.451 +/- 0.015 | 0.173 +/- 0.016 | 0.455 +/- 0.016 | 0.567 +/- 0.236 | 0.633 +/- 0.262 | 0.000 |

## Per-class test F1

| Class | Test flows | MLP | 1D-CNN | LSTM | Transformer |
|---|---|---|---|---|---|
| BENIGN | 27398 | 0.999 | 0.998 | 0.997 | 0.999 |
| Botnet Ares | 21469 | 1.000 | 1.000 | 1.000 | 1.000 |
| DDoS-HOIC | 30000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DDoS-LOIC-HTTP | 30000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DDoS-LOIC-UDP | 416 | 1.000 | 1.000 | 0.996 | 0.998 |
| DoS GoldenEye | 4029 | 1.000 | 0.999 | 0.999 | 1.000 |
| DoS Hulk | 30000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DoS Slowloris | 1615 | 0.999 | 0.996 | 0.996 | 0.999 |
| FTP-BruteForce | 9 | 1.000 | 0.900 | 0.000 | 0.947 |
| Infiltration - Communication Victim Attacker | 31 | 0.984 | 0.984 | 0.931 | 0.969 |
| Infiltration - Dropbox Download | 17 | 0.105 | 0.105 | 0.000 | 0.111 |
| Infiltration - NMAP Portscan | 7732 | 0.997 | 0.997 | 0.993 | 0.999 |
| SSH-BruteForce | 14129 | 1.000 | 1.000 | 1.000 | 1.000 |
| Web Attack - Brute Force | 40 | 0.854 | 0.878 | 0.864 | 0.886 |
| Web Attack - SQL | 8 | 0.500 | 0.400 | 0.267 | 0.500 |
| Web Attack - XSS | 18 | 0.971 | 0.971 | 0.971 | 0.971 |

## Drift detection rate by attacker fraction

| Model | Attack | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|
| MLP | PGD | 0.07 | 1.00 | 1.00 | 1.00 |
| MLP | Decoupled | 0.00 | 0.87 | 1.00 | 1.00 |
| MLP | Joint | 0.07 | 1.00 | 1.00 | 1.00 |
| 1D-CNN | PGD | 0.10 | 0.87 | 1.00 | 1.00 |
| 1D-CNN | Decoupled | 0.00 | 0.90 | 1.00 | 1.00 |
| 1D-CNN | Joint | 0.10 | 0.83 | 1.00 | 1.00 |
| LSTM | PGD | 0.03 | 0.37 | 1.00 | 1.00 |
| LSTM | Decoupled | 0.00 | 0.00 | 0.90 | 1.00 |
| LSTM | Joint | 0.03 | 0.47 | 1.00 | 1.00 |
| Transformer | PGD | 0.07 | 0.57 | 1.00 | 1.00 |
| Transformer | Decoupled | 0.00 | 0.57 | 1.00 | 1.00 |
| Transformer | Joint | 0.07 | 0.63 | 1.00 | 1.00 |
