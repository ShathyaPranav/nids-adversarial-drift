# Error analysis: transformer

## Clean test set: classes with F1 < 0.9

- Infiltration - Dropbox Download: F1 0.111, 1/17 correct. Mistaken for: BENIGN (16). Other classes predicted as it: none.
- Web Attack - Brute Force: F1 0.886, 35/40 correct. Mistaken for: BENIGN (4), Web Attack - SQL (1). Other classes predicted as it: Web Attack - SQL (3), BENIGN (1).
- Web Attack - SQL: F1 0.500, 3/8 correct. Mistaken for: Web Attack - Brute Force (3), BENIGN (2). Other classes predicted as it: Web Attack - Brute Force (1).

## PGD evasion per attack class (up to 300 correctly classified test flows each)

| class | flows attacked | evasion rate | constraint violations |
|---|---|---|---|
| Web Attack - SQL | 3 | 1.000 | 0 |
| Web Attack - Brute Force | 35 | 1.000 | 0 |
| Infiltration - Dropbox Download | 1 | 1.000 | 0 |
| DoS Hulk | 300 | 0.973 | 0 |
| Botnet Ares | 300 | 0.970 | 0 |
| DDoS-HOIC | 300 | 0.327 | 0 |
| DoS Slowloris | 300 | 0.200 | 0 |
| DDoS-LOIC-HTTP | 300 | 0.140 | 0 |
| Infiltration - Communication Victim Attacker | 31 | 0.065 | 0 |
| Infiltration - NMAP Portscan | 300 | 0.060 | 0 |
| DoS GoldenEye | 300 | 0.023 | 0 |
| SSH-BruteForce | 300 | 0.010 | 0 |
| DDoS-LOIC-UDP | 300 | 0.000 | 0 |
| FTP-BruteForce | 9 | 0.000 | 0 |
| Web Attack - XSS | 17 | 0.000 | 0 |

## Mutable features moved most on evading flows (n = 852, standardised units)

| feature | mean abs change | mean signed change |
|---|---|---|
| Subflow Fwd Bytes | 0.495 | -0.314 |
| Down/Up Ratio | 0.485 | +0.134 |
| Flow Bytes/s | 0.483 | +0.243 |
| Fwd IAT Std | 0.473 | -0.042 |
| Packet Length Variance | 0.472 | -0.113 |
| Flow IAT Mean | 0.470 | -0.023 |
| Flow Packets/s | 0.466 | -0.240 |
| Flow IAT Std | 0.458 | -0.222 |
| Flow IAT Max | 0.458 | +0.132 |
| Fwd Act Data Pkts | 0.452 | +0.122 |
