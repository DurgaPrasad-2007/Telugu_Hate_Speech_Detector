# Conformal allow/review/block policy and ensemble

Test n=2770 (spam-free split, THS_DATA_DIR=data/clean); transformer probs from `results\clean_xlm-roberta-base_lp8_lr5e-5\preds.csv`.

## Ensemble vs single models (test, threshold 0.5)

| model | macro-F1 | AUC | delta (ens - model) | 95% paired-bootstrap CI | McNemar p (b=ens right/other wrong, c=reverse) |
|---|---|---|---|---|---|
| tfidf | 0.8960 | 0.9562 | +0.0045 | [-0.0038, +0.0131] | 0.356 (b=77, c=65) |
| transformer | 0.8944 | 0.9574 | +0.0061 | [-0.0004, +0.0127] | 0.0821 (b=51, c=34) |
| ensemble | 0.9005 | 0.9630 | - | - | - |

Bootstrap: 10000 resamples, seed 0, same resamples for all models.

## Conformal policy (alpha = 2% wrongful blocks of non-abusive; calibrated on val half B, n_cal=1365; TF-IDF Platt fitted on val half A, n=1364)

| scorer | beta | lo | hi | realized false-block | realized miss | review % | block % |
|---|---|---|---|---|---|---|---|
| tfidf | 5% | 0.181 | 0.829 | 2.23% | 4.72% | 24.6% | 35.8% |
| tfidf | 10% | 0.344 | 0.829 | 2.23% | 9.30% | 15.2% | 35.8% |
| tfidf | 20% | 0.653 | 0.829 | 2.23% | 18.97% | 5.3% | 35.8% |
| transformer | 5% | 0.126 | 0.944 | 1.95% | 5.47% | 23.0% | 36.0% |
| transformer | 10% | 0.268 | 0.944 | 1.95% | 8.10% | 17.0% | 36.0% |
| transformer | 20% | 0.788 | 0.944 | 1.95% | 18.59% | 5.4% | 36.0% |
| ensemble | 5% | 0.185 | 0.811 | 1.95% | 4.57% | 21.8% | 37.1% |
| ensemble | 10% | 0.381 | 0.811 | 1.95% | 8.92% | 13.0% | 37.1% |
| ensemble | 20% | 0.725 | 0.811 | 1.95% | 19.34% | 3.2% | 37.1% |

Guarantee is marginal over calibration draws (class-conditional split conformal); one calibration set gives realized rates that scatter around alpha/beta (see rows where realized exceeds target).
