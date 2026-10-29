# Relabel experiment and 3-seed augmentation study (measured only)

All labels from `data/clean/label_audit_gold_review_labeled.csv` are model-authored (Claude), NOT native-speaker verified. Clean test n=2770, threshold 0.5, internal 1=abusive. Scripts: `src/relabel_exp.py`, `src/run_seeds.py`.

## 1. Relabel experiment (TF-IDF only)

`data/clean_relabel/` = clean train with the firm (borderline==0, 179 rows) labels applied where the text matches a train row; val/test copied untouched.
- 133 train rows matched firm sheet rows (text match), 98 labels changed. 231 of 300 sheet rows occur in train.
- TF-IDF (deployed recipe, C=0.3, `negation_aug.fit_tfidf`) retrained -> `results/relabel_tfidf/relabel.joblib` (deployed model untouched).
- Clean-test macro-F1 (raw decision > 0): baseline 0.8959, relabel 0.8981, delta +0.0022, paired bootstrap 95% CI [-0.0022, +0.0065] (10000 resamples, seed 0). Not significant. (Baseline is 0.8960 under the Platt 0.5 rule used elsewhere; 0.8959 here is decision>0.)
- Held-out overlap: firm sheet rows in test = 0, in val = 12. On those 12 val rows the val label disagrees with the human label in 11/12 (they were selected as suspects). Both baseline and relabel TF-IDF agree with the human label on 11/12 (same count). n=12: uninformative.
- Caveat: test labels carry the same noise, so scoring against them cannot reward fixing train noise; the flat/slightly positive result says only that 98 flips (0.4% of train) do not hurt. No native-speaker check of the flips was done.

## 2. Seeds (MuRIL lp8/lr5e-5, lora all-linear r16, 4 ep, bs32, maxlen128, fp16, early-stop 2)

Seeds 1, 2 trained here (`results/seeds/`); seed 42 = existing `results/clean_muril-base-cased_lp8_lr5e-5` and `results/aug_muril`. Preds dumped on the original clean val/test. Ensemble = mean of Platt-calibrated TF-IDF (clean: deployed; aug: `results/aug_tfidf/aug.joblib`, deterministic so no seed variance) and MuRIL.

| condition | seed | MuRIL F1 | ensemble F1 |
|---|---|---|---|
| clean | 1 | 0.9016 | 0.9059 |
| clean | 2 | 0.8941 | 0.9041 |
| clean | 42 | 0.8943 | 0.9052 |
| aug | 1 | 0.8990 | 0.9066 |
| aug | 2 | 0.8944 | 0.9019 |
| aug | 42 | 0.8922 | 0.9034 |

| condition | MuRIL mean +- sd | ensemble mean +- sd | TF-IDF |
|---|---|---|---|
| clean | 0.8967 +- 0.0043 | 0.9051 +- 0.0009 | 0.8960 |
| aug | 0.8952 +- 0.0035 | 0.9040 +- 0.0024 | 0.8953 |

Differences (aug - clean): MuRIL -0.0015, ensemble -0.0011, both far inside seed sd (sample sd, n=3).

### Functional pass rates (high-confidence, n=149; ensemble = deployed TF-IDF + seed's MuRIL; Moderator Console scorer)

| condition | overall ensemble mean +- sd | per seed (1/2/42) | negated_hate mean | threat mean | counter_speech | astrology | explicit_slur |
|---|---|---|---|---|---|---|---|
| clean | 96.6% +- 0.67 | 96.0 / 97.3 / 96.6 | 70% (0.7/0.7/0.7) | 93% | 90% | 97% | 100% |
| aug | 98.2% +- 0.39 | 98.7 / 98.0 / 98.0 | 93% (1.0/0.9/0.9) | 97% | 93% | 93% | 98% |

All other functionalities are 100% in both conditions. TF-IDF-only overall is 96.6% in both (deployed TF-IDF used for the functional ensemble in both conditions, so the difference is purely the MuRIL member). Per-group n is 5-12 and the tests/templates are model-authored with shared slur vocabulary; gains are on the same template family the augmentation was built from.

## Decision

Adopt the augmented MuRIL member (provisionally): negated_hate improves in 3/3 seeds (70% -> 90-100%), overall functional +1.6 points with non-overlapping seed ranges (aug 98.0-98.7 vs clean 96.0-97.3), and clean-test F1 change is within noise (-0.0011 ensemble, sd 0.001-0.004). Still do NOT adopt the augmented TF-IDF (no functional gain, 0.8953 vs 0.8960). Remaining blockers before release: native-speaker review of templates and test cases, native-written negation cases (the current test is in-distribution to the templates), and the relabel result above is unverified by native speakers.
