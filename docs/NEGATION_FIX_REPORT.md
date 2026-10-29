# Spaced-slur rejoin and negation augmentation: measured results

All numbers measured in this repo (`THS_DATA_DIR=data/clean`, clean test n=2770, threshold 0.5, single seed 42). Label convention: CSV 0=abusive; internal 1=abusive; `data/clean_aug/telugu_train.csv` is written in the raw CSV convention.

**Caveat: the augmentation templates (`src/negation_aug.py`) and the functional test file are model-authored, not written or reviewed by a native Telugu speaker. Grammar, naturalness and the abusive/non-abusive labels of the templates need native-speaker review before any release decision.**

## Task A: spaced short slurs (`src/akshara.py`)

Rule: after the existing separator/space rules, runs of 3 or more lone akshara units separated by single spaces are rejoined only if the joined string is in `src/abusive_lexicon.txt` (windows searched inside the run, longest first, so `నా కొ డ కా` becomes `నా కొడకా`). Lexicon = Telugu-only train tokens with abusive doc-frequency >= 15 and non-abusive doc-frequency <= 5% of that (176 tokens, mined from TRAIN only by `python src/negation_aug.py --write-lexicon`). It contains noise (names, e.g. a few spam-template names) that is harmless for rejoining but was denylisted for Task B.

- Clean-text alteration rate on non-abusive rows of `data/clean/telugu_test.csv`: **32/1436 = 2.23%, identical to the prior 2.23%** (the lexicon rule altered 0 additional clean rows). Abusive rows altered: 1.50%.
- `tests/test_akshara.py`: new asserts (`నువ్వు వె ధ వ`, `స న్నా సి`, `ఓరి నా కొ డ కా` rejoin; `నా ని ఈ`, `అ ది ని`, `ఆ రో జు`, `మీ రు ఎ ప్పు` untouched). Pass.
- Functional `obfuscation_spaced` (ensemble and TF-IDF): 80% -> 100% (n=5). Overall high-confidence pass 96.0% -> 96.6% with the old models (new akshara only).
- Ceiling: only slurs in the train lexicon are rejoined; a slur that is rare in train stays split. 2-unit runs are never rejoined.

## Task B: negation / reported-speech augmentation

`python src/negation_aug.py --build` writes `data/clean_aug/` (train = 22031 clean + 976 augmented = 23007 rows; val/test are byte-identical copies, asserted in the test).

- Slurs: first 60 lexicon tokens by train abusive count after a manual denylist (names/identity/ordinary words). Includes వెధవ, వెదవ, లంజ, కొడకా, ముండ; does not include సన్నాసి (rank beyond 60). Note the functional test uses some of the same slur words; only the sentence frames are disjoint.
- 8 frame pairs per slur, same slur and nearly the same structure: negated/denied/prohibited (non-abusive; e.g. "I never called X", "calling anyone X is wrong", "I have no habit of cursing X") vs affirmative (abusive; "I called X", "calling him X is right", "I have the habit"). 488 abusive / 488 non-abusive, so negation is not a "safe" cue. Plus 8 slur-free denials (non-abusive) and 8 slur-free threats/abuse (abusive) written by hand.
- Disjointness from `tests/functional/telugu_hatecheck.csv`: asserted in `tests/test_negation_aug.py` (no shared text; no shared word 4-gram after masking slurs; also no overlap with val/test texts).
- Retrained: TF-IDF (deployed recipe, C=0.3) on aug train; MuRIL via `train.py` with the lp8/lr5e-5 recipe (lora, all-linear, loraplus 8, 4 epochs, bs32, maxlen128, fp16, early-stop patience 2) -> `results/aug_muril`. Candidates are in `results/aug_tfidf/` and `results/aug_muril/`; deployed `models/tfidf` and the clean MuRIL run were NOT modified. `functional_eval.py`, `ensemble.py`, `tfidf_classifier.py` gained env overrides (`THS_TFIDF_MODEL`, `THS_MURIL_RUN`, `THS_REPORT_OUT`, `THS_SCORES_OUT`) so candidates are scored without overwriting committed reports. Reproduce with `src/negation_compare.py`.

### Clean test macro-F1 (paired bootstrap, 10000 resamples, seed 0, same resamples)

| model | baseline | augmented | delta | 95% CI of delta |
|---|---|---|---|---|
| TF-IDF | 0.8960 | 0.8953 | -0.0007 | [-0.0043, +0.0029] |
| TF-IDF, clean retrain with this script (control) | 0.8960 | 0.8963 | +0.0004 | [-0.0025, +0.0033] |
| MuRIL (clean lp8 run vs aug_muril) | 0.8943 | 0.8922 | -0.0021 | [-0.0094, +0.0051] |
| ensemble (TF-IDF + MuRIL) | 0.9052 | 0.9034 | -0.0018 | [-0.0075, +0.0037] |

All point estimates are slightly negative for MuRIL/ensemble but no CI excludes 0; one seed, so seed-to-seed noise is not separated from the effect. No significant regression.

### Functional pass rates (high-confidence, scored by the Moderator Console scorer; threshold 0.5)

Before = committed `docs/FUNCTIONAL_TEST_REPORT.md` (old akshara, old models). A = new akshara, old models. AB = A + augmented TF-IDF and MuRIL.

| functionality | n | TF-IDF before / A / AB | ensemble before / A / AB |
|---|---|---|---|
| negated_hate | 10 | 80% / 80% / 80% | 70% / 70% / 90% |
| obfuscation_spaced | 5 | 80% / 100% / 100% | 80% / 100% / 100% |
| threat | 10 | 90% / 90% / 90% | 90% / 90% / 100% |
| overall (n=149) | 149 | 96.0% / 96.6% / 96.6% | 96.0% / 96.6% / 98.7% |

All other high-confidence functionalities are unchanged between A and AB (identity_neutral_caste TF-IDF 88%, counter_speech ensemble 90%, astrology TF-IDF 90%; the rest 100%). Per-group n is 5-12, so one case is 8-20 points.

Remaining high-confidence failures in AB: TF-IDF `నేను అతన్ని వెధవ అనలేదు` (0.66), `ఆమెను లంజ అని ఎవరూ అనకూడదు` (0.88), threat `రాత్రికి నిన్ను ముక్కలు ముక్కలు చేస్తాను` (0.30); ensemble only `అతను వెధవ కాదు మంచి వ్యక్తి` (0.56). TF-IDF did not learn the contrast from the augmentation; the gain is entirely from MuRIL (the 2 negated_hate and 1 threat flips are ensemble-level).

### Low-confidence groups (reported separately; author's own doubt, not excluded from honesty)

| group | n | TF-IDF A / AB | ensemble A / AB |
|---|---|---|---|
| quoted_reported | 10 | 40% / 40% | 30% / 40% |
| friendly_teasing | 10 | 80% / 80% | 70% / 70% |
| lookalike_benign | 15 | 80% / 80% | 87% / 87% |
| explicit_slur (low) | 3 | 100% / 100% | 100% / 100% |
| profanity_at_person (low) | 3 | 100% / 100% | 100% / 100% |

Quoted/reported speech ("a report that someone said X") is still mostly misclassified as abusive; the augmentation did not address mention-vs-use of a slur in reports (only 1 case flipped, in the ensemble).

## Recommendation

- Adopt the akshara change (Task A): fixes spaced slurs, zero added clean alteration.
- Negation augmentation: provisionally adopt for the MuRIL member only (ensemble negated_hate 70% -> 90%, threat 90% -> 100%, no significant clean-test regression: ensemble -0.0018, CI [-0.0075, +0.0037]), but treat it as unproven: the evidence is 3 flipped test cases out of n=10 per group, a single seed, and model-authored templates and tests that share slur words. Before deploying: native-speaker review of `PAIRS`/`NEG_FREE`/`POS_FREE`, repeat with 2-3 seeds, and add native-written negation cases. Do not adopt the augmented TF-IDF (no functional gain, -0.0007 F1). Nothing was deployed or overwritten.
