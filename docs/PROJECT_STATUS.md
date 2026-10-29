# Project status (Oct 2026)

## What was built
Telugu abusive-comment detector (pure Telugu script) with: spam-free data (`data/clean`, `src/despam.py`), akshara-aware input guard (`src/akshara.py`, rejoins spaced slurs via a train-mined lexicon), TF-IDF + LinearSVC and LoRA+ MuRIL / XLM-R models, TF-IDF+MuRIL ensemble, conformal allow/review/block policy, label audit, a 190-case Telugu functional test suite, negation/reported-speech augmentation (candidate only), and a Moderator Console tab in `app.py`.

## Measured results (clean test, n=2770, macro-F1, threshold 0.5, single seed)
| model | macro-F1 |
|---|---|
| TF-IDF | 0.8960 |
| MuRIL (lp8, lr5e-5) | 0.8943 |
| XLM-R (lp8, lr5e-5) | 0.8944 |
| Ensemble TF-IDF+MuRIL (console scorer) | 0.9052 |
| Ensemble TF-IDF+XLM-R (POLICY_REPORT) | 0.9005 |

Ensemble gains are not significant: vs TF-IDF +0.0045 CI [-0.0038, +0.0131]; vs XLM-R +0.0061 CI [-0.0004, +0.0127] (the TF-IDF+XLM-R ensemble; MuRIL ensemble not separately tested here).

Functional suite (149 high-confidence cases, model-authored): TF-IDF 96.0%, ensemble 96.0% (committed report). With new akshara guard: 96.6%. With akshara + augmented models (not deployed): ensemble 98.7%, TF-IDF 96.6%. Weak spots: negated_hate, quoted/reported speech (30-40% on low-confidence cases), identity-mention caste/gender FPs (n=6-8).

## Allow/review/block policy (alpha=2% wrongful blocks, ensemble, test)
| beta (target miss) | review % | block % | realized false-block | realized miss |
|---|---|---|---|---|
| 5% | 21.8% | 37.1% | 1.95% | 4.57% |
| 10% | 13.0% | 37.1% | 1.95% | 8.92% |
| 20% | 3.2% | 37.1% | 1.95% | 19.34% |

Guarantee is marginal; a single calibration set scatters around the targets.

## Known limits
- Label noise: 687 OOF suspects; the top-300 gold sheet was labeled by a Claude model, not a human. Label precision is unmeasured; cleaning gave no significant gain (all CIs include 0).
- Functional tests and augmentation templates are model-authored and need native-speaker review before use as a release gate.
- Single seed for all new runs (no `SEEDS_RELABEL_REPORT.md` exists); the older XLM-R LoRA study used 3 seeds.
- No Romanized/code-mixed test set; `src/translit.py` is only unit-tested.
- Moderator Console has not been checked in a browser (only tests).
- Mamba/BLT/MoE ideas in RESEARCH_REVIEW.md section 4 were not built; projections there are unmeasured.

## How to run (repo root, PowerShell; `.venv\Scripts\python`)
```
$env:THS_DATA_DIR="data/clean"
python src/despam.py --eval                                   # despam counts/baseline
python src/functional_eval.py                                 # functional report
python src/ensemble.py results/clean_xlm-roberta-base_lp8_lr5e-5/preds.csv
python src/conformal_policy.py results/clean_xlm-roberta-base_lp8_lr5e-5/preds.csv
python src/label_audit.py                                     # label audit
python src/negation_aug.py --build                            # builds data/clean_aug
python src/negation_compare.py                                # aug vs baseline
python src/tfidf_akshara.py                                   # akshara variant
python tests/test_akshara.py; python tests/test_negation_aug.py; python tests/test_moderator.py
streamlit run app.py                                          # Moderator Console tab
```
Some scripts have extra args or env overrides (`THS_TFIDF_MODEL`, `THS_MURIL_RUN`, `THS_REPORT_OUT`, `THS_SCORES_OUT`); check each script header. Commands were taken from script usage lines, not all re-run for this doc.

## Next steps (priority order)
1. Native-speaker review of the functional suite, augmentation templates and the 300-row gold sheet.
2. Browser-check the Moderator Console.
3. Repeat the augmented-MuRIL and baseline runs with 2-3 seeds before deploying augmentation (MuRIL member only).
4. Build a Romanized/code-mixed test set (candidate data in MODEL_SHORTLIST.md: unverified links/licenses).
5. Address quoted/reported-speech (mention vs use) failures with native-written cases.
6. Only then consider new backbones (mmBERT, IndicBERTv2) from MODEL_SHORTLIST.md.

See: POLICY_REPORT, FUNCTIONAL_TEST_REPORT, NEGATION_FIX_REPORT, LABEL_AUDIT_REPORT, AKSHARA_REPORT, DESPAM_REPORT, MODEL_SHORTLIST, DATASET_SURVEY, APPROACH_SURVEY.
