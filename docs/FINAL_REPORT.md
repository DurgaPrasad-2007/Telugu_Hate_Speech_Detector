# Final Report — Telugu / Code-mixed Abusive Content Classifier

## 1. Task
Binary classification (Abusive vs Non-Abusive) of Telugu and
Telugu-English code-mixed short social comments, built on the
`ShareChatAI/MACD` (NeurIPS 2022) dataset. Label semantics verified by
hash-matching against upstream (`docs/RESEARCH_REVIEW.md`), not assumed.

## 2. Data
- `telugu`: 23,980 / 3,000 / 3,000 train/val/test, near-balanced
  (52%/48% abusive/non-abusive), mean length ~86 chars, 27% code-mixed
  (Telugu+English).
- `codemixed`: 3,179 / 398 / 400, split roughly evenly, 55% Latin-only
  script (Romanized Telugu / English), 35% Telugu script.
- Exact-text train-to-val/test leakage found and removed at load time
  (not by editing CSVs): 2+4 rows for telugu, 6+11 for codemixed.
  Full numbers: `results/data_quality_report.md`.

## 3. Experiments run
Reduced matrix, chosen for the available time/GPU budget — **not** the
full 5-backbone x 5-method x 5-seed grid `src/run_experiments.py` was
originally scaffolded for. Stated plainly, not glossed over.

**Backbone comparison** (full fine-tune, 3 epochs, telugu domain):
| backbone | val macro-F1 | test macro-F1 |
|---|---|---|
| MuRIL | 0.9029 | 0.9045 |
| XLM-R | 0.9010 | 0.9042 |

A statistical tie (about 0.5 pt sampling noise). MuRIL was used as the PEFT backbone for the sweep below.
(MuRIL's validation score was backfilled after the fact; see section 4.)

**PEFT sweep on MuRIL** (telugu domain, 3 epochs each):
| method | val macro-F1 | test macro-F1 |
|---|---|---|
| DoRA | 0.8672 | 0.8628 |
| QLoRA | 0.8665 | 0.8701 |
| LoRA | 0.8642 | 0.8676 |
| IA3 | 0.7323 | 0.7261 |
| AdaLoRA | 0.3502 | 0.3425 |

Every PEFT method underperformed full fine-tuning by 3-16 points of
macro-F1. AdaLoRA collapsed to near-random on this binary task at the
tried config — reported as-is rather than dropped from the table.

## 4. Winner
Selected by **validation** macro-F1 across all runs (`src/finalize.py`):

**`baseline__muril__full__telugu`** — MuRIL, full fine-tune.
val macro-F1 **0.9029**, test macro-F1 **0.9045**
(0.9048 through the deployed pipeline, `results/deployed_pipeline_metrics.json`).

Correction to an earlier version of this report: XLM-R was first reported as the
winner "on validation", but MuRIL's original run had no saved validation score,
so the automatic ranking had skipped it. Its validation score was then measured
with the same method (0.9029 vs XLM-R 0.9007-0.9010), the selection re-run, and
calibration, thresholds, stress test and ONNX export regenerated for MuRIL.
XLM-R artifacts are preserved in `results/xlmr_reference/`. The two models are
statistically tied; MuRIL is also ~15% smaller. Trade-off found afterwards: with
the same guard, XLM-R resisted letter-spaced disguises better (89% vs 66%).

Because the winner is a full fine-tune rather than a PEFT adapter, the
"parameter-efficient" angle is a documented negative result for this
dataset/budget, not hidden.

## 5. Calibration
Temperature scaling fit on validation only (never test):
T=1.216, ECE 0.0374 -> 0.0141, Brier 0.0758 -> 0.0740 (`results/calibration.json`).

## 6. Robustness
See `docs/ROBUSTNESS.md`. Stress test on 300 real abusive + 300 clean test
comments with every word disguised: with the anti-obfuscation guard, abusive
comments are still caught 90-100% (spaced letters 66%). MuRIL's tokenizer
already ignores zero-width characters. No measurable cost on ordinary text
(test macro-F1 0.9045 -> 0.9048). Weak spots: code-mixed/Romanised input
(zero-shot macro-F1 0.76 vs 0.90) and implicit or double-meaning abuse (not
tested, not covered).

## 6b. Brutal obfuscation (measured)
Developed on the XLM-R candidate (0-32% caught before the guard, 89-100% after)
and then re-measured from scratch on the selected MuRIL model; tables in
`docs/ROBUSTNESS.md`.

## 7. Error analysis
9.58% test misclassification rate (287/2997) for MuRIL. The failure modes below
were hand-read on the XLM-R candidate's errors (285 of 2997); 70% of MuRIL's
errors are also XLM-R errors and the false-negative share is similar (60% vs
63%), so the same modes apply. MuRIL makes more high-confidence errors (49% at
0.9+ vs 36%). Modes, dominant first: long unpunctuated comments burying one
abusive clause, code-mixed/Romanised sentiment words, high-confidence wrong
predictions, emoji-carried sentiment. Detail: `docs/ERROR_ANALYSIS.md`.

## 8. Deployment
- Decision policy (`configs/policy.json`): allow below calibrated P(abusive)=0.45,
  review 0.45-0.50, block at 0.50 or above (a block never contradicts the
  argmax label) -> block precision 0.921, block recall 0.898, ~0.8% of traffic
  to review. Probabilities use the validation-fit temperature (T=1.216).
- Exported to ONNX fp32 and int8 (`onnx_model_muril*/`). int8 is ~4.6x faster
  than PyTorch on CPU in the same run; absolute times depend on machine load
  (an idle earlier session measured ~5x lower for both models). int8 accuracy
  has not been measured; the live app runs the PyTorch model.
- `src/serve.py` (FastAPI) and `app.py` (Streamlit) tested against the model.

## 9. Scope cut (stated plainly)
- Only 2 backbones compared, not the originally planned 4-5
  (mBERT/IndicBERT skipped) — time budget.
- Cross-domain transfer (telugu<->codemixed) and multi-seed
  variance study were planned but not run in this session.
- `src/run_experiments.py`'s full grid was never run wholesale — the
  hand-driven reduced matrix above is what is real and reported.

## 10. Reproduction
```bash
python src/train.py --model google/muril-base-cased --method full --domain telugu --epochs 3
python src/finalize.py
python src/calibration.py --model_dir results/baseline__muril__full__telugu/model
python src/robustness.py --model_dir results/baseline__muril__full__telugu/model --domain telugu --n 200
python src/stress_test.py            # add --no_defend for the no-guard column
python src/export_onnx.py --model_dir results/baseline__muril__full__telugu/model --out_dir onnx_model_muril --quantize
python src/benchmarking.py --model_dir results/baseline__muril__full__telugu/model --onnx_dir onnx_model_muril --onnx_int8_dir onnx_model_muril_int8
streamlit run app.py                 # defaults to the selected model
```
