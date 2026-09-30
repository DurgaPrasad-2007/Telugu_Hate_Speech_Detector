# Model Card — Telugu / Code-mixed Abusive Content Classifier

## Model
- **Selected: `google/muril-base-cased`, full fine-tune** (not a PEFT adapter).
  Chosen by validation macro-F1 among two backbones and six adaptation
  methods (7 runs; matrix in `results/all_results.csv`).
- Backbone comparison (full fine-tune, telugu domain, 3 epochs):
  MuRIL val **0.9029** / test **0.9045** vs XLM-R val 0.9007 / test 0.9042.
  A statistical tie (sampling noise is about 0.5 pt at 3,000 rows) with MuRIL
  nominally ahead on both; it also has ~15% fewer parameters and is
  pre-trained on Indic text including transliteration.
- Selection note (corrected): MuRIL's original run had no saved validation
  score, so the automatic ranking initially skipped it and XLM-R was picked by
  default. MuRIL's validation score was later measured with the same method
  and backfilled (`results/baseline__muril__full__telugu/metrics.json`,
  flagged as backfilled); selection was then re-run and the whole pipeline
  (calibration, thresholds, stress test, ONNX) regenerated for MuRIL. The
  XLM-R artifacts are kept in `results/xlmr_reference/`.
- PEFT sweep on MuRIL (all underperformed full fine-tune): LoRA val 0.8642 /
  test 0.8676; QLoRA 0.8665 / 0.8701; DoRA 0.8672 / 0.8628 (best PEFT by val);
  IA3 0.7323 / 0.7261; AdaLoRA 0.3502 / 0.3425 (collapsed to near-random at the
  tried config; reported as-is).
- Deployed pipeline (`src/infer.py`: anti-obfuscation guard + emoji-stripped
  second view + temperature): val 0.9029, test 0.9048 macro-F1
  (`results/deployed_pipeline_metrics.json`), i.e. no measurable cost.
- Target modules: derived from `named_modules()` inspection, not assumed
  (`configs/targets.json`).

## Data
- `telugu_{train,val,test}.csv`: byte-identical to `ShareChatAI/MACD`
  (NeurIPS 2022) `dataset_80_10_10` Telugu split, ~24K/3K/3K rows.
- `codemixed_{train,validation,test}.csv`: same repo, ~3.2K/0.4K/0.4K,
  Telugu-English code-mixed (mix of Telugu script, Romanized Telugu, and
  Telugu+English switching — see `results/data_quality_report.md`).
- Train-into-val/test text leakage found and filtered at load time (not by
  editing the CSVs) in `src/data_utils.py`; counts documented there.

## Intended use
Assisting human moderators in flagging abusive/offensive Telugu and
Telugu-English code-mixed social comments (YouTube/Instagram/Google
comment-style short text) for review, using the allow/review/block policy
in `configs/policy.json`.

## Out-of-scope
- English-only or other-language text.
- Images, video, audio.
- Fully automated moderation without human review on the "review" band.
- Any language or code-mixing pattern not represented in MACD's Telugu
  data (this is not a general-purpose any-language hate-speech detector).

## Limitations
- Trained on a specific dataset's annotation guidelines/annotators; label
  noise and subjective abuse judgments are expected (see
  `docs/ERROR_ANALYSIS.md`).
- Romanized-Telugu coverage is uneven (~55% of code-mixed train is
  Latin-only per the data audit) — robustness on transliteration variants
  not seen in training is untested beyond `docs/ROBUSTNESS.md`.
- No dialect labels exist in the data, so dialect-level fairness can't be
  assessed; flagged as an open limitation, not measured.
- Temporal/slang drift: the data reflects language use at collection
  time; new slang and obfuscation patterns will degrade recall over time
  (see `docs/DEPLOYMENT.md` monitoring section).

## Bias / fairness
Error rates broken down by script (Telugu-only / Latin-only / mixed) are
in `docs/ERROR_ANALYSIS.md`. No claim is made beyond what was measured.

## Calibration
Temperature scaling on validation only: T=1.216, ECE 0.0374 -> 0.0141,
Brier 0.0758 -> 0.0740. Full detail: `results/calibration.json`.

## Robustness
See `docs/ROBUSTNESS.md`. Summary: with the guard, abusive comments whose every
word is disguised are still caught 90-100% of the time (spaced letters 66%);
zero-width characters, emoji and casing were already handled by MuRIL's
tokenizer. Weak spots: code-mixed/Romanised input (zero-shot macro-F1 0.76 vs
0.90 on Telugu, `results/codemixed_zero_shot.json`) and implicit or
double-meaning abuse (not tested, not covered).

## Deployment artifacts
- PyTorch model: `results/baseline__muril__full__telugu/model`
- ONNX fp32: `onnx_model_muril/`, ONNX int8: `onnx_model_muril_int8/`
  (XLM-R exports remain in `onnx_model/`, `onnx_model_int8/`).
- Latency (200-sample bench, `results/benchmark_results.json`): int8 ONNX is
  about 4.6x faster than PyTorch on CPU in the same run (28 vs 131 ms/comment).
  Absolute times depend on machine load: an idle earlier session measured about
  5x lower, and XLM-R and MuRIL measured equal on the same machine
  (`results/benchmark_xlmr_now.json`). int8 accuracy has not been measured.
- Policy thresholds (`configs/policy.json`): low=0.45, high=0.50 on calibrated
  probabilities -> block precision 0.921, block recall 0.898, ~0.8% of traffic
  routed to human review.
