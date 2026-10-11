# Model Card: Telugu Abusive-Comment Classifier

## Model
- **Backbone:** `FacebookAI/xlm-roberta-base`, sequence-classification head (2 classes).
- **Adapter:** LoRA, r=8, alpha=64, target modules `query` and `value`, dropout 0.05.
  Trainable parameters: 887,042 (0.32% of 278.9M total).
- **Shipped weights:** `models/best/` (adapter, 3.6 MB). The base model is fetched from the
  Hugging Face Hub at load time.
- **Run:** `xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024` (seed 2024). LR 2e-4, cosine
  schedule, batch 16, max length 96, weight decay 0.01, 10% warm-up, early stopping
  (patience 3) on validation macro-F1. Best epoch 6 of 10.
- **Label convention:** 1 = abusive, 0 = non-abusive.

## Selection
- Rule (`results/study/selection.md`, validation macro-F1 only; test is never used by the code):
  1. Group runs by configuration; keep configurations with at least 3 seeds.
  2. Top = highest mean validation macro-F1.
  3. Tie set = configurations with mean validation >= top mean - top SD.
  4. Winner = the tie-set member with the fewest trainable parameters.
- Top configuration: XLM-R full fine-tune, val 0.9040 ± 0.0011 (tie threshold 0.9030).
- Winner: XLM-R LoRA, val 0.9035 ± 0.0043, 887k trainable parameters.
- **The selection is knife-edge.** The winner's mean clears the tie threshold by 0.0005.
  MuRIL LoRA (val 0.9026 ± 0.0020, 296k parameters) misses it by 0.0004. Had it entered
  the tie set, it would have won the parameter rule. A different tie rule would change the
  choice.
- Caveat: test numbers were printed for every run while the choice was made. Selection
  used validation only, but the test column is for reporting, not a clean holdout.

## Data
- `data/telugu_{train,val,test}.csv`: pure Telugu, from the ShareChatAI/MACD dataset
  (the Telugu 80/10/10 split, per earlier project notes; not re-verified in this pass).
- Raw rows: 23,980 / 3,000 / 3,000. After the loader drops validation and test texts that
  also appear in train: 2,998 validation and 2,997 test rows are evaluated.
- The code-mixed CSVs were removed from `data/` and are not used.
  **Code-mixed performance was not evaluated in this study.** Earlier claims about
  code-mixed numbers (including a zero-shot macro-F1 of 0.76) are withdrawn and do not
  describe this model.

## Training and comparison (telugu, 3 seeds: 42, 123, 2024, mean ± sample SD)

Macro-F1. Test values are reporting only.

| Experiment | Config | Val | Test | Trainable params |
|---|---|---|---|---|
| E1 | XLM-R full FT, lr 2e-5, 5 ep | 0.9040 ± 0.0011 | 0.9057 ± 0.0033 | 278.0M |
| E2 | XLM-R LoRA r8 a64, lr 2e-4 (**selected**) | 0.9035 ± 0.0043 | 0.9035 ± 0.0033 | 887k |
| E5 | XLM-R DoRA r8 a64 | 0.9027 ± 0.0011 | 0.9055 ± 0.0009 | 905k |
| E4 | XLM-R AdaLoRA r8 a64 | 0.8855 ± 0.0015 | 0.8891 ± 0.0013 | 1.03M |
| extra | MuRIL full FT | 0.9035 ± 0.0023 | 0.9049 ± 0.0028 | 237.6M |
| extra | MuRIL LoRA r8 a64 | 0.9026 ± 0.0020 | 0.9052 ± 0.0030 | 296k |
| E6 | mBERT LoRA | 0.8999 ± 0.0010 | 0.8946 ± 0.0015 | 296k |
| E7 | IndicBERTv2-MLM-only LoRA | 0.8985 ± 0.0014 | 0.9017 ± 0.0039 | 296k |

- Shipped run (seed 2024): val macro-F1 0.9073, test macro-F1 0.9042
  (test accuracy 0.9042, test precision 0.9043, test recall 0.9050).
- Study run count and details: `results/study/study_results.csv`, `docs/models/INDEX.md`.
- The rank x alpha grid, learning-rate ablation and single-seed runs are in
  `docs/PEFT_STUDY_REPORT.md`.

## Calibration
- Temperature scaling on validation only (n=2,998): T = 1.334.
- ECE 0.0353 -> 0.0129. Brier 0.0730 -> 0.0711. (`results/calibration.json`)

## Decision policy
`configs/policy.json`: `low` = 0.45, `high` = 0.50, applied to calibrated P(abusive).
- allow: P < 0.45
- review: 0.45 <= P < 0.50
- block: P >= 0.50

On the validation set at this threshold pair (`results/threshold_sweep.csv`):
- block precision 0.9277
- block recall 0.8973
- 1.07% of validation comments go to review

## Robustness
Two measurements from `results/robustness_results_summary.json` and
`results/stress_test_summary.json`. Both were regenerated on 2026-10-11, after the model
was selected. Both were produced by one scripted run (`results/run_best_checks.ps1`) with
`--model_dir models/best`, i.e. the selected XLM-R LoRA model; the files themselves do not
record the model path.

**Perturbations** (200 test comments per transform; baseline accuracy 0.875).
Flip rate = share of predictions that change.

| Perturbation | Flip rate | Accuracy vs label |
|---|---|---|
| Character repetition | 12.5% | 0.780 |
| Extra spacing | 5.5% | 0.860 |
| Number substitution | 1.0% | 0.865 |
| Random case | 1.5% | 0.870 |
| Emoji insertion | 2.0% | 0.875 |
| Punctuation insertion | 0.5% | 0.870 |
| Capitalisation | 0.5% | 0.880 |

**Obfuscated abuse** (300 abusive and 300 clean test comments; guard on by default).
Columns are as the script labels them. "Clean wrongly blocked" is the share of clean
comments that were blocked under that morph.

| Morph | Abusive caught % | Clean wrongly blocked % |
|---|---|---|
| Zero-width joiner | 100.0 | 0.0 |
| Zero-width non-joiner | 100.0 | 0.0 |
| Letter spacing | 77.3 | 5.0 |
| Dots between letters | 98.7 | 1.0 |
| Dashes between letters | 98.7 | 3.3 |
| Emoji inside word | 100.0 | 2.7 |
| Emoji wrapped | 100.0 | 5.3 |
| Character repetition | 92.3 | 24.3 |
| Leetspeak digits | 99.0 | 2.3 |
| Star-censored | 100.0 | **99.0** |
| UPPERCASE | 99.3 | 1.0 |
| Trailing emoji spam | 100.0 | 6.0 |
| Combined (repeat + ZWJ + emoji) | 93.7 | 33.7 |

The star-censored row (99% of clean comments blocked) and the high false-block rates
under character repetition and the combined morph need checking before anyone relies on them.
They may be a measurement artefact of how the clean set was morphed.

## Intended use
- Assisting human moderators: flagging abusive or offensive Telugu social-media comments
  for review, using the allow / review / block policy above.
- The model's block decision is an input to a person, not a final action on its own.

## Out of scope
- Languages other than Telugu, and Telugu-English code-mixed or Romanised text as a
  supported case (not evaluated).
- Images, video, audio.
- Fully automated removal or sanctions without human review.
- Any register not represented in the MACD Telugu data.

## Limitations
- One dataset, one language, three seeds per configuration. Selection margin is 0.0005
  (see Selection).
- Annotation guidelines and annotator judgement shape the labels. Label noise is expected.
- Obfuscation: letter spacing (77%) and character repetition (92%) lose recall. Abuse that
  is implied or relies on double meaning is not tested.
- Slang and obfuscation drift over time. Recall will drop unless the model is re-evaluated.
- No dialect labels, so dialect-level fairness was not measured.
- Latency and memory were measured on the development PC only.

## Ethical considerations
- **False positives on political and critical speech are a real risk.** Criticism of
  politicians, institutions or communities can use strong language, and a model trained on
  abuse labels can flag it. This study did not measure false-positive rates on political or
  critical speech. Every block or review decision on such content needs a human reviewer, and
  the policy should not be used to suppress criticism.
- Misclassification has costs in both directions. A missed abusive comment harms the people
  it targets. A wrongly blocked comment silences a user. The policy thresholds trade these
  off on validation data only. The operator decides which error costs more.
- The labels reflect one annotation effort. They are not ground truth about any individual
  or group.

## Deployment artifacts
See [DEPLOYMENT.md](DEPLOYMENT.md). Short version:
- PyTorch model with adapter: `models/best/`.
- Merged fp32 HF model, fp32 ONNX and dynamic INT8 ONNX of the same run:
  `results/final_best/` (INT8 numbers in `docs/DEPLOYMENT.md`).
- `onnx_model_muril_int8/` is deprecated and must not be used.
