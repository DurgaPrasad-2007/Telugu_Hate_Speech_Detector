# Final Report (corrected summary)

This report replaces an earlier version. That version named a MuRIL full fine-tune as the
winner, and its numbers, calibration, policy and deployment claims are out of date. The
current state is below. Full evidence is in:

- [PEFT_STUDY_REPORT.md](PEFT_STUDY_REPORT.md): the parameter-efficient study, seeds, grids and quantization.
- [ARCHITECTURE.md](ARCHITECTURE.md): pipeline, selection rule, and components.
- [models/INDEX.md](models/INDEX.md): per-experiment reports and figures.
- [MODEL_CARD.md](MODEL_CARD.md): intended use, limitations, ethics.
- [DEPLOYMENT.md](DEPLOYMENT.md): deployment variants, latency, export and run commands.

(`docs/PAPER_ALIGNMENT.md` is referenced by an earlier plan but does not exist in this repository.)

## Task and data
Binary classification (1 = abusive, 0 = non-abusive) of short Telugu social-media comments.
Data: pure Telugu MACD split. 23,980 / 3,000 / 3,000 train / validation / test rows before
the loader drops validation and test texts that also appear in train. That leaves 2,998
validation and 2,997 test rows. The code-mixed data was removed and is not used. Code-mixed
performance was not evaluated.

## Selected model
- `FacebookAI/xlm-roberta-base` + LoRA (r=8, alpha=64, `query`/`value`, dropout 0.05),
  stored in `models/best/`.
- Run `xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024`: val macro-F1 0.9073, test macro-F1 0.9042.
- Configuration over 3 seeds: val 0.9035 ± 0.0043, test 0.9035 ± 0.0033.
- Chosen by the validation-only tie-set rule in `results/study/selection.md`. The margin is
  knife-edge: the winner clears the tie threshold (0.9030) by 0.0005, and MuRIL LoRA
  (0.9026) misses it by 0.0004. See MODEL_CARD.md.

## Calibration and policy
- Temperature T = 1.334 (validation). ECE 0.0353 -> 0.0129. Brier 0.0730 -> 0.0711.
- Thresholds `low` = 0.45, `high` = 0.50. On validation: block precision 0.9277,
  block recall 0.8973, 1.07% of comments to review.

## Deployment
- fp32 PyTorch (adapter merged): val 0.9073 / test 0.9042, 1.11 GB.
- INT8 ONNX (dynamic, per-channel): val 0.8995 / test 0.8926, 279 MB. Agreement with
  fp32 0.961 / 0.968.
- Latency on the development PC (CPU, batch 1, tokenization included): INT8 median 8.3 ms
  at 4 threads, 23.1 ms at 1 thread. fp32 median 17.9 ms at 4 threads, 43.1 ms at 1 thread.
  Memory not measured. Not measured on low-end hardware.
- `onnx_model_muril_int8/` is broken and deprecated (about 0.37 macro-F1 on a 600-comment
  sample). See its `DEPRECATED.md`.

## Robustness
- Perturbations (200 comments per transform, baseline accuracy 0.875): character repetition
  causes the most prediction flips (12.5%), then extra spacing (5.5%). Other transforms flip
  at most 2%.
- Obfuscated abuse: letter spacing is caught 77.3% of the time, character repetition 92.3%.
  Some clean-text false-block numbers in the stress test are very high (star-censored 99%).
  These need checking before use. MODEL_CARD.md has the full tables.

## Removed or withdrawn
- Code-mixed numbers, including the 0.76 zero-shot macro-F1 and `results/codemixed_zero_shot.json`,
  do not describe this model and are withdrawn.
- The MuRIL full-fine-tune winner and its ONNX/INT8 exports were removed from disk. Only
  their metrics remain.
- The earlier claims that every PEFT method trailed full fine-tuning by 3-16 points were an
  artefact of undertraining. The corrected study finds LoRA within noise of full fine-tuning
  for XLM-R.

## Scope and open items
- One dataset, one language, three seeds per configuration. The test column was visible
  during selection, so it is a report, not a clean holdout.
- Error analysis, robustness and ethics documents for the selected model still need a full review (see PROGRESS.md).
