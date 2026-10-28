# Progress

Current state of the project, as of 2026-10-11. The model card is [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Current model
- `FacebookAI/xlm-roberta-base` + LoRA (r=8, alpha=64) in `models/best/`.
- Run `xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024`: val macro-F1 0.9073, test 0.9042.
- Configuration mean over 3 seeds: val 0.9035 ± 0.0043, test 0.9035 ± 0.0033.
- Selected by validation-only tie-set rule (`results/study/selection.md`). Knife-edge: margin
  0.0005 over the tie threshold.

## Done
- Pure Telugu study: LoRA vs full FT, DoRA, AdaLoRA, backbones (XLM-R, MuRIL, mBERT, IndicBERTv2),
  rank x alpha grid, learning-rate ablation. See `docs/PEFT_STUDY_REPORT.md`.
- Selection (`src/select_best.py`), calibration (T = 1.334, ECE 0.0353 -> 0.0129),
  policy (`configs/policy.json`: low 0.45, high 0.50; block precision 0.9277, recall 0.8973 on validation).
- Export: fp32 PyTorch (merged), fp32 ONNX, INT8 ONNX (dynamic, per-channel) under `results/final_best/`.
  INT8 val 0.8995 / test 0.8926. Latency on the development PC only.
- Robustness and obfuscation tests regenerated on 2026-10-11 (`results/robustness_results_summary.json`,
  `results/stress_test_summary.json`). The stress script's default model path is the old MuRIL one,
  so confirm which model each file was run with.
- Docs refreshed: README, `docs/MODEL_CARD.md`, `docs/DEPLOYMENT.md`, `docs/FINAL_REPORT.md`.
- `onnx_model_muril_int8/` marked deprecated (`DEPRECATED.md`). Its files are kept.

## Removed or withdrawn
- Code-mixed CSVs removed from `data/`. Code-mixed results (including the 0.76 zero-shot figure)
  are withdrawn. Code-mixed performance was not evaluated in this study.
- MuRIL full-FT winner and its ONNX/INT8 exports removed from disk; metrics remain.

## Status & Architecture Alignment
- `render.yaml`, `Dockerfile`, `docker-compose.yml`, `README.md`, and `src/stress_test.py` consistently target `models/best` (shipped LoRA adapter) and `results/final_best/onnx_int8` (shipped per-channel dynamic INT8 ONNX).
- `docs/ARCHITECTURE.md` updated to reflect the codebase's actual `max_length=96`, `serve.py` defaults to `models/best`, and `results/final_best/` is documented.
- `docs/PAPER_ALIGNMENT.md` is present in `docs/`.
- `src/serve.py` title aligned to pure Telugu abusive content moderation.
- Offline tests (`tests/smoke_test.py`) verify the full-FT, LoRA adapter merge, and inference execution pipeline end-to-end.

## How to run
```bash
python src/infer.py --model_dir models/best --text "..."
MODEL_DIR=models/best uvicorn src.serve:app --port 8000
streamlit run app.py
```
