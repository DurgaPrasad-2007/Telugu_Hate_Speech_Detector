# Deployment

The selected model is `models/best/` (XLM-R base + LoRA adapter). It runs on CPU for
serving. The base model is downloaded from the Hugging Face Hub on first load, so the
first start needs network access or a populated HF cache.

## Deployment variants

Measured on the development PC (Windows 11, CPU). Source: `results/final_best/export_metrics.json`.
Run: `xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024`. Preprocessing: normalized. Max length 96.

| Variant | Files | Size on disk | Val macro-F1 | Test macro-F1 | Agreement with fp32 (val / test) |
|---|---|---|---|---|---|
| A. fp32 PyTorch, adapter merged | `results/final_best/hf/` | 1.11 GB | 0.9073 | 0.9042 | reference |
| B. fp32 ONNX | `results/final_best/onnx/model.onnx` | 1.11 GB | not measured separately | not measured separately | not measured |
| C. INT8 ONNX (dynamic, per-channel) | `results/final_best/onnx_int8/model_quantized.onnx` | 279 MB | 0.8995 | 0.8926 | 0.961 / 0.968 |

- Accuracy cost of INT8 vs fp32: about 0.8 points on validation and 1.2 points on test
  macro-F1. Choose A or B if accuracy matters most, C if disk and latency matter more.
- Variant C uses `AutoQuantizationConfig.avx2(is_static=False, per_channel=True)`
  (`src/export_onnx.py`). It is **not** the per-tensor recipe in `onnx_model_muril_int8/`,
  which is broken and deprecated.

## Latency (CPU, batch size 1, tokenization included)

Per comment, first 200 test comments, 10 warm-up calls. Measured on the development PC.
These are not low-end laptop numbers. Expect slower results on weaker hardware.

| Variant | 1 thread: median (mean, p90) | 4 threads: median (mean, p90) |
|---|---|---|
| INT8 ONNX (C) | 23.1 ms (30.6, 63.6) | 8.3 ms (10.7, 20.5) |
| fp32 PyTorch (A) | 43.1 ms (52.8, 96.3) | 17.9 ms (20.9, 35.0) |

**Memory:** not measured for any variant.

## Requirements

Serving (`requirements-deploy.txt`) plus the CPU torch wheel:

```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements-deploy.txt
```

Training and export need the full `requirements.txt` on a CUDA machine (training was run
on an RTX 4060 Ti, 8 GB). Export runs on CPU (`CUDA_VISIBLE_DEVICES` is cleared in the script).

## Run the selected model

```bash
# CLI (PyTorch with adapter; base model fetched from HF on first run)
python src/infer.py --model_dir models/best --text "your comment here"

# HTTP API
MODEL_DIR=models/best uvicorn src.serve:app --host 0.0.0.0 --port 8000
curl localhost:8000/readyz
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{"texts":["..."]}'
```

PowerShell: `$env:MODEL_DIR="models/best"; uvicorn src.serve:app --port 8000`.

`/health` is liveness only. `/readyz` loads the model and runs a real inference. `/version`
reports the model directory in use. `MAX_BATCH` (default 64) and `MAX_TEXT_CHARS`
(default 2000) cap request size. Raw text is not logged at INFO level.

Streamlit demo:

```bash
streamlit run app.py          # defaults to models/best; set MODEL_DIR to override
```

Serving the INT8 ONNX folder (`results/final_best/onnx_int8`) is supported by the loader
(`infer.py` detects ONNX directories), but this was **not re-run in this pass**. Check it
before relying on it. The same applies to the fp32 ONNX folder.

## Export

```bash
# Merged fp32 HF model, fp32 ONNX, INT8 ONNX, latency and agreement for one study run
python src/export_best.py --run xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024 --out results/final_best

# ONNX only, from an adapter directory (base model needed for PEFT)
python src/export_onnx.py --model_dir models/best --out_dir results/onnx_only --quantize
```

`export_best.py` reads the adapter from `results/study/<run>/model/`. That folder is
git-ignored, so on a fresh clone use `models/best/` with `export_onnx.py` or copy the adapter
back into the study folder.

## Policy configuration

`configs/policy.json`: `{"low": 0.45, "high": 0.50}`, applied to temperature-scaled P(abusive)
(T = 1.334 from `results/calibration.json`).
- P < 0.45: allow
- 0.45 <= P < 0.50: review
- P >= 0.50: block

Validation at this pair: block precision 0.9277, block recall 0.8973, 1.07% of comments
sent to review (`results/threshold_sweep.csv`). `set_policy.py` searches thresholds with
block precision >= 0.9, maximising recall.

Note: `infer.py` loads `configs/policy.json` and `results/calibration.json` regardless of
model directory. These files were written for the selected model, but the files do not
record which model they were fitted on.

## Known risks before production

- The first start needs the base model from the Hugging Face Hub.
- No load testing, no per-request timeout, one classifier per process.
- Obfuscation and false-positive behaviour: see [MODEL_CARD.md](MODEL_CARD.md). Human review
  is required for review-band and block decisions.
- Drift: slang and obfuscation change over time. Keep a labelled sample and re-measure
  precision and recall. The adapter can be retrained without touching the base weights.

## Deprecated

- `onnx_model_muril_int8/`: broken per-tensor INT8 MuRIL model (about 0.37 macro-F1 on a
  600-comment sample; predicts abusive about 3% of the time). Do not deploy. See its
  `DEPRECATED.md`.
- The MuRIL full-fine-tune winner and its exports were removed from disk. Only their
  metrics remain under `results/`.
