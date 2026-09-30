# Telugu / Code-mixed Abusive Content Detection

A parameter-efficient fine-tuning (PEFT) research pipeline for detecting
abusive/hate content in **Telugu and code-mixed Telugu-English text**
(the register used in Google/Instagram-style comments), built for an SCI
paper: a systematic comparison of Full Fine-Tuning vs LoRA vs QLoRA vs
AdaLoRA vs DoRA vs IA3.

## Honest scope note

This is **not** a general-purpose, any-language, any-format ("catches
hate speech in any format used on Google/Instagram") production model.
Building that requires massive multilingual + multimodal training data
and infrastructure at a scale no individual project can replicate. What
this pipeline gives you:

- A rigorous, publishable comparison of adaptation strategies for
  **low-resource Telugu abusive-content detection**, using your two real
  datasets (pure Telugu, ~24K/3K/3K train/val/test; code-mixed
  Telugu-English, ~3.2K/0.4K/0.4K).
- The **best resulting model is fast and genuinely deployable** as a
  Telugu/code-mixed-Telugu text classifier (via `infer.py`, `serve.py`,
  or the ONNX export) — that part is real and usable.
- It does **not** extend to English-only hate speech, images/video, or
  languages the training data didn't cover, without retraining on data
  for those cases.

## Dataset

| File | Rows | Domain |
|---|---|---|
| `telugu_train/val/test.csv` | 24000/3000/3000 | Pure Telugu |
| `codemixed_train/validation/test.csv` | 3200/400/400 | Telugu-English code-mixed |

Both are `text,label` with `label`: 0 = Non-Abusive, 1 = Abusive.

## Project layout

```
data/                 the 6 CSVs
src/
  data_utils.py        loading, cleaning, normalization (raw/normalized/clean levels)
  metrics_utils.py      accuracy/precision/recall/macro-F1, param counting, GPU memory/latency tracking
  train.py               single-run trainer: --model --method {full,lora,qlora,adalora,dora,ia3} --domain
  run_experiments.py     orchestrates the full comparison matrix (baselines/peft/seeds/ablations/cross-domain)
  analysis.py             seed mean±std tables, paired significance tests, error analysis, main results table
  infer.py                 fast inference wrapper (loads full model OR PEFT adapter, auto-merges adapter)
  serve.py                  FastAPI HTTP endpoint for production use
  export_onnx.py            ONNX + int8 quantized export for lower-latency serving
notebooks/run_all_experiments.ipynb   ready-to-run Colab notebook, GPU required
tests/smoke_test.py                     offline sanity check of the whole pipeline (no network needed)
results/                                    all run outputs land here (metrics.json, confusion matrices, saved models)
```

## Quickstart (on Colab / a GPU machine — HF Hub access required)

```bash
pip install -r requirements.txt

# one run
python src/train.py --model xlm-roberta-base --method qlora --domain telugu --epochs 5

# the full paper-scale sweep (baselines + PEFT methods + seeds + ablations + cross-domain)
python src/run_experiments.py --stage all --domain telugu --epochs 5

# aggregate everything into paper tables
python src/analysis.py --action main_table --domain telugu
python src/analysis.py --action seed_summary
python src/analysis.py --action significance --method_a full --method_b qlora
```

Or just open `notebooks/run_all_experiments.ipynb` in Colab and run cells
top to bottom — it does all of the above with a GPU runtime.

**Why not run in this sandbox:** this environment's network is allowlisted
and doesn't include huggingface.co, so model downloads (xlm-roberta-base,
mBERT, MuRIL, etc.) fail here. `tests/smoke_test.py` validates the entire
codepath (training loop, LoRA wiring, param counting, save/reload/merge,
inference) offline using a tiny locally-trained tokenizer + randomly
initialized model, and all of it passes — the pipeline itself is correct
and just needs a GPU environment with internet access to run for real.

## What each experiment stage produces

Every run in `results/<run_name>/` contains:
- `metrics.json` — accuracy, precision, recall, macro-F1, weighted-F1, total/trainable params, trainable %, training time, peak GPU memory, inference latency/throughput
- `confusion_matrix.png`
- `classification_report.txt`
- `model/` — the saved full model or PEFT adapter

`run_experiments.py --stage aggregate` merges all `metrics.json` files into
`results/all_results.csv`, which `analysis.py` turns into the paper's main
comparison table, the mean±std multi-seed table, and paired significance
tests (paired t-test between two methods across matching seeds).

## 🚀 Instant Run on Any GPU-less Laptop (e.g. 2019 Core i3/i5, 4GB RAM)

The repository **includes the winning quantized INT8 MuRIL model** (`onnx_model_muril_int8/`) via Git LFS. It runs locally on pure CPU with **~7.6ms per sample** and uses under **300 MB of RAM**.

### 1. One-line Setup (Terminal / PowerShell)
```bash
# Clone repo (Git LFS pulls the quantized weights automatically)
git clone https://github.com/DurgaPrasad-2007/Telugu_Hate_Speech_Detector.git
cd Telugu_Hate_Speech_Detector

# Create virtual environment & install lightweight CPU packages
python -m venv .venv

# On Windows:
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements-deploy.txt

# On Linux/macOS:
# source .venv/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-deploy.txt
```

### 2. Instant Run Interactive Web App (Streamlit)
```bash
# Runs immediately using the bundled lightweight CPU model:
.venv\Scripts\streamlit run app.py
```
Open `http://localhost:8501` in your browser to test live text screening, anti-obfuscation defenses, and blurred word highlights!

### 3. Instant Run CLI Inference
```bash
.venv\Scripts\python src/infer.py --model_dir onnx_model_muril_int8 --text "చాలా మంచి వీడియో"
```

---

## 🌐 Cloud Deployment (Render Blueprint)

This repository includes a pre-configured [render.yaml](render.yaml) blueprint tailored for CPU serving.

1. Push this repository to your GitHub account (`git push origin main`).
2. Go to **[Render Dashboard](https://dashboard.render.com/)** > **New +** > **Blueprint**.
3. Select this repository (`Telugu_Hate_Speech_Detector`).
4. Render automatically reads `render.yaml`, installs CPU-only PyTorch, loads `onnx_model_muril_int8/`, and starts the Streamlit app. It stays well within memory limits (<300 MB).

---

## Local Service & CLI

```bash
# CLI
python src/infer.py --model_dir onnx_model_muril_int8 --text "your text here"

# HTTP REST API (FastAPI)
MODEL_DIR=onnx_model_muril_int8 uvicorn src.serve:app --host 0.0.0.0 --port 8000
# Test health: curl localhost:8000/readyz
# POST /predict {"texts": ["మీ వ్యాఖ్య ఇక్కడ"]}
```

`infer.py` auto-detects whether `model_dir` holds an ONNX model, full PyTorch model, or PEFT adapter (merging adapter weights into the base model on load).

## Notes on faithfulness to your professor's plan

- Preprocessing keeps abusive-relevant tokens — no aggressive stopword
  removal (`data_utils.normalize_text` only masks URLs/mentions and
  normalizes whitespace/unicode; `level="clean"` additionally collapses
  character-repetition noise).
- Classification is framed as supervised binary classification (label 0/1
  via a sequence-classification head), not free-text LLM generation, so
  results are directly comparable across mBERT/IndicBERT/MuRIL/XLM-R and
  all PEFT variants — exactly as recommended.
- IA3, AdaLoRA and DoRA are included alongside LoRA/QLoRA since your
  professor's later message said not to restrict to QLoRA alone.
- Cross-domain transfer (`telugu -> codemixed`, `codemixed -> telugu`) is
  wired up via `--eval_domain`, matching the MACD/HOLD generalization idea
  from the chat.
