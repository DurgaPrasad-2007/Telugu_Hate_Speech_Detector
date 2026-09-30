# Deployment

## Installation
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # + optimum[onnxruntime], streamlit (see requirements.txt)
```
Measured environment for all numbers in this doc: Windows 11, Python
3.11, `torch==2.6.0+cu124`, `transformers==5.17.0`, `peft==0.21.0`,
`onnxruntime==1.30.0`, `optimum==2.1.0`, GPU = RTX 4060 Ti (8 GB VRAM).

## GPU execution (training)
```bash
python src/train.py --model google/muril-base-cased --method lora --domain telugu --epochs 3
```
Target modules for a new backbone: run
```bash
python -c "from transformers import AutoModel; m=AutoModel.from_pretrained('...'); print(sorted(set(n.split('.')[-1] for n,_ in m.named_modules())))"
```
and add the result to `configs/targets.json` before training — do not
reuse `query,value` blindly (see `docs/RESEARCH_REVIEW.md` sec 2).

## CPU execution
```bash
python src/infer.py --model_dir results/<run>/model --text "some text"
```
`HateSpeechClassifier` auto-selects `cuda` if available, else `cpu` — no
code change needed.

## ONNX / quantized execution
```bash
python src/export_onnx.py --model_dir results/<run>/model --out_dir onnx_model --quantize
python src/benchmarking.py --model_dir results/<run>/model --onnx_dir onnx_model --onnx_int8_dir onnx_model_int8
```

## Streamlit
```bash
MODEL_DIR=results/<winning_run>/model streamlit run app.py
```
Model is loaded once via `@st.cache_resource`; falls back to CPU if no
GPU is present. Configure via `MODEL_DIR` / `BASE_MODEL_FOR_ADAPTER` env
vars, not code edits.

## API
```bash
MODEL_DIR=results/<winning_run>/model uvicorn src.serve:app --host 0.0.0.0 --port 8000
curl localhost:8000/readyz
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{"texts":["..."]}'
```
`/health` = liveness only. `/readyz` = model loaded AND a real inference
actually runs. `/version` reports the model dir in use. `MAX_BATCH` /
`MAX_TEXT_CHARS` env vars cap request size; raw text is never written to
INFO logs (only counts/latency), per Rule 9.

## Threshold configuration
`configs/policy.json` — `{"low": ..., "high": ...}` on P(abusive):
below `low` → allow, above `high` → block, between → human review. Values
are picked from a validation-set precision/recall/review-rate sweep (see
`docs/FINAL_REPORT.md`), not arbitrary.

## Monitoring / retraining
Track over time (see `docs/FINAL_REPORT.md` "Monitoring" for the concrete
plan): input length distribution, script mix (Telugu vs Romanized vs
mixed), P(abusive) distribution, review-band %, confidence drift, and a
periodically-relabeled sample for real accuracy tracking. A new LoRA/DoRA
adapter can be retrained on top of the frozen backbone without touching
the base weights — cheaper than a full retrain when only recent slang
needs to be captured.
