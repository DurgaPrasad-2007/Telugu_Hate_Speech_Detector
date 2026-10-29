# Telugu Abusive-Comment Detector

Binary classifier (abusive / non-abusive) for short **Telugu** social-media comments.
Selected model: `FacebookAI/xlm-roberta-base` with a LoRA adapter (r=8, alpha=64),
shipped in `models/best/`. Decisions are allow / review / block using thresholds
in `configs/policy.json`. Model details: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Results

Macro-F1. Mean ± sample SD over 3 seeds (42, 123, 2024) for the selected configuration,
and the shipped run (seed 2024) alone.

| | Validation | Test |
|---|---|---|
| Selected config, mean ± SD (n=3) | 0.9035 ± 0.0043 | 0.9035 ± 0.0033 |
| Shipped run (seed 2024) | 0.9073 | 0.9042 |
| Shipped run, INT8 ONNX (dynamic, per-channel) | 0.8995 | 0.8926 |

Full-fine-tune XLM-R (E1) reaches 0.9040 ± 0.0011 val and 0.9057 ± 0.0033 test, which
is within the seed noise. The selection rule chose the LoRA config because it trains 887k
parameters instead of 278M. See [docs/PEFT_STUDY_REPORT.md](docs/PEFT_STUDY_REPORT.md).


**Start here:** [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) is the one-page verdict with the proofs. Headline: LoRA matches full fine-tuning (no detectable difference, paired tests), at about 0.3% of the trainable parameters; a TF-IDF + LinearSVC baseline is just as accurate on this dataset; the input guard, not the model, drives robustness.

## Quick start

Inference runs on CPU. The adapter needs the base model `FacebookAI/xlm-roberta-base`,
downloaded from the Hugging Face Hub on first use (or taken from the local cache).

```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements-deploy.txt

# CLI
.venv\Scripts\python src/infer.py --model_dir models/best --text "your comment here"

# HTTP API
$env:MODEL_DIR="models/best"; .venv\Scripts\uvicorn src.serve:app --port 8000
curl localhost:8000/readyz

# Streamlit demo
.venv\Scripts\streamlit run app.py
```

Training needs a CUDA GPU (about 8 GB VRAM was used) and the full `requirements.txt`.

## Repo map

```
data/                       telugu_{train,val,test}.csv (pure Telugu, MACD split)
models/best/                selected LoRA adapter + tokenizer (the shipped model)
configs/policy.json         allow / review / block thresholds
configs/targets.json        LoRA target modules per backbone
src/
  train.py                  single training run (full, LoRA, DoRA, AdaLoRA, ...)
  sweep.py                  named experiment plans (grid, rest, prof, extra, paper)
  aggregate_study.py        study tables -> results/study/summary.md
  select_best.py            validation-only selection -> models/best, selection.md
  calibration.py            temperature scaling -> results/calibration.json
  set_policy.py             threshold search -> configs/policy.json
  export_best.py            fp32 HF, fp32 ONNX, INT8 ONNX + latency -> results/final_best/
  canon.py                  anti-obfuscation rules applied before inference
  infer.py                  HateSpeechClassifier (adapter / merged / ONNX)
  serve.py                  FastAPI service
  robustness.py, stress_test.py   perturbation and obfuscation tests
app.py                      Streamlit demo
tests/smoke_test.py         offline pipeline check
results/study/              per-run metrics for the study runs; selection.md
results/final_best/         export metrics and ONNX exports of the selected run
docs/                       MODEL_CARD, DEPLOYMENT, PEFT_STUDY_REPORT, ARCHITECTURE, models/INDEX.md
```

Deprecated: `onnx_model_muril_int8/` (broken per-tensor INT8; see its `DEPRECATED.md`).
The MuRIL full-fine-tune winner from an earlier version of this project was removed from
disk; only its metrics remain.

## Reproduction (GPU, several hours)

```bash
python src/sweep.py --plan grid      # rank x alpha grid (MuRIL, 1 seed)
python src/sweep.py --plan rest      # lr, backbones, DoRA/AdaLoRA, seeds
python src/sweep.py --plan prof      # rank-16 LoRA runs
python src/sweep.py --plan extra     # extra backbones and runs
python src/sweep.py --plan paper     # paper-style configs
python src/aggregate_study.py
python src/select_best.py
python src/make_figures.py
python src/model_reports.py
python src/calibration.py --model_dir models/best --out results/calibration.json
python src/set_policy.py models/best
python src/export_best.py --run xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024 --out results/final_best
```

## Limitations

- Pure Telugu only. The code-mixed CSVs were removed from `data/` and are not used.
  Code-mixed and Romanised Telugu were **not evaluated in this study**.
- One dataset (MACD Telugu split). Three seeds. Selection margin is tiny: see
  [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
- Obfuscation: the guard handles most disguises but letter spacing and character
  repetition still lose recall, and some clean-text stress numbers are very high
  (see the model card). Treat these numbers as unverified until rechecked.
- Latency was measured on the development PC only, not on a low-end laptop.
- Human review is required for the review band and for any moderation decision.
  False positives on political or critical speech have not been measured.

## Modern moderation pipeline (Oct 2026)

Added on top of the study above: spam-free data, akshara input guard, TF-IDF+MuRIL ensemble, conformal allow/review/block policy, label audit, Telugu functional tests, and a **Moderator Console** tab in `app.py` (`streamlit run app.py`). Start with [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) (measured results, limits, run commands, next steps). Details: [policy](docs/POLICY_REPORT.md), [functional tests](docs/FUNCTIONAL_TEST_REPORT.md), [negation fix](docs/NEGATION_FIX_REPORT.md), [label audit](docs/LABEL_AUDIT_REPORT.md), [akshara](docs/AKSHARA_REPORT.md), [despam](docs/DESPAM_REPORT.md), [model shortlist](docs/MODEL_SHORTLIST.md). Functional tests are model-authored and unreviewed by native speakers; the console is not browser-checked.
