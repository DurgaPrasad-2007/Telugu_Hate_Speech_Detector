# Progress snapshot — COMPLETE, demo-ready

See `docs/FINAL_REPORT.md` for the full write-up with real numbers.

## Done, verified
- Label semantics resolved by hash-matching against upstream
  `ShareChatAI/MACD` (`src/data_utils.py` `RAW_LABEL_MAP`,
  `docs/RESEARCH_REVIEW.md`). Train->val/test leakage filtered at load
  time (no CSV edits).
- Backbone comparison (full fine-tune, telugu): MuRIL 0.9045 test
  macro-F1 (val backfilled 0.9029), XLM-R 0.9042 test / 0.9010 val macro-F1.
- PEFT sweep on MuRIL, all 5 methods run: LoRA (0.8676 test),
  QLoRA (0.8701 test), DoRA (0.8628 test), IA3 (0.7261 test),
  AdaLoRA (0.3425 test, collapsed — reported as-is). All underperformed
  full fine-tune.
- **Winner** (`src/finalize.py`, picked by validation macro-F1):
  `baseline__muril__full__telugu` — val 0.9029 / test 0.9045 macro-F1 (a tie with XLM-R;
  MuRIL's val score was backfilled, see docs/FINAL_REPORT.md section 4). XLM-R artifacts: `results/xlmr_reference/`.
  `configs/policy.json` written (low=0.45, high=0.50).
- Calibration done (MuRIL): T=1.216, ECE 0.0374->0.0141 (`results/calibration.json`).
- Robustness done: fragile to char-repeat/extra-spacing (~15% flip rate),
  robust to case/punctuation/emoji/digit obfuscation
  (`results/robustness_results_summary.json`, `docs/ROBUSTNESS.md`).
- Error analysis done: 9.51% test error rate, 60 hand-read, 4 real
  failure modes documented (`results/error_analysis.csv`,
  `docs/ERROR_ANALYSIS.md`).
- ONNX export + benchmark done: int8 CPU inference 7.6ms/sample (132/s),
  ~3.2x speedup over CPU fp32 (`onnx_model/`, `onnx_model_int8/`,
  `results/benchmark_results.json`).
- `src/serve.py` (FastAPI) and `app.py` (Streamlit) tested live against
  the winning model: /health, /readyz, /version, /predict (including
  empty/whitespace/oversized-batch edge cases) all verified; Streamlit
  starts and serves without error.
- Smoke test (`tests/smoke_test.py`) passes end-to-end.
- All docs filled with real measured numbers: `docs/MODEL_CARD.md`,
  `docs/DEPLOYMENT.md`, `docs/ERROR_ANALYSIS.md`, `docs/ROBUSTNESS.md`,
  `docs/FINAL_REPORT.md`.

## Scope cut (documented, not hidden)
- Only 2 backbones compared (MuRIL, XLM-R), not the full 4-5 in the
  original spec (mBERT/IndicBERT skipped) — time budget.
- Multi-seed study and cross-domain transfer (telugu<->codemixed) not
  run this session.
- `src/run_experiments.py`'s full 5x5x5 grid was never run wholesale;
  the hand-driven reduced matrix above is what's real — say so in any
  presentation rather than claim the full matrix ran.

## Live logging (for the demo)
`streamlit run app.py` prints a step-by-step trace to the terminal for every
Analyze click: input (masked), anti-obfuscation fixes, tokenization, logits,
raw vs calibrated probability, policy decision, per-word occlusion drops and
which words were hidden. INFO logs show masked text only; `LOG_RAW=1` also
prints raw input, `LOG_LEVEL=WARNING` silences it.
Stress test: `python src/stress_test.py` (real model, real test data).

## How to demo
```bash
# CLI inference
python src/infer.py --model_dir results/baseline__muril__full__telugu/model --text "..."

# API
MODEL_DIR=results/baseline__muril__full__telugu/model uvicorn src.serve:app --port 8000
curl localhost:8000/readyz
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{"texts":["..."]}'

# UI (Predict / Batch CSV / Model Info / Metrics Dashboard / Error Explorer / Robustness Playground)
MODEL_DIR=results/baseline__muril__full__telugu/model streamlit run app.py
```
Walk the professor through `docs/FINAL_REPORT.md` for the numbers and
`docs/RESEARCH_REVIEW.md` for the label-semantics verification work —
that's the most defensible/rigorous part of the project.
