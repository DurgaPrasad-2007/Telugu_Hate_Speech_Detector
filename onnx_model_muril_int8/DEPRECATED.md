# DEPRECATED - do not use

This folder holds a MuRIL dynamic INT8 ONNX model that was exported with
**per-tensor** quantization. That recipe breaks the model:

- On a 600-comment test sample it scores about **0.37 macro-F1**.
- It predicts "abusive" for only about **3%** of comments.

It is kept only so the history stays intact. Do not deploy it or recommend it.

Use the selected model instead:

- Shipped PyTorch model: `models/best/` (XLM-R base + LoRA adapter).
- INT8 ONNX of the same model (dynamic, per-channel), produced by
  `src/export_best.py`: `results/final_best/onnx_int8/`. Validation macro-F1
  0.8995, test 0.8926 (see `results/final_best/export_metrics.json`).

Details: `docs/PEFT_STUDY_REPORT.md` section 6, and `docs/DEPLOYMENT.md`.
