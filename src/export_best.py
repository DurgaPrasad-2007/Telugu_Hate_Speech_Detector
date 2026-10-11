"""
Export one study run to a merged HF model, int8 ONNX, and CPU metrics.

Usage:
  python src/export_best.py --run lora_r8_a64_lr2e-4_e10_s42 --out exports/best

Writes:
  <out>/hf/              merged fp32 HF model + tokenizer
  <out>/onnx/            fp32 ONNX (via src/export_onnx.py)
  <out>/onnx_int8/       int8 ONNX (model_quantized.onnx)
  <out>/export_metrics.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only: GPU is busy with the training queue

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from data_utils import load_split  # same text loading/normalisation as load_dataset_dict

MAX_LEN = 96
WARMUP, BENCH_N = 10, 200
THREADS = [1, 4]


def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro"))}


def torch_predict(model, tok, texts, bs=32):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            out.extend(model(**enc).logits.argmax(-1).tolist())
    return np.array(out)


def onnx_predict(sess, tok, texts, bs=32):
    names = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
        out.extend(sess.run(None, feed)[0].argmax(-1).tolist())
    return np.array(out)


def bench(run_one, texts):
    """Per-comment latency, batch size 1, tokenisation included. Times one call per comment."""
    for t in texts[:WARMUP]:
        run_one([t])
    ms = []
    for t in texts[:BENCH_N]:
        t0 = time.perf_counter()
        run_one([t])
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return {"n": len(ms), "mean_ms": statistics.mean(ms), "median_ms": statistics.median(ms),
            "p90_ms": ms[int(0.9 * len(ms)) - 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under results/study/")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    run_dir = os.path.join(ROOT, "results", "study", args.run, "model")
    out = os.path.abspath(args.out)
    hf_dir = os.path.join(out, "hf")
    os.makedirs(hf_dir, exist_ok=True)

    # 1. Merge adapter into its base model (full-finetune runs have no adapter).
    if os.path.exists(os.path.join(run_dir, "adapter_config.json")):
        from peft import PeftConfig, PeftModel
        base_name = PeftConfig.from_pretrained(run_dir).base_model_name_or_path
        base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
        model = PeftModel.from_pretrained(base, run_dir).merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(run_dir)
    model.eval()
    model.save_pretrained(hf_dir)
    tok = AutoTokenizer.from_pretrained(run_dir)
    tok.save_pretrained(hf_dir)

    # 2. ONNX export + int8 quantization: reuse the existing script.
    subprocess.run([sys.executable, os.path.join(HERE, "export_onnx.py"), "--model_dir", hf_dir,
                    "--out_dir", os.path.join(out, "onnx"), "--quantize"], check=True)
    int8_dir = os.path.join(out, "onnx_int8")
    onnx_path = os.path.join(int8_dir, "model_quantized.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"expected quantized model at {onnx_path}")

    # 3. Accuracy / macro-F1 / agreement on val and test.
    splits = {s: load_split("telugu", s, "normalized") for s in ["val", "test"]}
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    res = {"int8_onnx": {}, "fp32_torch": {}, "agreement_int8_vs_fp32": {}}
    for s, df in splits.items():
        texts, y = df["text"].tolist(), df["label"].to_numpy()
        p_i8 = onnx_predict(sess, tok, texts)
        p_fp = torch_predict(model, tok, texts)
        res["int8_onnx"][s] = metrics(y, p_i8)
        res["fp32_torch"][s] = metrics(y, p_fp)
        res["agreement_int8_vs_fp32"][s] = float(np.mean(p_i8 == p_fp))
        res.setdefault("n", {})[s] = len(texts)

    # 4. CPU latency, batch size 1, first 200 test comments after 10 warmup.
    test_texts = splits["test"]["text"].tolist()
    res["latency_ms"] = {}
    for n in THREADS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = n
        s_ort = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        names = {i.name for i in s_ort.get_inputs()}

        def run_ort(batch, s_ort=s_ort, names=names):
            enc = tok(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
            s_ort.run(None, {k: v.astype(np.int64) for k, v in enc.items() if k in names})

        torch.set_num_threads(n)

        def run_torch(batch):
            enc = tok(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            with torch.inference_mode():
                model(**enc)

        res["latency_ms"][f"int8_onnx_threads{n}"] = bench(run_ort, test_texts)
        res["latency_ms"][f"fp32_torch_threads{n}"] = bench(run_torch, test_texts)

    res["run"] = args.run
    res["max_length"] = MAX_LEN
    res["preprocess"] = "normalized"
    res["int8_onnx_size_mb"] = os.path.getsize(onnx_path) / 1e6

    with open(os.path.join(out, "export_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.join(out, 'export_metrics.json')}")


if __name__ == "__main__":
    main()
