"""
Latency/throughput benchmark across PyTorch (GPU/CPU) and ONNX (FP32/INT8).
Real measurements only -- whatever backend isn't available/exported is
skipped and reported as such, never faked.

Usage:
  python src/benchmarking.py --model_dir results/<run>/model \
      --base_model_for_adapter <backbone> --domain telugu --n 200
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split  # noqa: E402
from infer import HateSpeechClassifier  # noqa: E402


def bench_predict(predict_fn, texts, warmup=5, reps=3):
    for _ in range(warmup):
        predict_fn(texts[:8])
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        predict_fn(texts)
        times.append((time.perf_counter() - start) * 1000)
    per_sample = [t / len(texts) for t in times]
    return {
        "p50_ms_per_sample": sorted(per_sample)[len(per_sample) // 2],
        "mean_ms_per_sample": sum(per_sample) / len(per_sample),
        "throughput_per_sec": 1000.0 / (sum(per_sample) / len(per_sample)),
        "n": len(texts),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--base_model_for_adapter", default=None)
    p.add_argument("--onnx_dir", default=None)
    p.add_argument("--onnx_int8_dir", default=None)
    p.add_argument("--domain", default="telugu")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="results/benchmark_results.json")
    args = p.parse_args()

    texts = load_split(args.domain, "test", preprocess="normalized")["text"].tolist()[:args.n]
    results = {}

    if torch.cuda.is_available():
        clf_gpu = HateSpeechClassifier(args.model_dir, device="cuda",
                                        base_model_for_adapter=args.base_model_for_adapter)
        results["pytorch_gpu_fp32"] = bench_predict(clf_gpu.predict, texts)
        del clf_gpu
        torch.cuda.empty_cache()
    else:
        results["pytorch_gpu_fp32"] = "NOT RUN (no CUDA device)"

    clf_cpu = HateSpeechClassifier(args.model_dir, device="cpu",
                                    base_model_for_adapter=args.base_model_for_adapter)
    results["pytorch_cpu_fp32"] = bench_predict(clf_cpu.predict, texts)
    del clf_cpu

    for key, path in [("onnx_fp32", args.onnx_dir), ("onnx_int8", args.onnx_int8_dir)]:
        if path and os.path.isdir(path):
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(path)
                ort_model = ORTModelForSequenceClassification.from_pretrained(path)

                def onnx_predict(batch, tok=tok, ort_model=ort_model):
                    enc = tok(batch, truncation=True, padding=True, max_length=96, return_tensors="pt")
                    return ort_model(**enc)

                results[key] = bench_predict(onnx_predict, texts)
            except Exception as e:
                results[key] = f"NOT RUN (error: {e})"
        else:
            results[key] = "NOT RUN (no export at this path)"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
