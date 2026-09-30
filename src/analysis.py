"""
Post-hoc analysis for the SCI paper:
  - Multi-seed mean +/- std tables
  - Paired statistical significance test (paired t-test / Wilcoxon) between two methods
  - Error analysis: categorize misclassified examples
  - SHAP-based token importance for the best model (optional, slower)

Run after run_experiments.py has produced results/*/metrics.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def load_all_metrics(output_root="results"):
    rows = []
    for path in glob.glob(os.path.join(output_root, "*", "metrics.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def seed_summary(df: pd.DataFrame, run_prefix="seedstudy__"):
    """mean +/- std of macro_f1 (and other metrics) across seeds, grouped by method."""
    sub = df[df["run_name"].str.startswith(run_prefix)].copy()
    if sub.empty:
        print("No seed-study runs found.")
        return pd.DataFrame()
    sub["method_key"] = sub["run_name"].str.extract(r"seedstudy__([a-z]+)__")
    summary = sub.groupby("method_key").agg(
        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        n_seeds=("test_macro_f1", "count"),
    ).reset_index()
    summary["macro_f1_fmt"] = summary.apply(
        lambda r: f"{r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}", axis=1)
    return summary


def significance_test(df: pd.DataFrame, method_a: str, method_b: str, run_prefix="seedstudy__",
                       metric="test_macro_f1"):
    """Paired t-test across matching seeds between two methods."""
    sub = df[df["run_name"].str.startswith(run_prefix)].copy()
    sub["method_key"] = sub["run_name"].str.extract(r"seedstudy__([a-z]+)__")
    sub["seed"] = sub["run_name"].str.extract(r"seed(\d+)$").astype(int)

    a = sub[sub["method_key"] == method_a].sort_values("seed")[metric].values
    b = sub[sub["method_key"] == method_b].sort_values("seed")[metric].values

    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        print(f"Cannot compare {method_a} vs {method_b}: mismatched or missing seed runs.")
        return None

    t_stat, p_value = stats.ttest_rel(a, b)
    print(f"{method_a} ({a.mean():.4f}) vs {method_b} ({b.mean():.4f}): "
          f"paired t={t_stat:.3f}, p={p_value:.4f}")
    return {"t_stat": t_stat, "p_value": p_value, "mean_a": a.mean(), "mean_b": b.mean()}


def error_analysis(model_dir, domain="telugu", preprocess="normalized", max_length=256,
                    output_csv="results/error_analysis.csv"):
    """
    Loads a saved model/tokenizer and produces a CSV of misclassified test
    examples with heuristic error categories for manual annotation/discussion.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_utils import load_split

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    test_df = load_split(domain, "test", preprocess)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(test_df), 32):
            batch = test_df["text"].iloc[i:i + 32].tolist()
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length,
                             return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
            batch_confs = probs.max(dim=-1).values.cpu().numpy()
            preds.extend(batch_preds.tolist())
            confs.extend(batch_confs.tolist())

    test_df = test_df.copy()
    test_df["prediction"] = preds
    test_df["confidence"] = confs
    test_df["correct"] = test_df["prediction"] == test_df["label"]

    errors = test_df[~test_df["correct"]].copy()

    def heuristic_category(row):
        text = row["text"]
        if len(text.split()) <= 3:
            return "short/ambiguous"
        if any(ch.isascii() and ch.isalpha() for ch in text) and any(
                'ఀ' <= ch <= '౿' for ch in text):
            return "code-mixed/script-mixed"
        if row["confidence"] < 0.6:
            return "low-confidence/ambiguous"
        return "unclassified (needs manual review)"

    errors["heuristic_category"] = errors.apply(heuristic_category, axis=1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    errors.to_csv(output_csv, index=False)

    print(f"Total test examples: {len(test_df)}")
    print(f"Misclassified: {len(errors)} ({100*len(errors)/len(test_df):.2f}%)")
    print(errors["heuristic_category"].value_counts())
    print(f"\nSaved detailed error list to {output_csv}")
    print("NOTE: heuristic_category is a rough automatic split (short text, "
          "script-mixing, low confidence). For the paper's error-analysis "
          "section, manually re-annotate a sample into linguistic categories "
          "(sarcasm, implicit abuse, slang, misspelling, etc.) as planned.")
    return errors


def build_main_comparison_table(df: pd.DataFrame, domain="telugu", out_path="results/main_comparison_table.csv"):
    cols = ["run_name", "model", "method", "domain", "trainable_params", "trainable_pct",
            "test_accuracy", "test_precision", "test_recall", "test_macro_f1",
            "training_time_sec", "peak_gpu_memory_mb", "avg_inference_latency_ms"]
    sub = df[df["domain"] == domain][cols].sort_values("test_macro_f1", ascending=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(sub.to_string(index=False))
    print(f"\nSaved to {out_path}")
    return sub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--action", required=True,
                    choices=["seed_summary", "significance", "error_analysis", "main_table"])
    p.add_argument("--domain", default="telugu")
    p.add_argument("--method_a", default="full")
    p.add_argument("--method_b", default="qlora")
    p.add_argument("--model_dir", default=None)
    args = p.parse_args()

    df = load_all_metrics()

    if args.action == "seed_summary":
        print(seed_summary(df))
    elif args.action == "significance":
        significance_test(df, args.method_a, args.method_b)
    elif args.action == "error_analysis":
        if not args.model_dir:
            raise SystemExit("--model_dir required for error_analysis")
        error_analysis(args.model_dir, domain=args.domain)
    elif args.action == "main_table":
        build_main_comparison_table(df, domain=args.domain)


if __name__ == "__main__":
    main()
