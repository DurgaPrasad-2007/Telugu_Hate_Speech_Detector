"""
Pick the winning config by VALIDATION macro-F1 (Rule 3), sweep thresholds
on validation to propose an allow/review/block policy, and write
results/final_summary.json. Test metrics are read out (once) only for the
already-selected winner.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split  # noqa: E402
from infer import HateSpeechClassifier  # noqa: E402


def collect_runs(output_root="results"):
    rows = []
    for path in glob.glob(os.path.join(output_root, "*", "metrics.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["_path"] = os.path.dirname(path)
        rows.append(d)
    return pd.DataFrame(rows)


def threshold_sweep(probs, labels, steps=19):
    labels = np.asarray(labels)
    rows = []
    for low in np.linspace(0.05, 0.5, steps):
        # block only at/above 0.5 so a "block" never contradicts the argmax label
        for high in np.linspace(max(0.5, low + 0.05), 0.95, steps):
            pred_block = probs >= high
            pred_allow = probs < low
            pred_review = ~pred_block & ~pred_allow
            # precision/recall of the "block" decision vs true abusive label
            tp = int((pred_block & (labels == 1)).sum())
            fp = int((pred_block & (labels == 0)).sum())
            fn_ = int((~pred_block & (labels == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn_) if (tp + fn_) else 0.0
            rows.append({"low": round(float(low), 3), "high": round(float(high), 3),
                         "block_precision": precision, "block_recall": recall,
                         "review_pct": float(pred_review.mean())})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_for_adapter", default=None,
                    help="Set only if the winning run is a PEFT adapter, not a full model")
    p.add_argument("--domain", default="telugu")
    p.add_argument("--min_block_precision", type=float, default=0.9)
    args = p.parse_args()

    df = collect_runs()
    if df.empty:
        print("No results found; run training first.")
        return
    df.to_csv("results/all_results.csv", index=False)

    # Selection: validation macro-F1 only (Rule 3). Runs predating the
    # val_macro_f1 fix (see train.py) are excluded from selection, not
    # silently treated as 0 -- they're re-rankable once re-run.
    ranked = df.dropna(subset=["val_macro_f1"]).sort_values("val_macro_f1", ascending=False)
    if ranked.empty:
        print("No runs have val_macro_f1 recorded; re-run with the updated train.py.")
        return
    winner = ranked.iloc[0]
    print(f"Winner by validation macro-F1: {winner['run_name']} (val_macro_f1={winner['val_macro_f1']:.4f}, "
          f"test_macro_f1={winner['test_macro_f1']:.4f})")

    model_dir = os.path.join(winner["_path"], "model")
    clf = HateSpeechClassifier(model_dir, base_model_for_adapter=args.base_model_for_adapter)
    val_df = load_split(args.domain, "val", preprocess="normalized")
    preds = clf.predict(val_df["text"].tolist())
    probs = np.array([r["probs"]["Abusive"] for r in preds])
    labels = val_df["label"].tolist()

    sweep = threshold_sweep(probs, labels)
    sweep.to_csv("results/threshold_sweep.csv", index=False)
    candidates = sweep[sweep["block_precision"] >= args.min_block_precision]
    if candidates.empty:
        candidates = sweep.sort_values("block_precision", ascending=False)
    best = candidates.sort_values(["block_recall", "review_pct"], ascending=[False, True]).iloc[0]
    policy = {"low": float(best["low"]), "high": float(best["high"])}
    with open("configs/policy.json", "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    print(f"Policy thresholds: {policy} -> block_precision={best['block_precision']:.3f}, "
          f"block_recall={best['block_recall']:.3f}, review_pct={best['review_pct']:.3f}")

    calib = {}
    if os.path.exists("results/calibration.json"):
        with open("results/calibration.json", encoding="utf-8") as f:
            calib = json.load(f)

    summary = {
        "best_model": winner["model"],
        "best_configuration": f"{winner['method']} on {winner['domain']}",
        "run_name": winner["run_name"],
        "validation_macro_f1": winner["val_macro_f1"],
        "test_macro_f1": winner["test_macro_f1"],
        "precision": winner["test_precision"],
        "recall": winner["test_recall"],
        "ece": calib.get("after", {}).get("ece"),
        "model_size_mb": None,  # filled by benchmarking.py output if available
        "policy_thresholds": policy,
    }
    with open("results/final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
