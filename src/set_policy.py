"""Fit allow/review/block thresholds for models/best on VALIDATION only (reuses finalize.threshold_sweep)."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split
from infer import HateSpeechClassifier
from finalize import threshold_sweep

MIN_BLOCK_PRECISION = 0.9
clf = HateSpeechClassifier(sys.argv[1] if len(sys.argv) > 1 else "models/best")
val = load_split("telugu", "val", preprocess="normalized")
probs = np.array([r["probs"]["Abusive"] for r in clf.predict(val["text"].tolist())])
sweep = threshold_sweep(probs, val["label"].tolist())
sweep.to_csv("results/threshold_sweep.csv", index=False)
ok = sweep[sweep["block_precision"] >= MIN_BLOCK_PRECISION]
best = (ok if len(ok) else sweep.sort_values("block_precision", ascending=False)).sort_values(
    ["block_recall", "review_pct"], ascending=[False, True]).iloc[0]
json.dump({"low": float(best["low"]), "high": float(best["high"])}, open("configs/policy.json", "w"), indent=2)
print("policy", dict(best))
