"""Extra analyses for the paper, computed only from saved artifacts (no training).

Outputs results/study/paper_extras.json:
  - slice metrics on the test set (length, emoji, Latin script) for selected LoRA, full FT and TF-IDF
  - error overlap between systems (candidate label noise = all strong systems disagree with the label)
  - confidence profile of the selected model's errors
  - calibration reliability bins, threshold trade-off, per-attack robustness
Run: python src/paper_extras.py
"""
import json, os, re, sys
import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from data_utils import load_split  # noqa: E402

P = os.path.join(ROOT, "results", "study", "preds")
test = load_split("telugu", "test", "normalized")
texts, y = test["text"].tolist(), test["label"].to_numpy()
pred = {
    "lora": np.load(os.path.join(P, "A_xlmr_lora_s2024.npy")),
    "full": np.load(os.path.join(P, "B_xlmr_full_s2024.npy")),
    "dora": np.load(os.path.join(P, "C_xlmr_dora_s2024.npy")),
    "muril_lora": np.load(os.path.join(P, "E_muril_lora_s2024.npy")),
    "tfidf": np.load(os.path.join(P, "tfidf_best.npy")),
}
for k, v in pred.items():
    assert len(v) == len(y), (k, len(v), len(y))
out = {"n_test": int(len(y)),
       "check_macro_f1": {k: round(float(f1_score(y, v, average="macro")), 4) for k, v in pred.items()}}

# ---- slices
EMO = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿]")
LAT = re.compile("[A-Za-z]")
words = np.array([len(t.split()) for t in texts])
slices = {
    "1-5 words": words <= 5, "6-10 words": (words > 5) & (words <= 10),
    "11-20 words": (words > 10) & (words <= 20), "over 20 words": words > 20,
    "contains emoji": np.array([bool(EMO.search(t)) for t in texts]),
    "contains Latin letters": np.array([bool(LAT.search(t)) for t in texts]),
    "Telugu script only": np.array([not LAT.search(t) for t in texts]),
}
out["slices"] = {}
for name, m in slices.items():
    row = {"n": int(m.sum()), "abusive_share": round(float(y[m].mean()), 3)}
    for k in ("lora", "full", "tfidf"):
        row[k] = round(float(f1_score(y[m], pred[k][m], average="macro")), 4)
    out["slices"][name] = row

# ---- error overlap
err = {k: v != y for k, v in pred.items()}
strong = ["lora", "full", "dora", "muril_lora", "tfidf"]
all_wrong = np.logical_and.reduce([err[k] for k in strong])
out["errors"] = {k: int(e.sum()) for k, e in err.items()}
out["all_five_wrong"] = int(all_wrong.sum())
out["all_five_wrong_share_of_lora_errors"] = round(float(all_wrong.sum() / err["lora"].sum()), 3)
out["all_five_wrong_by_gold"] = {"gold_abusive": int((all_wrong & (y == 1)).sum()), "gold_non_abusive": int((all_wrong & (y == 0)).sum())}
inter = (err["lora"] & err["tfidf"]).sum()
out["lora_tfidf_error_jaccard"] = round(float(inter / (err["lora"] | err["tfidf"]).sum()), 3)
out["lora_full_error_jaccard"] = round(float((err["lora"] & err["full"]).sum() / (err["lora"] | err["full"]).sum()), 3)
out["oracle_any_correct_acc"] = round(float(1 - all_wrong.mean()), 4)

# ---- confidence of the selected model's errors (calibrated, from analysis.py output)
ea = os.path.join(ROOT, "results", "error_analysis.csv")
if os.path.exists(ea):
    import pandas as pd
    d = pd.read_csv(ea)
    wrong = d[~d["correct"]]
    out["error_confidence"] = {
        "n_errors": int(len(wrong)),
        "conf_ge_0.9": int((wrong["confidence"] >= 0.9).sum()),
        "conf_lt_0.6": int((wrong["confidence"] < 0.6).sum()),
        "false_negatives": int(((wrong["label"] == 1) & (wrong["prediction"] == 0)).sum()),
        "false_positives": int(((wrong["label"] == 0) & (wrong["prediction"] == 1)).sum()),
    }

# ---- calibration bins, thresholds, per-attack robustness
cal = json.load(open(os.path.join(ROOT, "results", "calibration.json"), encoding="utf-8"))
out["calibration"] = {"temperature": cal["temperature"],
                      "before": cal["before"]["reliability_bins"], "after": cal["after"]["reliability_bins"]}
import pandas as pd
sw = pd.read_csv(os.path.join(ROOT, "results", "threshold_sweep.csv"))
pol = json.load(open(os.path.join(ROOT, "configs", "policy.json")))
lo = sw.loc[(sw["low"] - pol["low"]).abs().idxmin(), "low"]
curve = sw[sw["low"] == lo].sort_values("high")
out["threshold_curve"] = {"low": float(lo), "rows": curve[["high", "block_precision", "block_recall", "review_pct"]].round(4).to_dict("records")}
rb = json.load(open(os.path.join(ROOT, "results", "study", "robustness_compare.json"), encoding="utf-8"))["results"]
out["per_attack"] = {a: {m: round(rb[m][a]["acc"], 3) for m in rb} for a in rb[next(iter(rb))]}

json.dump(out, open(os.path.join(ROOT, "results", "study", "paper_extras.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ("calibration", "threshold_curve", "per_attack")}, indent=1, ensure_ascii=False))
print("calibration bins (after):", out["calibration"]["after"][:3], "...")
print("threshold curve rows:", len(out["threshold_curve"]["rows"]), out["threshold_curve"]["rows"][:2])
print("attacks:", list(out["per_attack"])[:3], "... models:", list(next(iter(out["per_attack"].values())).keys()))
