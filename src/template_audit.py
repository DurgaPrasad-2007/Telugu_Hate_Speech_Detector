"""Audit of advertisement-like template posts (astrology / love-spell / 'all problems solved' ads).

Markers are surface strings that the lexical analysis found almost only in one class. We report how many
comments contain them per split and label, and how the systems score with and without those comments.
Writes results/study/template_audit.json. Run: python src/template_audit.py
"""
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from data_utils import load_split  # noqa: E402

# 'problems', 'for any problem', 'love spell', 'will be done', 'will be told', 'progeny', 'husband-wife'
MARKERS = ["సమస్యలు", "సమస్యకైనా", "వశీకరణ", "చేయబడును", "చెప్పబడును", "సంతాన", "భార్యాభర్తల"]


def flag(texts):
    return np.array([any(m in t for m in MARKERS) for t in texts])


out = {"markers": MARKERS, "splits": {}}
for sp in ("train", "val", "test"):
    d = load_split("telugu", sp, "normalized")
    f, y = flag(d["text"].tolist()), d["label"].to_numpy()
    uniq = d["text"][f].nunique()
    out["splits"][sp] = {"n": int(len(d)), "flagged": int(f.sum()), "flagged_abusive": int((f & (y == 1)).sum()),
                         "flagged_non_abusive": int((f & (y == 0)).sum()), "flagged_unique_texts": int(uniq),
                         "flagged_share_of_abusive": round(float((f & (y == 1)).sum() / (y == 1).sum()), 3)}

test = load_split("telugu", "test", "normalized")
f, y = flag(test["text"].tolist()), test["label"].to_numpy()
P = os.path.join(ROOT, "results", "study", "preds")
pred = {"lora": np.load(os.path.join(P, "A_xlmr_lora_s2024.npy")), "full": np.load(os.path.join(P, "B_xlmr_full_s2024.npy")),
        "tfidf": np.load(os.path.join(P, "tfidf_best.npy"))}
out["test_scores"] = {}
for k, p in pred.items():
    out["test_scores"][k] = {
        "all_macro_f1": round(float(f1_score(y, p, average="macro")), 4),
        "flagged_accuracy": round(float((p[f] == y[f]).mean()), 4) if f.any() else None,
        "without_flagged_macro_f1": round(float(f1_score(y[~f], p[~f], average="macro")), 4),
    }
json.dump(out, open(os.path.join(ROOT, "results", "study", "template_audit.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
