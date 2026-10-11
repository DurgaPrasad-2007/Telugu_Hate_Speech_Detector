"""Classical floor: TF-IDF (char_wb 2-5 + word 1-2) -> LogisticRegression / LinearSVC.

C is chosen per model family by VALIDATION macro-F1 only. Test is touched once,
for the final report and the saved test predictions.
Run: .venv\Scripts\python -I src\baselines_classical.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import time
import warnings
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_utils import load_split  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
STUDY = os.path.join(ROOT, "results", "study")
PREDS = os.path.join(STUDY, "preds")
MODEL_DIR = os.path.join(STUDY, "tfidf")
os.makedirs(PREDS, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

C_GRID = [0.1, 0.3, 1, 3, 10]
warnings.filterwarnings("ignore", category=UserWarning)


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, p)),
        "abusive_precision": float(pr[0]),
        "abusive_recall": float(rc[0]),
        "abusive_f1": float(f1[0]),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "weighted_f1": float(f1_score(y, p, average="weighted")),
        "confusion_matrix_[[tn,fp],[fn,tp]]": cm.tolist(),
    }


def main():
    t0 = time.time()
    tr = load_split("telugu", "train", "normalized")
    va = load_split("telugu", "val", "normalized")
    te = load_split("telugu", "test", "normalized")
    print(f"train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    print("NOTE: word analyzer uses whitespace tokens (\S+); the default \w+ pattern "
          "splits Telugu words at vowel signs (Mn/Mc marks are not \w).", flush=True)

    union = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 max_features=300000, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2),
                                 max_features=300000, sublinear_tf=True)),
    ])
    Xtr = union.fit_transform(tr["text"])
    Xva = union.transform(va["text"])
    Xte = union.transform(te["text"])
    print(f"features={Xtr.shape[1]} fit+transform {time.time()-t0:.1f}s", flush=True)

    ytr, yva, yte = tr["label"].values, va["label"].values, te["label"].values
    candidates = []
    for fam, mk in [
        ("logreg", lambda C: LogisticRegression(C=C, solver="liblinear", max_iter=2000)),
        ("linearsvc", lambda C: LinearSVC(C=C, max_iter=10000)),
    ]:
        for C in C_GRID:
            clf = mk(C).fit(Xtr, ytr)
            pv = clf.predict(Xva)
            vf1 = f1_score(yva, pv, average="macro")
            candidates.append({"family": fam, "C": C, "val_macro_f1": float(vf1), "clf": clf})
            print(f"  {fam:10s} C={C:<4} val macro-F1={vf1:.4f}", flush=True)

    # choose C per family by validation macro-F1; then the better family by validation
    best_per_family = {}
    for fam in ["logreg", "linearsvc"]:
        cands = [c for c in candidates if c["family"] == fam]
        best_per_family[fam] = max(cands, key=lambda c: c["val_macro_f1"])
    best = max(best_per_family.values(), key=lambda c: c["val_macro_f1"])
    print(f"selected: {best['family']} C={best['C']} (val macro-F1 {best['val_macro_f1']:.4f})", flush=True)

    clf = best["clf"]
    pipe = Pipeline([("features", union), ("clf", clf)])
    model_path = os.path.join(MODEL_DIR, "tfidf_best.joblib")
    joblib.dump(pipe, model_path)
    size_mb = os.path.getsize(model_path) / 1e6

    pv = clf.predict(Xva)
    pt = clf.predict(Xte)
    np.save(os.path.join(PREDS, "tfidf_best.npy"), pt.astype(np.int64))

    # CPU latency: single thread, batch 1, first 200 test rows, end-to-end (featurize + predict)
    texts = te["text"].tolist()[:200]
    for t in texts[:10]:
        pipe.predict([t])
    lat = []
    for t in texts:
        s = time.perf_counter()
        pipe.predict([t])
        lat.append((time.perf_counter() - s) * 1000)
    lat = np.array(lat)

    out = {
        "features": {"char_wb": "2-5, max_features 300000, sublinear_tf",
                     "word": "1-2, token_pattern \S+, max_features 300000, sublinear_tf",
                     "n_features_total": int(Xtr.shape[1])},
        "c_grid": C_GRID,
        "val_grid": [{"family": c["family"], "C": c["C"], "val_macro_f1": c["val_macro_f1"]}
                     for c in candidates],
        "selected": {"family": best["family"], "C": best["C"]},
        "val": metrics(yva, pv),
        "test": metrics(yte, pt),
        "model_size_mb_joblib": size_mb,
        "model_path": os.path.relpath(model_path, ROOT),
        "cpu_latency_ms_batch1_200rows_1thread": {
            "median": float(np.median(lat)), "mean": float(lat.mean()),
            "p90": float(np.percentile(lat, 90))},
        "n": {"train": len(tr), "val": len(va), "test": len(te)},
        "test_preds": "results/study/preds/tfidf_best.npy",
    }
    with open(os.path.join(STUDY, "baseline_classical.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"VAL macro-F1={out['val']['macro_f1']:.4f} acc={out['val']['accuracy']:.4f} | "
          f"TEST macro-F1={out['test']['macro_f1']:.4f} acc={out['test']['accuracy']:.4f} | "
          f"size={size_mb:.1f}MB | latency median={np.median(lat):.2f}ms | total {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
