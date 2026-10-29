"""Remove commercial astrology / vashikaran spam templates from data/telugu_{train,val,test}.csv.

A comment is spam if it has a marker string (same list as template_audit.py) or is a near-duplicate
(char TF-IDF cosine >= NEAR) of a marker-flagged comment from any split.
Writes data/clean/telugu_{split}.csv and data/clean/spam_flagged.csv. Run: python src/despam.py
"""
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA, CLEAN = os.path.join(ROOT, "data"), os.path.join(ROOT, "data", "clean")
SPLITS = ("train", "val", "test")
# 'problems', 'for any problem', 'love spell', 'will be done', 'will be told', 'progeny', 'husband-wife', 'horoscope'
MARKERS = ["సమస్యలు", "సమస్యకైనా", "వశీకరణ", "చేయబడును", "చెప్పబడును", "సంతాన", "భార్యాభర్తల", "జాతకం"]
NEAR = 0.8


def marker_flag(texts):
    return [any(m in t for m in MARKERS) for t in texts]


def flag_all(frames):
    """frames: {split: df}. Returns {split: bool list}; marker hit or near-duplicate of a marker hit."""
    allt = pd.concat([frames[s]["text"] for s in SPLITS], ignore_index=True).astype(str)
    hit = pd.Series(marker_flag(allt.tolist()))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform(allt)
    seeds = X[hit.values]
    # ponytail: dense-free but O(n*seeds) sparse product; fine at ~10^4 rows
    near = ((X @ seeds.T).max(axis=1).toarray().ravel() >= NEAR) if seeds.shape[0] else hit.values
    flag, out, i = (hit.values | near), {}, 0
    for s in SPLITS:
        out[s] = flag[i:i + len(frames[s])].tolist()
        i += len(frames[s])
    return out, hit.values.sum(), flag.sum()


def main():
    frames = {s: pd.read_csv(os.path.join(DATA, f"telugu_{s}.csv")) for s in SPLITS}
    flags, n_marker, n_all = flag_all(frames)
    os.makedirs(CLEAN, exist_ok=True)
    spam = []
    print("label 0 = abusive, 1 = non-abusive (data convention)")
    print(f"{'split':6s} {'n':>6s} {'spam':>6s} {'spam_lbl0':>12s} {'spam_lbl1':>9s} {'kept':>6s}")
    for s in SPLITS:
        d, f = frames[s], pd.Series(flags[s], index=frames[s].index)
        d[~f].to_csv(os.path.join(CLEAN, f"telugu_{s}.csv"), index=False)
        spam.append(d[f].assign(split=s))
        y = d["label"]
        print(f"{s:6s} {len(d):6d} {f.sum():6d} {(f & (y == 0)).sum():12d} {(f & (y == 1)).sum():9d} {(~f).sum():6d}")
    pd.concat(spam).to_csv(os.path.join(CLEAN, "spam_flagged.csv"), index=False)
    print(f"marker hits {n_marker}, + near-duplicates {n_all - n_marker}")


def evaluate():
    """TF-IDF + LinearSVC (features and C grid from baselines_classical.py), C picked on val macro-F1."""
    sys.path.insert(0, HERE)
    from baselines_classical import C_GRID, metrics
    from sklearn.feature_extraction.text import TfidfVectorizer as T
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC
    from data_utils import normalize_text
    rd = lambda base, s: pd.read_csv(os.path.join(base, f"telugu_{s}.csv")).assign(
        text=lambda d: d["text"].astype(str).map(lambda t: normalize_text(t, "normalized")))
    D = {k: {s: rd(b, s) for s in SPLITS} for k, b in (("raw", DATA), ("clean", CLEAN))}
    for tr in ("raw", "clean"):
        u = FeatureUnion([("c", T(analyzer="char_wb", ngram_range=(2, 5), max_features=300000, sublinear_tf=True)),
                          ("w", T(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_features=300000, sublinear_tf=True))])
        X = {s: (u.fit_transform if s == "train" else u.transform)(D[tr][s]["text"]) for s in SPLITS}
        fit = lambda C: LinearSVC(C=C, max_iter=10000).fit(X["train"], D[tr]["train"]["label"])
        best = max(C_GRID, key=lambda C: metrics(D[tr]["val"]["label"], fit(C).predict(X["val"]))["macro_f1"])
        clf = fit(best)
        for te in ("raw", "clean"):
            Xt = X["test"] if te == tr else u.transform(D[te]["test"]["text"])
            m = metrics(D[te]["test"]["label"], clf.predict(Xt))
            print(f"train={tr:5s} C={best} test={te:5s} n={len(D[te]['test'])} macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    evaluate() if "--eval" in sys.argv else main()
