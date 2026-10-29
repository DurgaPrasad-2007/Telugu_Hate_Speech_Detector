"""char n-gram + akshara n-gram TF-IDF variant, evaluated vs the deployed TF-IDF baseline.
Run: .venv\Scripts\python -I src\tfidf_akshara.py"""
import os, random, re, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import akshara  # noqa: E402
from tfidf_classifier import DEFAULT_MODEL, guard as canon_guard, joblib, load_split  # noqa: E402
from canon import canonicalize  # noqa: E402
from translit import translit_text  # noqa: E402
from data_utils import normalize_text  # noqa: E402


def guard_ak(text):
    return normalize_text(translit_text(akshara.normalize(text)), "normalized")


def build():
    return Pipeline([("features", FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=300000, sublinear_tf=True)),
        ("aks", TfidfVectorizer(analyzer=akshara.ngrams, max_features=300000, sublinear_tf=True))])),
        ("clf", LinearSVC(C=0.3, max_iter=10000))])


# attacks applied to EVERY word (as in src/stress_test.py); words of >=3 letters get disguised
def _atk(sep):
    def f(t):
        return " ".join(sep.join(akshara.segment(w)) if sep != " " else " ".join(akshara.segment(w))
                        for w in t.split())
    return f

ATTACKS = {"spaced": _atk(" "), "dash": _atk("-"), "dot": _atk("."), "star": _atk("*")}


def main():
    tr, te = load_split("telugu", "train", "raw"), load_split("telugu", "test", "raw")
    base = joblib.load(DEFAULT_MODEL)
    var = build().fit([guard_ak(t) for t in tr["text"]], tr["label"])
    models = {"baseline (canon guard)": (base, lambda t: canon_guard(t)[0]),
              "char+akshara (akshara guard)": (var, guard_ak)}
    X, y = te["text"].tolist(), te["label"].values
    rows = []
    for name, (m, g) in models.items():
        pred = m.predict([g(t) for t in X])
        r = {"model": name, "macro_f1": f1_score(y, pred, average="macro"),
             "clean_fp": float(((pred == 1) & (y == 0)).sum() / (y == 0).sum())}
        ab = [t for t, p, l in zip(X, pred, y) if l == 1 and p == 1]
        cl = [t for t, l in zip(X, y) if l == 0]
        for a, f in ATTACKS.items():
            r[f"caught_{a}"] = float(np.mean(m.predict([g(f(t)) for t in ab]) == 1))
            r[f"cleanfp_{a}"] = float(np.mean(m.predict([g(f(t)) for t in cl]) == 1))
        r["guard_changes_clean_pct"] = float(np.mean([(akshara.normalize(t) != t if "akshara" in name else canonicalize(t)[0] != t) for t in cl]))
        r["n_abusive_caught_clean"] = len(ab)
        rows.append(r)
    for r in rows:
        print(r)
    return rows


if __name__ == "__main__":
    main()
