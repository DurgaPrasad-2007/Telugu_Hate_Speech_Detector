"""Clean-test macro-F1 (threshold 0.5) baseline vs augmented models + paired bootstrap of the difference,
and functional pass rates before/after from the scores CSVs written by functional_eval (THS_SCORES_OUT).
Label convention: internal 1=abusive. Rows aligned by text. Writes docs/NEGATION_FIX_REPORT.md numbers to stdout.
usage: THS_DATA_DIR=data/clean python src/negation_compare.py <muril_base_preds> <muril_aug_preds> <scoresA.csv> <scoresAB.csv>"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from conformal_policy import split_val
from significance import macro_f1_counts
from tfidf_classifier import guard
import joblib
N_BOOT = 10000


def scored(pipe_path, preds):
    df = pd.read_csv(preds)
    pipe = joblib.load(pipe_path)
    df["d"] = pipe.decision_function([guard(t)[0] for t in df.text])
    A, _ = split_val(df)
    pl = LogisticRegression().fit(A[["d"]].values, A.label.values)
    df["p_tfidf"] = pl.predict_proba(df[["d"]].values)[:, 1]
    df["p_ens"] = (df.p_tfidf + df.p_abusive) / 2
    return df[df.split == "test"].set_index("text")


def main(mb, ma, fa, fab):
    P = lambda *a: os.path.join(ROOT, *a)
    base = scored(P("models", "tfidf", "tfidf_best.joblib"), mb)
    ctrl = scored(P("results", "aug_tfidf", "clean_retrain.joblib"), mb)
    aug = scored(P("results", "aug_tfidf", "aug.joblib"), ma)
    common = base.index.intersection(aug.index)
    assert len(common) == len(base) == len(aug), (len(common), len(base), len(aug))
    base, ctrl, aug = base.loc[common], ctrl.loc[common], aug.loc[common]
    y = base.label.values; n = len(y)
    assert (aug.label.values == y).all()
    idx = np.random.default_rng(0).integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    def boot(p):
        return np.concatenate([macro_f1_counts(y[idx[s:s + 500]], p[idx[s:s + 500]]) for s in range(0, N_BOOT, 500)])
    pr = lambda d, c: (d[c].values >= 0.5).astype(int)
    rows = [("TF-IDF", base, "p_tfidf", aug, "p_tfidf"), ("TF-IDF clean retrain (control)", base, "p_tfidf", ctrl, "p_tfidf"),
            ("MuRIL", base, "p_abusive", aug, "p_abusive"), ("ensemble", base, "p_ens", aug, "p_ens")]
    print(f"clean test n={n}; threshold 0.5; paired bootstrap {N_BOOT} resamples seed 0\n")
    print("| model | baseline F1 | candidate F1 | delta | 95% CI of delta |\n|---|---|---|---|---|")
    for name, b, bc, a, ac in rows:
        pb, pa = pr(b, bc), pr(a, ac)
        d = boot(pa) - boot(pb); lo, hi = np.percentile(d, [2.5, 97.5])
        fb, fa_ = f1_score(y, pb, average="macro"), f1_score(y, pa, average="macro")
        print(f"| {name} | {fb:.4f} | {fa_:.4f} | {fa_ - fb:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    print()
    A, B = pd.read_csv(fa), pd.read_csv(fab)
    for c in ("p_tfidf", "p_ens"):
        for d in (A, B):
            d["ok_" + c] = (d[c] >= 0.5) == (d.expected == "abusive")
    def tab(group_hi):
        out = []
        for c in ("p_tfidf", "p_ens"):
            for nm, d in (("A", A), ("AB", B)):
                g = d[d.confidence == group_hi].groupby("functionality")["ok_" + c].agg(["size", "mean"])
                out.append(g["mean"].rename(f"{c}_{nm}"))
        t = pd.concat(out, axis=1); t.insert(0, "n", A[A.confidence == group_hi].groupby("functionality").size())
        return t
    for conf in ("high", "low"):
        print(f"functional pass rate, {conf}-confidence (A = new akshara only, AB = + negation aug TF-IDF/MuRIL)")
        print(tab(conf).round(3).to_string()); print()
        h = [d[d.confidence == conf] for d in (A, B)]
        print("overall", {f"{c}_{nm}": round(float(d['ok_' + c].mean()), 4) for c in ("p_tfidf", "p_ens") for nm, d in zip(("A", "AB"), h)}, "\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
