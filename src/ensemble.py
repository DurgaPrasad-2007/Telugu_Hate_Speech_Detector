"""Average calibrated TF-IDF + transformer P(abusive); compare on TEST vs each single model and run the
conformal policy on all three; writes docs/POLICY_REPORT.md (measured numbers only).
Label convention: internal 1=abusive. Rows are keyed by the text inside preds.csv (no positional merge).
usage: THS_DATA_DIR=data/clean python src/ensemble.py results/<run>/preds.csv"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conformal_policy import tfidf_scored, run_policy, split_val
from significance import macro_f1_counts  # same macro-F1 used for the paired bootstrap there

N_BOOT, SEED = 10000, 0
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main(preds):
    df, A, B = tfidf_scored(pd.read_csv(preds))
    df["p_ens"] = (df.p_tfidf + df.p_abusive) / 2
    A, B = split_val(df)  # same seed -> same halves, now carrying p_ens
    T = df[df.split == "test"]
    y, n = T.label.values, len(T)
    pred = {k: (T[c].values >= 0.5).astype(int) for k, c in
            [("tfidf", "p_tfidf"), ("transformer", "p_abusive"), ("ensemble", "p_ens")]}
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    boot = {k: np.concatenate([macro_f1_counts(y[idx[s:s + 500]], p[idx[s:s + 500]])
                               for s in range(0, N_BOOT, 500)]) for k, p in pred.items()}
    lines = [f"Test n={n} (spam-free split, THS_DATA_DIR=data/clean); transformer probs from `{os.path.relpath(preds, ROOT)}`.", "",
             "## Ensemble vs single models (test, threshold 0.5)", "",
             "| model | macro-F1 | AUC | delta (ens - model) | 95% paired-bootstrap CI | McNemar p (b=ens right/other wrong, c=reverse) |",
             "|---|---|---|---|---|---|"]
    col = {"tfidf": "p_tfidf", "transformer": "p_abusive", "ensemble": "p_ens"}
    for k in ("tfidf", "transformer", "ensemble"):
        f1, auc = f1_score(y, pred[k], average="macro"), roc_auc_score(y, T[col[k]])
        if k == "ensemble":
            lines.append(f"| ensemble | {f1:.4f} | {auc:.4f} | - | - | - |"); continue
        d = boot["ensemble"] - boot[k]; lo, hi = np.percentile(d, [2.5, 97.5])
        ce, co = pred["ensemble"] == y, pred[k] == y
        b, c = int((ce & ~co).sum()), int((~ce & co).sum())
        pm = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
        lines.append(f"| {k} | {f1:.4f} | {auc:.4f} | {f1_score(y, pred['ensemble'], average='macro') - f1:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {pm:.3g} (b={b}, c={c}) |")
    lines += ["", f"Bootstrap: {N_BOOT} resamples, seed {SEED}, same resamples for all models.", "",
              "## Conformal policy (alpha = 2% wrongful blocks of non-abusive; calibrated on val half B, "
              f"n_cal={len(B)}; TF-IDF Platt fitted on val half A, n={len(A)})", "",
              "| scorer | beta | lo | hi | realized false-block | realized miss | review % | block % |", "|---|---|---|---|---|---|---|---|"]
    for k in ("tfidf", "transformer", "ensemble"):
        for beta in (0.05, 0.10, 0.20):
            r = run_policy(B, T, col[k], 0.02, beta)
            lines.append(f"| {k} | {beta:.0%} | {r['lo']:.3f} | {r['hi']:.3f} | {r['false_block_rate']:.2%} | "
                         f"{r['miss_rate']:.2%} | {r['review_pct']:.1%} | {r['block_pct']:.1%} |")
    lines += ["", "Guarantee is marginal over calibration draws (class-conditional split conformal); one calibration "
              "set gives realized rates that scatter around alpha/beta (see rows where realized exceeds target)."]
    out = "# Conformal allow/review/block policy and ensemble\n\n" + "\n".join(lines) + "\n"
    open(os.environ.get("THS_REPORT_OUT") or os.path.join(ROOT, "docs", "POLICY_REPORT.md"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
