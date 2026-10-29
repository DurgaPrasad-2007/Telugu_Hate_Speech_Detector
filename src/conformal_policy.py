"""Two-threshold allow/review/block policy with finite-sample conformal guarantees.

Label convention (pinned): internal y=1 ABUSIVE, y=0 NON-ABUSIVE (data_utils.load_split flips the raw
CSV convention 0=abusive/1=non-abusive on load). score s = P(abusive).
  block  : s >  hi   -> wrongful-block rate  P(block | non-abusive) <= alpha
  allow  : s <= lo   -> miss rate            P(allow | abusive)     <= beta
  review : otherwise
Both are class-conditional split-conformal quantiles (for a monotone 0/1 loss this equals conformal risk
control, Angelopoulos & Bates, with risk = per-class error). Guarantee is marginal over calibration draws
and needs calibration/test exchangeability within each class.
usage: THS_DATA_DIR=data/clean python src/conformal_policy.py [results/<run>/preds.csv]
"""
import math, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ABUSIVE, NON_ABUSIVE = 1, 0


def conformal_thresholds(s, y, alpha, beta):
    s, y = np.asarray(s, float), np.asarray(y)
    s0, s1 = np.sort(s[y == NON_ABUSIVE]), np.sort(s[y == ABUSIVE])
    k = math.ceil((len(s0) + 1) * (1 - alpha))      # block only above the k-th smallest non-abusive score
    hi = s0[k - 1] if k <= len(s0) else np.inf      # P(s_new > s0_(k)) <= alpha
    j = math.floor((len(s1) + 1) * beta)            # allow at/below the j-th smallest abusive score
    lo = s1[j - 1] if j >= 1 else -np.inf           # P(s_new <= s1_(j)) <= j/(n+1) <= beta
    return float(lo), float(hi)


def decide(s, lo, hi):
    s = np.asarray(s, float)
    block = s > hi
    allow = (s <= lo) & ~block
    return np.where(block, "block", np.where(allow, "allow", "review"))


def realized(s, y, lo, hi):
    d, y = decide(s, lo, hi), np.asarray(y)
    return {"false_block_rate": float((d[y == NON_ABUSIVE] == "block").mean()),
            "miss_rate": float((d[y == ABUSIVE] == "allow").mean()),
            "review_pct": float((d == "review").mean()),
            "block_pct": float((d == "block").mean()), "n": int(len(y))}


def split_val(df, seed=0):
    """val -> (A: fit calibrators/Platt, B: conformal calibration), stratified-free random halves."""
    v = df[df.split == "val"].sample(frac=1, random_state=seed)
    h = len(v) // 2
    return v.iloc[:h], v.iloc[h:]


def tfidf_raw(texts):
    import joblib
    from tfidf_classifier import DEFAULT_MODEL, STUDY_MODEL, guard
    pipe = joblib.load(DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else STUDY_MODEL)
    return pipe.decision_function([guard(t)[0] for t in texts])


def tfidf_scored(df):
    """Add p_tfidf (Platt fitted on val half A only). Returns (df, A, B)."""
    from sklearn.linear_model import LogisticRegression
    df = df.copy()
    df["d_tfidf"] = tfidf_raw(df.text.tolist())
    A, B = split_val(df)
    platt = LogisticRegression().fit(A[["d_tfidf"]].values, A.label.values)
    df["p_tfidf"] = platt.predict_proba(df[["d_tfidf"]].values)[:, 1]
    A, B = split_val(df)
    return df, A, B


def load_tfidf_only():
    from data_utils import load_split
    parts = []
    for sp in ("val", "test"):
        d = load_split("telugu", sp, "normalized")
        parts.append(pd.DataFrame({"split": sp, "text": d.text, "label": d.label}))
    return pd.concat(parts, ignore_index=True)


def run_policy(B, T, col, alpha=0.02, beta=0.10):
    lo, hi = conformal_thresholds(B[col], B.label, alpha, beta)
    return {"lo": lo, "hi": hi, "alpha": alpha, "beta": beta, "n_cal": len(B), **realized(T[col], T.label, lo, hi)}


def selfcheck():
    rng = np.random.default_rng(0)
    def draw(n):
        y = rng.integers(0, 2, n)
        return np.clip(rng.normal(0.25 + 0.5 * y, 0.2), 0, 1), y
    sc, yc = draw(4000)
    lo, hi = conformal_thresholds(sc, yc, 0.02, 0.10)
    assert lo < hi
    assert realized(sc, yc, lo, hi)["false_block_rate"] <= 0.02 + 1e-9   # in-sample bound is exact
    fb, ms = [], []
    for _ in range(200):                                                 # coverage over fresh test draws
        st, yt = draw(2000)
        r = realized(st, yt, lo, hi); fb.append(r["false_block_rate"]); ms.append(r["miss_rate"])
    assert np.mean(fb) <= 0.02 + 0.003 and np.mean(ms) <= 0.10 + 0.01, (np.mean(fb), np.mean(ms))
    assert (decide([0.0, 1.0], -np.inf, np.inf) == ["review", "review"]).all()
    print(f"selfcheck ok: mean false-block {np.mean(fb):.4f} (<=0.02), mean miss {np.mean(ms):.4f} (<=0.10)")


if __name__ == "__main__":
    selfcheck()
    p = sys.argv[1] if len(sys.argv) > 1 else None
    df = pd.read_csv(p) if p else load_tfidf_only()
    df, A, B = tfidf_scored(df)
    T = df[df.split == "test"]
    for beta in (0.05, 0.10, 0.20):
        print("tfidf", run_policy(B, T, "p_tfidf", 0.02, beta))
    if "p_abusive" in df:
        for beta in (0.05, 0.10, 0.20):
            print("transformer", run_policy(B, T, "p_abusive", 0.02, beta))
