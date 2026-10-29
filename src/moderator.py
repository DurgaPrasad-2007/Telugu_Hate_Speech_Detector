"""Helpers for the Streamlit 'Moderator Console' tab (app.py): conformal allow/review/block verdict,
leave-one-word-out rationale, obfuscation note, correction log. Label convention: internal y=1 abusive;
the corrections CSV uses the RAW dataset convention 0=abusive, 1=non-abusive."""
import csv, html, os, sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CORRECTIONS = os.path.join(ROOT, "data", "clean", "moderator_corrections.csv")
BETAS = (0.05, 0.10, 0.20)
ALPHA = 0.02


def build_policy(preds_csv, alpha=ALPHA, betas=BETAS):
    """Platt + conformal (lo, hi) per beta from val. With preds_csv (transformer p_abusive) the score is the
    TF-IDF/transformer mean (src/ensemble.py); without it, TF-IDF only. Returns dict(platt, thr, ensemble)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from conformal_policy import tfidf_scored, split_val, conformal_thresholds, load_tfidf_only
    ens = bool(preds_csv) and os.path.isfile(preds_csv)
    df, A, B = tfidf_scored(pd.read_csv(preds_csv) if ens else load_tfidf_only())
    platt = LogisticRegression().fit(A[["d_tfidf"]].values, A.label.values)  # same fit as tfidf_scored
    col = "p_tfidf"
    if ens:
        df["p_ens"] = (df.p_tfidf + df.p_abusive) / 2
        _, B = split_val(df)
        col = "p_ens"
    thr = {b: conformal_thresholds(B[col], B.label, alpha, b) for b in betas}
    return {"platt": platt, "thr": thr, "ensemble": ens, "alpha": alpha}


def load_tfidf_pipe():
    import joblib
    from tfidf_classifier import DEFAULT_MODEL, STUDY_MODEL
    return joblib.load(DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else STUDY_MODEL)


def p_tfidf(pipe, platt, texts):
    from tfidf_classifier import guard
    d = pipe.decision_function([guard(t)[0] for t in texts])
    return platt.predict_proba(np.asarray(d).reshape(-1, 1))[:, 1]


def verdict(p, lo, hi):
    from conformal_policy import decide
    return str(decide([p], lo, hi)[0]).upper()


def occlusion(text, score_fn, min_drop=0.03, max_words=80):
    """Leave-one-word-out. score_fn(list[str]) -> probs. Returns [(word, drop)], drop = base - without."""
    words = text.split()
    if not words or len(words) > max_words:
        return [(w, 0.0) for w in words]
    ps = score_fn([text] + [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))])
    return [(w, float(ps[0] - ps[i + 1])) for i, w in enumerate(words)]


def highlight_html(pairs, min_drop=0.03):
    """Red background proportional to drop (>= min_drop); text is HTML-escaped."""
    top = max([d for _, d in pairs] + [1e-9])
    out = []
    for w, d in pairs:
        e = html.escape(w)
        if d >= min_drop:
            out.append(f'<span style="background:rgba(220,38,38,{0.15 + 0.6 * d / top:.2f});border-radius:4px;'
                       f'padding:0 3px" title="drop {d:+.3f}">{e}</span>')
        else:
            out.append(e)
    return " ".join(out)


def normalization(text):
    """(normalized, changed) via akshara.normalize."""
    from akshara import normalize
    n = normalize(text)
    return n, n != text


def save_correction(text, label, path=CORRECTIONS):
    """Append (text,label,timestamp). label 0=abusive, 1=non-abusive (raw CSV convention)."""
    assert label in (0, 1)
    new = not os.path.isfile(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["text", "label", "timestamp"])
        w.writerow([text, label, datetime.now().isoformat(timespec="seconds")])
