"""HateCheck-style functional tests for Telugu (tests/functional/telugu_hatecheck.csv), CPU only.
Scores each case with TF-IDF and the TF-IDF+MuRIL ensemble (same scorer/thresholds as the Moderator Console),
reports pass rate per functionality and, for identity-mention neutral cases, false-positive rate and
REVIEW rate (conformal 'uncertain' verdict) per group as a bias proxy. Writes docs/FUNCTIONAL_TEST_REPORT.md.
Pass = (p >= 0.5) matches expected. Low-confidence cases are excluded from pass rates (listed separately).
Test cases are model-authored: need native-speaker review before use as a release gate.
usage: THS_DATA_DIR=data/clean python src/functional_eval.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CASES = os.path.join(ROOT, "tests", "functional", "telugu_hatecheck.csv")
MURIL_RUN = os.environ.get("THS_MURIL_RUN") or os.path.join(ROOT, "results", "clean_muril-base-cased_lp8_lr5e-5")
REPORT = os.environ.get("THS_REPORT_OUT") or os.path.join(ROOT, "docs", "FUNCTIONAL_TEST_REPORT.md")  # env: candidate run, keep committed report
BETA = 0.10


def load_cases(path=CASES):
    df = pd.read_csv(path, encoding="utf-8")
    assert list(df.columns) == ["text", "expected", "functionality", "confidence"], list(df.columns)
    assert set(df.expected) <= {"abusive", "non_abusive"} and set(df.confidence) <= {"low", "high"}
    return df


def add_pass(df, p_col, thr=0.5):
    df = df.copy()
    df["pred_" + p_col] = np.where(df[p_col] >= thr, "abusive", "non_abusive")
    df["pass_" + p_col] = df["pred_" + p_col] == df.expected
    return df


def pass_rates(df, pass_col):
    """Per functionality over high-confidence rows: (n, pass rate)."""
    h = df[df.confidence == "high"]
    g = h.groupby("functionality")[pass_col].agg(["size", "mean"])
    return g.rename(columns={"size": "n", "mean": "rate"})


def identity_bias(df, p_col, verdict_col):
    """Per identity_neutral_* group (high confidence): FP rate (p>=0.5) and REVIEW rate."""
    h = df[(df.confidence == "high") & df.functionality.str.startswith("identity_neutral_")]
    return h.groupby("functionality").agg(n=("text", "size"), fp=(p_col, lambda s: float((s >= 0.5).mean())),
                                          review=(verdict_col, lambda s: float((s == "REVIEW").mean())))


def score(df):
    import moderator
    preds = os.path.join(MURIL_RUN, "preds.csv")
    pol = moderator.build_policy(preds)
    lo, hi = pol["thr"][BETA]
    pipe = moderator.load_tfidf_pipe()
    norm = [moderator.normalization(t)[0] for t in df.text]
    df = df.copy()
    df["p_tfidf"] = moderator.p_tfidf(pipe, pol["platt"], norm)
    from infer import HateSpeechClassifier
    m = HateSpeechClassifier(os.path.join(MURIL_RUN, "model"), device="cpu")
    pm = np.array([r.get("probability") or 0.0 for r in m.predict(norm, _quiet=True)])
    df["p_muril"] = pm
    df["p_ens"] = (df.p_tfidf + pm) / 2
    # TF-IDF-only verdict needs TF-IDF-only thresholds
    pol_t = moderator.build_policy(None)
    lo_t, hi_t = pol_t["thr"][BETA]
    df["v_tfidf"] = [moderator.verdict(p, lo_t, hi_t) for p in df.p_tfidf]
    df["v_ens"] = [moderator.verdict(p, lo, hi) for p in df.p_ens]
    return df


def main():
    df = load_cases()
    df = score(df)
    for c in ("p_tfidf", "p_ens"):
        df = add_pass(df, c)
    rt, re_ = pass_rates(df, "pass_p_tfidf"), pass_rates(df, "pass_p_ens")
    both = rt.join(re_, lsuffix="_tfidf", rsuffix="_ens")
    hc = df[df.confidence == "high"]
    L = [f"Cases: {len(df)} total, {len(hc)} high-confidence (scored), {len(df) - len(hc)} low-confidence (excluded from pass rates, "
         "shown in failure lists as 'low'). Pass = P(abusive) >= 0.5 matches expected. Ensemble = mean of TF-IDF and clean MuRIL "
         f"(Moderator Console scorer); REVIEW verdict uses conformal thresholds at alpha=2%, beta={BETA:.0%}.", "",
         "**Caveat: all test cases are model-authored (not written by a native Telugu speaker). They need native-speaker "
         "review before being used as a release gate; low-confidence flags are the author's own doubt, not a validation.**", "",
         f"Overall (high-confidence): TF-IDF {hc.pass_p_tfidf.mean():.1%}, ensemble {hc.pass_p_ens.mean():.1%} (n={len(hc)}).", "",
         "## Pass rate per functionality (high-confidence)", "", "| functionality | n | TF-IDF | ensemble |", "|---|---|---|---|"]
    for f, r in both.iterrows():
        L.append(f"| {f} | {int(r.n_tfidf)} | {r.rate_tfidf:.0%} | {r.rate_ens:.0%} |")
    L += ["", "## Identity-mention neutral cases: false-positive rate and REVIEW rate by group (bias proxy)", "",
          "| group | n | FP TF-IDF | FP ensemble | REVIEW TF-IDF | REVIEW ensemble |", "|---|---|---|---|---|---|"]
    bt, be = identity_bias(df, "p_tfidf", "v_tfidf"), identity_bias(df, "p_ens", "v_ens")
    for g in bt.index:
        L.append(f"| {g.replace('identity_neutral_', '')} | {int(bt.n[g])} | {bt.fp[g]:.0%} | {be.fp[g]:.0%} | {bt.review[g]:.0%} | {be.review[g]:.0%} |")
    L += ["", "Small n per group (5-9): differences of one case are 11-20 points; read as a smoke signal, not a measured disparity.", "",
          "## Failing cases", ""]
    for name, pc in (("TF-IDF", "p_tfidf"), ("Ensemble", "p_ens")):
        bad = df[~df["pass_" + pc]].sort_values("functionality")
        L += [f"### {name} ({len(bad)} failures incl. low-confidence)", "", "| functionality | expected | p | conf | text |", "|---|---|---|---|---|"]
        L += [f"| {r.functionality} | {r.expected} | {r[pc]:.2f} | {r.confidence} | {r.text.replace('|', '/')} |" for _, r in bad.iterrows()]
        L.append("")
    out = "# Telugu HateCheck-style functional test report\n\n" + "\n".join(L) + "\n"
    open(REPORT, "w", encoding="utf-8").write(out)
    if os.environ.get("THS_SCORES_OUT"):
        df.to_csv(os.environ["THS_SCORES_OUT"], index=False, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8"); print(out)


if __name__ == "__main__":
    main()
