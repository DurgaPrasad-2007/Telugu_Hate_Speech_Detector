"""Label audit: 5-fold OOF TF-IDF probs on clean train+val, confident-learning suspect scoring, review CSVs,
and a test-F1 check of removing/relabeling the top-k auto-suspects (unreviewed heuristic).
Label convention: CSV 0=abusive; load_split already maps to internal 1=abusive (pinned below).
usage: THS_DATA_DIR=data/clean python src/label_audit.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "data", "clean")
MURIL = os.path.join(ROOT, "results", "clean_muril-base-cased_lp8_lr5e-5", "preds.csv")
KS, N_BOOT, SEED, BASE = (50, 100, 200, 400), 10000, 0, 0.8960


def cl_scores(p, y):
    """Confident-learning style. p = OOF P(abusive), y in {0,1} (internal: 1=abusive).
    Per-class threshold t_j = mean P(class j) over rows given label j. Row is flagged if the other class's
    prob >= its threshold and exceeds the given-class prob. score = p_other - p_given (higher = more suspect)."""
    P = np.stack([1 - p, p], 1)
    t = np.array([P[y == j, j].mean() for j in (0, 1)])
    other = 1 - y
    p_given, p_other = P[np.arange(len(y)), y], P[np.arange(len(y)), other]
    flag = (p_other >= t[other]) & (p_other > p_given)
    return p_other - p_given, flag, t


def features():
    from sklearn.feature_extraction.text import TfidfVectorizer as T
    from sklearn.pipeline import FeatureUnion
    return FeatureUnion([("c", T(analyzer="char_wb", ngram_range=(2, 5), max_features=300000, sublinear_tf=True)),
                         ("w", T(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_features=300000, sublinear_tf=True))])


def oof_probs(texts, y, C, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC
    d = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(texts, y):
        u = features(); X = u.fit_transform(texts[tr])
        d[te] = LinearSVC(C=C, max_iter=10000).fit(X, y[tr]).decision_function(u.transform(texts[te]))
    # Platt on OOF margins (1-D, 2 params; negligible optimism)
    return LogisticRegression().fit(d[:, None], y).predict_proba(d[:, None])[:, 1]


def fit_eval(trt, try_, vat, vay, tet, C=None):
    from sklearn.metrics import f1_score
    from sklearn.svm import LinearSVC
    u = features(); X = u.fit_transform(trt)
    Xv, Xt = u.transform(vat), u.transform(tet)
    fit = lambda c: LinearSVC(C=c, max_iter=10000).fit(X, try_)
    if C is None:
        C = max([0.1, 0.3, 1, 3, 10], key=lambda c: f1_score(vay, fit(c).predict(Xv), average="macro"))
    return C, fit(C).predict(Xt)


def main():
    from sklearn.metrics import f1_score
    from significance import macro_f1_counts
    from data_utils import load_split
    S = {s: load_split("telugu", s, "normalized") for s in ("train", "val", "test")}
    for s in S:
        assert set(S[s].label) <= {0, 1}
    print(f"internal labels: 1=abusive; train abusive share {S['train'].label.mean():.3f}")
    C, base_pred = fit_eval(S["train"].text.values, S["train"].label.values, S["val"].text.values,
                            S["val"].label.values, S["test"].text.values)
    yt = S["test"].label.values
    base_f1 = f1_score(yt, base_pred, average="macro")
    print(f"baseline C={C} test macro-F1={base_f1:.4f} (reference {BASE})")

    A = pd.concat([S["train"].assign(split="train"), S["val"].assign(split="val")], ignore_index=True)
    A = A.drop_duplicates("text").reset_index(drop=True)
    p = oof_probs(A.text.values, A.label.values, C)
    A["p_abusive_oof"] = p
    A["score"], A["flagged"], thr = cl_scores(p, A.label.values)
    # MuRIL preds exist only for val/test rows (model trained on train) -> held-out column for val rows only
    m = pd.read_csv(MURIL); m = m[m.split == "val"].drop_duplicates("text").set_index("text").p_abusive
    A["p_muril_val"] = A.text.map(m)
    A["muril_disagrees"] = np.where(A.p_muril_val.isna(), "", ((A.p_muril_val >= .5).astype(int) != A.label).astype(str))
    A["given_label_csv"] = 1 - A.label  # CSV convention 0=abusive
    A = A.sort_values("score", ascending=False).reset_index(drop=True)
    A["rank"] = np.arange(1, len(A) + 1)
    cols = ["rank", "text", "given_label_csv", "label", "p_abusive_oof", "score", "flagged", "split", "p_muril_val", "muril_disagrees"]
    A[cols].rename(columns={"label": "given_label_internal_1abusive"}).to_csv(
        os.path.join(OUT, "label_audit_suspects.csv"), index=False, encoding="utf-8-sig")
    A.head(300)[["rank", "text", "given_label_csv"]].assign(human_label="").to_csv(
        os.path.join(OUT, "label_audit_gold_review.csv"), index=False, encoding="utf-8-sig")
    nf = int(A.flagged.sum()); fa, fn = int((A.flagged & (A.label == 1)).sum()), int((A.flagged & (A.label == 0)).sum())
    print(f"pooled n={len(A)}; thresholds t(non-abusive,abusive)={thr.round(3)}; flagged {nf} (given abusive {fa}, given non-abusive {fn})")
    mv = A[A.p_muril_val.notna()]
    mm = ((mv.p_muril_val >= .5).astype(int) != mv.label)
    print(f"MuRIL val: {len(mv)} rows, disagrees on {mm.sum()}; among OOF-flagged val rows {int(mm[mv.flagged].sum())}/{int(mv.flagged.sum())}")
    ex = pd.concat([A.head(8), A.iloc[50:300].sample(7, random_state=SEED)])
    print("\n15 example suspects (ranks 1-8, then 7 sampled from ranks 51-300):")
    for _, r in ex.iterrows():
        print(f"#{r['rank']} given={'abusive' if r.label == 1 else 'non-abusive'} p_ab={r.p_abusive_oof:.3f} | {r.text[:140]}")

    # --- test impact: act on TRAIN-only suspects (val/test never modified) ---
    tr = S["train"]; sus = A[A.split == "train"]
    rng = np.random.default_rng(SEED); n = len(yt)
    idx = rng.integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    bs = lambda pr: np.concatenate([macro_f1_counts(yt[idx[s:s + 500]], pr[idx[s:s + 500]]) for s in range(0, N_BOOT, 500)])
    b0, rows = bs(base_pred), []
    for k in KS:
        top = set(sus.head(k).text); rnd = set(rng.choice(tr.text.values, k, replace=False))
        variants = {"remove": (tr[~tr.text.isin(top)].text.values, tr[~tr.text.isin(top)].label.values),
                    "relabel": (tr.text.values, np.where(tr.text.isin(top), 1 - tr.label, tr.label)),
                    "random-remove (control)": (tr[~tr.text.isin(rnd)].text.values, tr[~tr.text.isin(rnd)].label.values)}
        for name, (t_, y_) in variants.items():
            _, pr = fit_eval(t_, y_, S["val"].text.values, S["val"].label.values, S["test"].text.values, C=C)
            f1 = f1_score(yt, pr, average="macro"); d = bs(pr) - b0; lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append((k, name, f1, f1 - base_f1, lo, hi)); print(f"k={k} {name}: {f1:.4f} ({f1 - base_f1:+.4f}) [{lo:+.4f},{hi:+.4f}]")
    L = [f"Data: THS_DATA_DIR=data/clean (spam-free). Internal labels 1=abusive (CSV 0=abusive). Pooled train+val n={len(A)}, "
         f"test n={n}. 5-fold stratified OOF LinearSVC (char_wb 2-5 + word 1-2 TF-IDF, C={C} chosen on val), Platt on OOF margins.", "",
         f"Per-class thresholds (mean P(class) among rows labeled that class): non-abusive {thr[0]:.3f}, abusive {thr[1]:.3f}. "
         f"Flagged suspects: {nf} (given abusive {fa}, given non-abusive {fn}).", "",
         f"MuRIL (`clean_muril-base-cased_lp8_lr5e-5`) has predictions only for val/test (it trained on train), so it is a held-out "
         f"cross-check on val rows only: {len(mv)} val rows, MuRIL disagrees with the given label on {int(mm.sum())}; "
         f"among {int(mv.flagged.sum())} OOF-flagged val rows it disagrees on {int(mm[mv.flagged].sum())}.", "",
         "## Test macro-F1 after acting on top-k TRAIN suspects", "",
         f"Untouched baseline (this script, same features, C={C}): {base_f1:.4f} (reference {BASE}). Val/test untouched, C fixed. "
         f"Delta CI = paired bootstrap ({N_BOOT} resamples, seed {SEED}, same resamples for all rows, macro_f1_counts from significance.py). "
         "Control = removing k random train rows.", "",
         "| k | action | test macro-F1 | delta vs baseline | 95% CI |", "|---|---|---|---|---|"]
    L += [f"| {k} | {a} | {f:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |" for k, a, f, d, lo, hi in rows]
    L += ["", "## Example suspects (sanity check; not human verified)", ""]
    L += [f"- #{r['rank']} given {'abusive' if r.label == 1 else 'non-abusive'}, p_abusive_oof={r.p_abusive_oof:.3f}: {r.text[:140]}" for _, r in ex.iterrows()]
    L += ["", "## Caveats", "",
          "- Removing/relabeling on auto-suspects is an unreviewed heuristic: no human has confirmed any suspect is a true mislabel. "
          "Relabeling assumes the model is right. The deltas say nothing about label precision.",
          "- Suspects come from a TF-IDF model, so they are biased toward what bag-of-ngrams cannot fit.",
          "- `label_audit_gold_review.csv` (top 300) has a blank `human_label` column (use CSV convention 0=abusive); "
          "label precision of the top 300 is unmeasured until it is filled."]
    open(os.path.join(ROOT, "docs", "LABEL_AUDIT_REPORT.md"), "w", encoding="utf-8").write("# Label audit report\n\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
