"""Per-comment TEST predictions for the study runs + paired significance tests vs the selected model.

Inference: raw model logits, argmax (no policy thresholds, no temperature, no emoji
second view, no anti-obfuscation canonicalisation -- defend=False), max_length 96,
normalized text (load_split already applies normalize_text). Predictions cached to
results/study/preds/<run>.npy.

Run: .venv\\Scripts\\python -I src\\significance.py
"""
import os
import sys
import json
import time
import gc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
STUDY = os.path.join(ROOT, "results", "study")
PREDS = os.path.join(STUDY, "preds")
os.makedirs(PREDS, exist_ok=True)

from data_utils import load_split  # noqa: E402
from sklearn.metrics import f1_score, accuracy_score  # noqa: E402
from scipy.stats import binomtest, ttest_ind  # noqa: E402
import pandas as pd  # noqa: E402

MAXLEN = 96
BATCH = 64
N_BOOT = 10000
SEED = 0

# run name -> directory under results/study (weights in <dir>/model when present)
RUNS = {
    "A_xlmr_lora_s2024": "xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024",
    "B_xlmr_full_s2024": "xlm-roberta-base_full_lr2e-5_e5_s2024",
    "B_xlmr_full_s42": "xlm-roberta-base_full_lr2e-5_e5_s42",
    "B_xlmr_full_s123": "xlm-roberta-base_full_lr2e-5_e5_s123",
    "C_xlmr_dora_s2024": "xlm-roberta-base_dora_r8_a64_lr2e-4_e10_s2024",
    "D_xlmr_adalora_s2024": "xlm-roberta-base_adalora_r8_a64_lr2e-4_e10_s2024",
    "E_muril_lora_s2024": "lora_r8_a64_lr2e-4_e10_s2024",
    "E_muril_full_s2024": "full_lr2e-5_e5_s2024",
    "F_paper_xlmr_lora_r4_a32_s2024": "paper_xlmr_lora_r4_a32_s2024",
    "G_mbert_lora_s2024": "bert-base-multilingual-cased_lora_r8_a64_lr2e-4_e10_s2024",
    "G_indicbertv2_lora_s2024": "IndicBERTv2-MLM-only_lora_r8_a64_lr2e-4_e10_s2024",
    # seed-mates: only for the scorecard's abusive-recall mean over 3 seeds
    "X_xlmr_lora_s42": "xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s42",
    "X_xlmr_lora_s123": "xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s123",
    "X_xlmr_dora_s42": "xlm-roberta-base_dora_r8_a64_lr2e-4_e10_s42",
    "X_xlmr_dora_s123": "xlm-roberta-base_dora_r8_a64_lr2e-4_e10_s123",
}
TFIDF_PATH = os.path.join(PREDS, "tfidf_best.npy")
RECORDED_A_TEST_F1 = 0.9041991738179893  # results/final_best/export_metrics.json fp32_torch.test


def model_dir_of(d):
    p = os.path.join(STUDY, d, "model")
    return p if os.path.isdir(p) else os.path.join(STUDY, d)


def infer_run(model_dir, texts):
    import torch
    from infer import HateSpeechClassifier
    clf = HateSpeechClassifier(model_dir, defend=False, max_length=MAXLEN)
    preds = []
    with torch.no_grad():
        for s in range(0, len(texts), BATCH):
            enc = clf.tokenizer(texts[s:s + BATCH], truncation=True, padding=True,
                                max_length=MAXLEN, return_tensors="pt").to(clf.device)
            logits = clf.model(**enc).logits
            preds.append(logits.argmax(-1).cpu().numpy())
    device = clf.device
    del clf
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(preds).astype(np.int64)


def macro_f1_counts(y, p):
    """Macro-F1 over classes {0,1} computed from (..., n) arrays via sums (fast bootstrap)."""
    y1 = y == 1
    p1 = p == 1
    tp1 = (y1 & p1).sum(-1)
    fn1 = (y1 & ~p1).sum(-1)
    fp1 = (~y1 & p1).sum(-1)
    tn1 = (~y1 & ~p1).sum(-1)
    d1 = np.maximum(2 * tp1 + fp1 + fn1, 1)
    d0 = np.maximum(2 * tn1 + fn1 + fp1, 1)
    return (2 * tn1 / d0 + 2 * tp1 / d1) / 2


def load_pred(name):
    if name == "H_tfidf_best":
        return np.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None
    f = os.path.join(PREDS, f"{name}.npy")
    return np.load(f) if os.path.exists(f) else None


def main():
    t_all = time.time()
    te = load_split("telugu", "test", "normalized")
    y = te["label"].values.astype(np.int64)
    texts = te["text"].tolist()
    n = len(y)
    print(f"test n={n}", flush=True)

    timing, failed = {}, {}
    for name, d in RUNS.items():
        out = os.path.join(PREDS, f"{name}.npy")
        if os.path.exists(out):
            print(f"[cache] {name}", flush=True)
            continue
        mdir = model_dir_of(d)
        if not os.path.isdir(mdir):
            failed[name] = f"missing dir {mdir}"
            print(f"[FAIL] {name}: {failed[name]}", flush=True)
            continue
        t0 = time.time()
        try:
            p = infer_run(mdir, texts)
        except Exception as e:  # record and continue with the other runs
            failed[name] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"[FAIL] {name}: {failed[name]}", flush=True)
            continue
        dt = time.time() - t0
        timing[name] = dt
        np.save(out, p)
        print(f"[infer] {name}: {dt:.1f}s incl. load, test macro-F1={f1_score(y, p, average='macro'):.4f}",
              flush=True)

    A = load_pred("A_xlmr_lora_s2024")
    if A is None:
        raise SystemExit("selected model A has no predictions; cannot run comparisons")

    checks = {"A_xlmr_lora_s2024": {
        "recomputed_test_macro_f1": float(f1_score(y, A, average="macro")),
        "recorded_test_macro_f1": RECORDED_A_TEST_F1}}
    print("self-check:", checks, flush=True)

    # same bootstrap resamples for every comparison (paired design)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n), dtype=np.int32)

    def boot_f1(p):
        out = np.empty(N_BOOT)
        for s in range(0, N_BOOT, 500):
            ii = idx[s:s + 500]
            out[s:s + 500] = macro_f1_counts(y[ii], p[ii])
        return out

    def mcnemar(pa, pb):
        ca = pa == y
        cb = pb == y
        b = int(np.sum(ca & ~cb))  # A right, other wrong
        c = int(np.sum(~ca & cb))  # A wrong, other right
        p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
        return b, c, p

    f1_A = float(f1_score(y, A, average="macro"))
    boot_A = boot_f1(A)
    a_ci = np.percentile(boot_A, [2.5, 97.5])
    own = {"name": "A_xlmr_lora_s2024", "macro_f1": f1_A,
           "macro_f1_ci95": [float(a_ci[0]), float(a_ci[1])],
           "accuracy": float(accuracy_score(y, A))}

    per_run = {}
    ab = y == 1
    for name in list(RUNS.keys()) + ["H_tfidf_best"]:
        p = load_pred(name)
        if p is None:
            continue
        pr = p == 1
        per_run[name] = {
            "test_macro_f1": float(f1_score(y, p, average="macro")),
            "test_accuracy": float(accuracy_score(y, p)),
            "abusive_recall": float((pr & ab).sum() / max(ab.sum(), 1)),
            "abusive_precision": float((pr & ab).sum() / max(pr.sum(), 1))}

    comp_specs = [
        ("B: XLM-R full FT, seed 2024", "B_xlmr_full_s2024"),
        ("B: XLM-R full FT, seed 42", "B_xlmr_full_s42"),
        ("B: XLM-R full FT, seed 123", "B_xlmr_full_s123"),
        ("C: XLM-R DoRA r8 a64, seed 2024", "C_xlmr_dora_s2024"),
        ("D: XLM-R AdaLoRA r8 a64, seed 2024", "D_xlmr_adalora_s2024"),
        ("E: MuRIL LoRA r8 a64, seed 2024", "E_muril_lora_s2024"),
        ("E: MuRIL full FT, seed 2024", "E_muril_full_s2024"),
        ("F: paper XLM-R LoRA r4 a32, seed 2024", "F_paper_xlmr_lora_r4_a32_s2024"),
        ("G: mBERT LoRA r8 a64, seed 2024", "G_mbert_lora_s2024"),
        ("G: IndicBERTv2-MLM-only LoRA r8 a64, seed 2024", "G_indicbertv2_lora_s2024"),
        ("H: TF-IDF (selected by val) baseline", "H_tfidf_best"),
    ]
    comparisons = []
    for label, name in comp_specs:
        p = load_pred(name)
        if p is None:
            comparisons.append({"comparison": label, "run": name,
                                "error": failed.get(name, "no predictions")})
            continue
        delta = f1_A - float(f1_score(y, p, average="macro"))
        bd = boot_A - boot_f1(p)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        b, c, pm = mcnemar(A, p)
        if lo <= 0 <= hi:
            verdict = "no significant difference"
        elif lo > 0:
            verdict = "A better"
        else:
            verdict = "A worse"
        comparisons.append({"comparison": label, "run": name,
                            "delta_macro_f1_A_minus_other": float(delta),
                            "bootstrap_mean_delta": float(bd.mean()),
                            "ci95": [float(lo), float(hi)], "mcnemar_p": pm,
                            "b_A_right_other_wrong": b, "c_A_wrong_other_right": c,
                            "verdict": verdict})
        print(f"{label}: delta={delta:+.4f} CI=[{lo:+.4f},{hi:+.4f}] McNemar p={pm:.3g} "
              f"b={b} c={c} -> {verdict}", flush=True)

    # seed-level Welch t-test over the 3 seeds (study_results.csv)
    sr = pd.read_csv(os.path.join(STUDY, "study_results.csv"))
    base = sr[sr["model"] == "FacebookAI/xlm-roberta-base"]
    lora = base[(base["method"] == "lora") & (base["lora_r"] == 8) & (base["lora_alpha"] == 64) &
                np.isclose(base["lr"], 2e-4) & (base["epochs"] == 10)]["test_macro_f1"].values
    full = base[(base["method"] == "full") & np.isclose(base["lr"], 2e-5) &
                (base["epochs"] == 5)]["test_macro_f1"].values
    welch = {"n_lora_seeds": int(len(lora)), "n_full_seeds": int(len(full)),
             "lora_test_macro_f1": [float(v) for v in lora],
             "full_test_macro_f1": [float(v) for v in full],
             "note": "n=3 per arm; very low power; Welch (unequal variance) t-test"}
    if len(lora) >= 2 and len(full) >= 2:
        t, pw = ttest_ind(lora, full, equal_var=False)
        welch.update({"t": float(t), "p_welch": float(pw),
                      "delta_mean_lora_minus_full": float(lora.mean() - full.mean())})

    out = {
        "test_n": int(n),
        "inference": {"max_length": MAXLEN, "batch": BATCH, "output": "argmax of raw logits",
                      "defend": False, "policy_thresholds": "not used", "temperature": "not used",
                      "emoji_second_view": "not used"},
        "bootstrap": {"n_resamples": N_BOOT, "seed": SEED,
                      "type": "paired percentile 95% CI of macro-F1 difference"},
        "selected_A_own": own,
        "self_check": checks,
        "per_run_test": per_run,
        "comparisons": comparisons,
        "seed_level_welch": welch,
        "failed": failed,
        "inference_seconds_incl_load": timing,
        "total_seconds": time.time() - t_all,
    }
    with open(os.path.join(STUDY, "significance.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = ["# Significance: selected model (A) vs other runs, TEST set", "",
             f"n = {n} test comments. Delta = macro-F1(A) - macro-F1(other). "
             f"95% CI from paired bootstrap ({N_BOOT} resamples, seed {SEED}). "
             "McNemar = exact two-sided binomial test on discordant pairs.", "",
             f"A own macro-F1 = {f1_A:.4f}, 95% bootstrap CI [{a_ci[0]:.4f}, {a_ci[1]:.4f}].", "",
             "| Comparison | Delta macro-F1 | 95% CI | McNemar p | b (A right, other wrong) | "
             "c (A wrong, other right) | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for c_ in comparisons:
        if "error" in c_:
            lines.append(f"| {c_['comparison']} | n/a | n/a | n/a | n/a | n/a | not run: {c_['error']} |")
            continue
        lines.append(f"| {c_['comparison']} | {c_['delta_macro_f1_A_minus_other']:+.4f} | "
                     f"[{c_['ci95'][0]:+.4f}, {c_['ci95'][1]:+.4f}] | {c_['mcnemar_p']:.3g} | "
                     f"{c_['b_A_right_other_wrong']} | {c_['c_A_wrong_other_right']} | {c_['verdict']} |")
    lines += ["", "## Seed-level (study_results.csv, XLM-R, 3 seeds each)", ""]
    if "p_welch" in welch:
        lines.append(f"LoRA r8 a64 (n={welch['n_lora_seeds']}) vs full FT (n={welch['n_full_seeds']}): "
                     f"delta mean = {welch['delta_mean_lora_minus_full']:+.4f}, Welch t = {welch['t']:.3f}, "
                     f"p = {welch['p_welch']:.3g}. n=3 per arm, so this test has very low power.")
    if failed:
        lines += ["", "## Failed / not run", ""] + [f"- {k}: {v}" for k, v in failed.items()]
    with open(os.path.join(STUDY, "significance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"done in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
