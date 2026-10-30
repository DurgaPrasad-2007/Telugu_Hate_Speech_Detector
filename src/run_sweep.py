"""Backbone sweep on data/clean (seed 42 unless noted). Stages (each skips outputs that exist):
  python src/run_sweep.py train            # all configs in CFG (GPU, sequential)
  python src/run_sweep.py seeds NAME ...   # seeds 1,2 for named configs
  python src/run_sweep.py report [NAME..]  # F1, ensemble, functional, bootstrap vs MuRIL baseline -> results/sweep/summary.csv
Preds come from train.py --dump_preds (original clean val/test). A failed config is logged and skipped."""
import os, sys, subprocess, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LP = ["--method", "lora", "--target-modules", "all-linear", "--loraplus-ratio", "8", "--lr", "5e-5"]
FULL = ["--method", "full", "--lr", "2e-5"]
COMMON = ["--domain", "telugu", "--epochs", "4", "--train_bs", "32", "--max_length", "128", "--fp16", "--early_stop_patience", "2"]
MM, IN, MU = "jhu-clsp/mmBERT-base", "ai4bharat/IndicBERTv2-MLM-only", "google/muril-base-cased"
CFG = {  # name -> (model, method args, extra)
    "mmbert_lp": (MM, LP, []), "mmbert_full": (MM, FULL, []),
    "indic_lp": (IN, LP, []), "indic_full": (IN, FULL, []),
    "muril_full": (MU, FULL, []),
    "sarvam_qlora": ("sarvamai/sarvam-1", ["--method", "qlora", "--target-modules", "all-linear", "--lr", "1e-4"],
                     ["--train_bs", "4", "--grad_accum", "8", "--eval_bs", "8", "--epochs", "2", "--grad_ckpt"]),
}
BASE = "results/clean_muril-base-cased_lp8_lr5e-5"  # MuRIL LoRA+ seed 42 baseline


def run_dir(name, seed): return f"results/sweep/{name}_s{seed}"


def train(name, seed=42):
    m, meth, extra = CFG[name]; r = run_dir(name, seed)
    if os.path.exists(os.path.join(ROOT, r, "preds.csv")): return
    cmd = [PY, "src/train.py", "--model", m, *meth, *COMMON, *extra, "--seed", str(seed), "--dump_preds",
           "--output_root", "results/sweep", "--run_name", f"{name}_s{seed}"]
    log = open(os.path.join(ROOT, "results", "sweep", f"{name}_s{seed}.log"), "w")
    rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "THS_DATA_DIR": "data/clean"}).returncode
    shutil.rmtree(os.path.join(ROOT, r, "checkpoints"), ignore_errors=True)  # optimizer checkpoints are GBs; disk is tight
    print(name, seed, "rc", rc, flush=True)


def report(names):
    os.environ["THS_DATA_DIR"] = os.path.join(ROOT, "data", "clean"); sys.path.insert(0, HERE)
    import numpy as np, pandas as pd
    from sklearn.metrics import f1_score
    from negation_compare import scored
    from significance import macro_f1_counts
    tf = os.path.join(ROOT, "models", "tfidf", "tfidf_best.joblib")
    runs = {"muril_lp(base)": [BASE if s == 42 else f"results/seeds/clean_s{s}" for s in (1, 2, 42)]}
    for n in names: runs[n] = [run_dir(n, s) for s in (1, 2, 42) if os.path.exists(os.path.join(ROOT, run_dir(n, s), "preds.csv"))]
    rows, probs = [], {}
    for n, rs in runs.items():
        for r in rs:
            d = scored(tf, os.path.join(ROOT, r, "preds.csv")); y = d.label.values
            f = lambda c: f1_score(y, (d[c] >= .5).astype(int), average="macro")
            seed = int(r.rsplit("_s", 1)[-1]) if "_s" in r.rsplit("/", 1)[-1] and r != BASE else 42
            sc = os.path.join(ROOT, r, "func_scores.csv")
            if not os.path.exists(sc):
                subprocess.run([PY, "src/functional_eval.py"], cwd=ROOT, check=True, env={**os.environ, "THS_MURIL_RUN": os.path.join(ROOT, r),
                    "THS_REPORT_OUT": os.path.join(ROOT, "results", "sweep", "func_report.md"), "THS_SCORES_OUT": sc})
            g = pd.read_csv(sc); h = g[g.confidence == "high"]
            ok = lambda p: ((h[p] >= .5) == (h.expected == "abusive")).mean()
            rows.append(dict(name=n, seed=seed, f1=f("p_abusive"), ens=f("p_ens"), func_model=ok("p_muril"), func_ens=ok("p_ens")))
            probs[(n, seed)] = d["p_abusive"]
            probs[("tfidf", 0)] = d["p_tfidf"]; probs[("y", 0)] = d.label
    R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "results", "sweep", "summary.csv"), index=False)
    A = R.groupby("name").agg(n=("f1", "size"), f1=("f1", "mean"), f1_sd=("f1", "std"), ens=("ens", "mean"), ens_sd=("ens", "std"),
                              func_ens=("func_ens", "mean"), func_model=("func_model", "mean"))
    print(R.round(4).to_string(index=False)); print(A.round(4).to_string())
    # paired bootstrap vs MuRIL baseline seed 42 (model-alone macro-F1), same resamples
    y = probs[("y", 0)].values; n = len(y); idx = np.random.default_rng(0).integers(0, n, size=(10000, n), dtype=np.int32)
    bf = lambda p: np.concatenate([macro_f1_counts(y[idx[s:s + 500]], p[idx[s:s + 500]]) for s in range(0, 10000, 500)])
    base = bf((probs[("muril_lp(base)", 42)].reindex(probs[("y", 0)].index).values >= .5).astype(int))
    tfp = probs[("tfidf", 0)]; b0 = probs[("muril_lp(base)", 42)]
    baseens = bf((((b0 + tfp) / 2).reindex(probs[("y", 0)].index).values >= .5).astype(int))
    for (nm, s), p in probs.items():
        if nm in ("y", "tfidf") or (nm, s) == ("muril_lp(base)", 42): continue
        for tag, pv in (("model", p), ("ens", (p + probs[("tfidf", 0)]) / 2)):  # ens uses the last-scored tfidf probs (same TF-IDF for every run)
            dd = bf((pv.values >= .5).astype(int)) - (base if tag == 'model' else baseens); lo, hi = np.percentile(dd, [2.5, 97.5])
            print(f"boot {nm} s{s} {tag} vs muril base s42 (same kind: model-alone or ensemble): delta {dd.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "results", "sweep"), exist_ok=True)
    a = sys.argv[1] if len(sys.argv) > 1 else "train"
    if a == "train":
        for n in CFG: train(n)
    elif a == "seeds":
        for n in sys.argv[2:]:
            for s in (1, 2): train(n, s)
    elif a == "report": report(sys.argv[2:])
