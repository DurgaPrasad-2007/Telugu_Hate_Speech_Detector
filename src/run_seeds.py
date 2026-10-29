"""Seed study: MuRIL lp8/lr5e-5 on data/clean and data/clean_aug (seeds 1,2 trained; 42 = existing runs).
Trains (GPU), dumps preds on ORIGINAL clean val/test (CPU), then reports macro-F1, ensemble-with-TFIDF macro-F1
and functional pass rates mean+-sd per condition. Skips steps whose outputs exist.
usage: python src/run_seeds.py [train|report|all]"""
import os, sys, subprocess, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
COND = {"clean": ("data/clean", "results/clean_muril-base-cased_lp8_lr5e-5", "models/tfidf/tfidf_best.joblib"),
        "aug": ("data/clean_aug", "results/aug_muril", "results/aug_tfidf/aug.joblib")}
SEEDS = (1, 2, 42)
run_dir = lambda c, s: COND[c][1] if s == 42 else f"results/seeds/{c}_s{s}"


def sh(cmd, **env):
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, **{k: str(v) for k, v in env.items()}})


def train():
    for c in COND:
        for s in (1, 2):
            r = run_dir(c, s)
            if not os.path.exists(os.path.join(ROOT, r, "model")):
                sh([PY, "src/train.py", "--model", "google/muril-base-cased", "--method", "lora", "--domain", "telugu",
                    "--target-modules", "all-linear", "--loraplus-ratio", "8", "--lr", "5e-5", "--epochs", "4",
                    "--train_bs", "32", "--max_length", "128", "--fp16", "--early_stop_patience", "2", "--seed", str(s),
                    "--output_root", "results/seeds", "--run_name", f"{c}_s{s}"], THS_DATA_DIR=COND[c][0])
            if not os.path.exists(os.path.join(ROOT, r, "preds.csv")):  # original clean val/test for both conditions
                sh([PY, "src/dump_probs.py", r, "128"], THS_DATA_DIR="data/clean")


def report():
    os.environ["THS_DATA_DIR"] = os.path.join(ROOT, "data", "clean")
    sys.path.insert(0, HERE)
    from negation_compare import scored
    from sklearn.metrics import f1_score
    rows, func = [], []
    for c in COND:
        for s in SEEDS:
            pr = os.path.join(ROOT, run_dir(c, s), "preds.csv")
            d = scored(os.path.join(ROOT, COND[c][2]), pr)
            y = d.label.values; f = lambda col: f1_score(y, (d[col] >= .5).astype(int), average="macro")
            rows.append(dict(cond=c, seed=s, muril=f("p_abusive"), ens=f("p_ens"), tfidf=f("p_tfidf")))
            sc = os.path.join(ROOT, "results", "seeds", f"func_{c}_s{s}.csv")
            if not os.path.exists(sc):
                sh([PY, "src/functional_eval.py"], THS_DATA_DIR="data/clean", THS_MURIL_RUN=os.path.join(ROOT, run_dir(c, s)),
                   THS_REPORT_OUT=os.path.join(ROOT, "results", "seeds", "func_report.md"), THS_SCORES_OUT=sc)
            g = pd.read_csv(sc); h = g[g.confidence == "high"]; ok = lambda p: (h[p] >= .5) == (h.expected == "abusive")
            ent = dict(cond=c, seed=s, overall_ens=ok("p_ens").mean(), overall_tfidf=ok("p_tfidf").mean())
            for fn, grp in h.assign(o=ok("p_ens")).groupby("functionality"): ent[fn] = grp.o.mean()
            func.append(ent)
    R, F = pd.DataFrame(rows), pd.DataFrame(func)
    R.to_csv(os.path.join(ROOT, "results", "seeds", "f1_by_seed.csv"), index=False); F.to_csv(os.path.join(ROOT, "results", "seeds", "func_by_seed.csv"), index=False)
    print(R.round(4).to_string(index=False))
    print(R.groupby("cond")[["muril", "ens", "tfidf"]].agg(["mean", "std"]).round(4).to_string())
    print(F.groupby("cond").mean(numeric_only=True).drop(columns="seed").T.round(3).to_string())
    print(F.groupby("cond")[["overall_ens"]].agg(["mean", "std"]).round(4).to_string())
    print(F[["cond", "seed", "overall_ens", "negated_hate", "threat"]].round(3).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(os.path.join(ROOT, "results", "seeds"), exist_ok=True)
    if a in ("train", "all"): train()
    if a in ("report", "all"): report()
