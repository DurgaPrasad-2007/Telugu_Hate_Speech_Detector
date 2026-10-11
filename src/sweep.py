"""Sequential sweep driver: python src/sweep.py --plan grid | --raw "<train.py args>"."""
import argparse, csv, json, os, shlex, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "results", "study")
ERRORS, CSV = os.path.join(STUDY, "errors.log"), os.path.join(STUDY, "study_results.csv")
MURIL = "google/muril-base-cased"
COMMON = ["--output_root", "results/study", "--domain", "telugu", "--max_length", "96",
          "--train_bs", "16", "--eval_bs", "64", "--early_stop_patience", "3"]
FIELDS = ["run_name", "method", "model", "lr", "epochs", "lora_r", "lora_alpha", "seed", "val_macro_f1",
          "test_macro_f1", "trainable_params", "trainable_pct", "training_time_sec", "peak_gpu_memory_mb"]
PLANS = {"grid": [["--model", MURIL, "--method", "lora", "--target_modules", "query,value",
                   "--lora_r", str(r), "--lora_alpha", str(a), "--lr", "2e-4", "--epochs", "10",
                   "--seed", "42"] + COMMON for r in (4, 8, 16, 32) for a in (16, 32, 64)]}


def _l(model, method, lr="2e-4", seed=42, ep="10", r=8, a=64):
    # best grid cell by validation macro-F1: r=8, alpha=64
    return ["--model", model, "--method", method, "--target_modules", "query,value", "--lora_r", str(r),
            "--lora_alpha", str(a), "--lr", lr, "--epochs", ep, "--seed", str(seed)] + COMMON


XLMR, MBERT = "FacebookAI/xlm-roberta-base", "bert-base-multilingual-cased"
PLANS["rest"] = (
    [_l(MURIL, "lora", lr=lr) for lr in ("1e-4", "5e-4")]
    + [_l(MURIL, "adalora"), _l(MURIL, "dora"), _l(XLMR, "lora"), _l(MBERT, "lora"), _l(XLMR, "dora")]
    + [_l(MURIL, "full", lr="2e-5", ep="5")]
    + [_l(MURIL, m, lr=("2e-5" if m == "full" else "2e-4"), ep=("5" if m == "full" else "10"), seed=sd)
       for sd in (123, 2024) for m in ("lora", "dora", "full")]
)

# professor's E1-E7 on XLM-R (+ mBERT, IndicBERTv2 LoRA), 3 seeds each; config from the MuRIL val-selected grid
INDIC = "ai4bharat/IndicBERTv2-MLM-only"
PLANS["prof"] = (
    [_l(XLMR, "full", lr="2e-5", ep="5", seed=sd) for sd in (42,)]
    + [_l(XLMR, "adalora")]
    + [_l(XLMR, m, seed=sd) for sd in (123, 2024) for m in ("lora", "dora")]
    + [_l(XLMR, "full", lr="2e-5", ep="5", seed=sd) for sd in (123, 2024)]
    + [_l(INDIC, "lora", seed=sd) for sd in (42, 123, 2024)]
    + [_l(MBERT, "lora", seed=sd) for sd in (123, 2024)]
)

PLANS["extra"] = (
    [_l(XLMR, "adalora", seed=sd) for sd in (123, 2024)]
    + [_l(MURIL, "adalora", seed=sd) for sd in (123, 2024)]
    + [_l(MBERT, "dora")]
)

# Exact configs from the paper (Table 2: r4 a32 do0.1 bs8 len128 ep5) and the professor's
# recommended start (r16 a32 do0.1 bs16 len128 lr2e-4 ep5), on XLM-R, 3 seeds. Later flags override COMMON.
def _paper(tag, r, a, bs, seed):
    return ["--model", XLMR, "--method", "lora", "--target_modules", "query,value", "--lora_r", str(r),
            "--lora_alpha", str(a), "--lr", "2e-4", "--epochs", "5", "--seed", str(seed)] + COMMON + [
            "--max_length", "128", "--train_bs", str(bs), "--lora_dropout", "0.1",
            "--run_name", f"{tag}_xlmr_lora_r{r}_a{a}_s{seed}"]


PLANS["paper"] = ([_paper("paper", 4, 32, 8, sd) for sd in (42, 123, 2024)]
                  + [_paper("prof", 16, 32, 16, sd) for sd in (42, 123, 2024)])


def parse(argv):
    d = {}
    for k, v in zip(argv, argv[1:] + [None]):
        if k.startswith("--"):
            d[k] = None if v is None or v.startswith("--") else v
    return d


def auto_name(argv):
    d = parse(argv)
    parts = [d.get("--method") or "full"]
    if parts[0] in {"lora", "qlora", "dora", "adalora"}:
        parts += [f"r{d.get('--lora_r') or 16}", f"a{d.get('--lora_alpha') or 32}"]
    parts += [f"lr{d.get('--lr') or 'default'}", f"e{d.get('--epochs') or 5}", f"s{d.get('--seed') or 42}"]
    if (d.get("--model") or MURIL) != MURIL:
        parts.insert(0, d["--model"].split("/")[-1])
    return "_".join(parts)


def append_row(name):
    m = json.load(open(os.path.join(STUDY, name, "metrics.json"), encoding="utf-8"))
    hp = m.get("hyperparams", {})
    row = {"run_name": name, "method": m["method"], "model": m["model"], "lr": hp.get("lr"),
           "epochs": hp.get("epochs"), "lora_r": hp.get("lora_r"), "lora_alpha": hp.get("lora_alpha"),
           "seed": m["seed"], **{k: m.get(k) for k in FIELDS[8:]}}
    logged = os.path.exists(CSV) and any(r["run_name"] == name for r in csv.DictReader(open(CSV, encoding="utf-8")))
    if not logged:
        with open(CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if not os.path.exists(CSV) or os.path.getsize(CSV) == 0:
                w.writeheader()
            w.writerow(row)


def run(argv):
    name = parse(argv)["--run_name"]
    metrics = os.path.join(STUDY, name, "metrics.json")
    if not os.path.exists(metrics):
        p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "train.py"), *argv], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode or not os.path.exists(metrics):
            os.makedirs(STUDY, exist_ok=True)
            tail = (p.stdout + "\n" + p.stderr).splitlines()[-30:]
            open(ERRORS, "a", encoding="utf-8").write(f"== {name} (exit {p.returncode})\n" + "\n".join(tail) + "\n")
            print(f"FAILED {name}")
            return
    append_row(name)
    print(f"ok {name}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", choices=sorted(PLANS))
    g.add_argument("--raw")
    a = ap.parse_args()
    for argv in PLANS[a.plan] if a.plan else [shlex.split(a.raw)]:
        argv = argv + ([] if "--output_root" in argv else ["--output_root", "results/study"])
        argv = argv + ([] if "--run_name" in argv else ["--run_name", auto_name(argv)])
        run(argv)
if __name__ == "__main__":
    main()

