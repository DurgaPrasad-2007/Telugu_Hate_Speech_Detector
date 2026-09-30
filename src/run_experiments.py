"""
Experiment orchestrator: runs the full comparison matrix for the SCI paper
by shelling out to train.py for each configuration, then aggregates results.

Covers:
  - Baseline full fine-tuning: mBERT, IndicBERT, MuRIL, XLM-R
  - PEFT sweep on the strongest baseline backbone: LoRA, QLoRA, AdaLoRA, DoRA, IA3
  - Multi-seed runs for statistical significance
  - Ablations: LoRA rank, quantization level, target modules
  - Cross-domain transfer: telugu <-> codemixed

Run everything:
  python src/run_experiments.py --stage all

Run a specific stage:
  python src/run_experiments.py --stage baselines
  python src/run_experiments.py --stage peft
  python src/run_experiments.py --stage seeds
  python src/run_experiments.py --stage ablation_rank
  python src/run_experiments.py --stage ablation_quant
  python src/run_experiments.py --stage ablation_targets
  python src/run_experiments.py --stage cross_domain

This is designed to run on Colab / a GPU box. It is intentionally
sequential (one process at a time) to keep GPU memory accounting clean
per run; edit MODELS/SEEDS below to trim scope if compute is limited.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(ROOT, "src", "train.py")

BASELINE_MODELS = {
    "mBERT": "bert-base-multilingual-cased",
    "IndicBERT": "ai4bharat/indic-bert",
    "MuRIL": "google/muril-base-cased",
    "XLM-R": "xlm-roberta-base",
}

# The backbone used for the PEFT method comparison (pick the strongest/most
# standard multilingual encoder; swap after baselines finish if a different
# model wins).
PEFT_BACKBONE = "xlm-roberta-base"
PEFT_METHODS = ["lora", "qlora", "adalora", "dora", "ia3"]

SEEDS = [42, 123, 2024, 3407, 999]


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def stage_baselines(domain, epochs):
    for name, ckpt in BASELINE_MODELS.items():
        run([sys.executable, TRAIN_PY,
             "--model", ckpt, "--method", "full", "--domain", domain,
             "--epochs", str(epochs),
             "--run_name", f"baseline__{name}__{domain}"])


def stage_peft(domain, epochs):
    for method in PEFT_METHODS:
        run([sys.executable, TRAIN_PY,
             "--model", PEFT_BACKBONE, "--method", method, "--domain", domain,
             "--epochs", str(epochs),
             "--run_name", f"peft__{method}__{domain}"])
    # full FT on the same backbone, for a fair like-for-like comparison
    run([sys.executable, TRAIN_PY,
         "--model", PEFT_BACKBONE, "--method", "full", "--domain", domain,
         "--epochs", str(epochs),
         "--run_name", f"peft__fullft__{domain}"])


def stage_seeds(domain, epochs, methods=("full", "lora", "qlora")):
    for method in methods:
        for seed in SEEDS:
            run([sys.executable, TRAIN_PY,
                 "--model", PEFT_BACKBONE, "--method", method, "--domain", domain,
                 "--epochs", str(epochs), "--seed", str(seed),
                 "--run_name", f"seedstudy__{method}__{domain}__seed{seed}"])


def stage_ablation_rank(domain, epochs):
    for r in [4, 8, 16, 32]:
        run([sys.executable, TRAIN_PY,
             "--model", PEFT_BACKBONE, "--method", "lora", "--domain", domain,
             "--epochs", str(epochs), "--lora_r", str(r), "--lora_alpha", str(r * 2),
             "--run_name", f"ablation_rank__lora_r{r}__{domain}"])


def stage_ablation_quant(domain, epochs):
    configs = [
        ("full", {}),
        ("lora", {}),
        ("qlora", {}),
    ]
    for method, extra in configs:
        run([sys.executable, TRAIN_PY,
             "--model", PEFT_BACKBONE, "--method", method, "--domain", domain,
             "--epochs", str(epochs),
             "--run_name", f"ablation_quant__{method}__{domain}"])


def stage_ablation_targets(domain, epochs):
    target_sets = {
        "qv": "query,value",
        "qkvo": "query,key,value,output.dense",
    }
    for label, targets in target_sets.items():
        run([sys.executable, TRAIN_PY,
             "--model", PEFT_BACKBONE, "--method", "lora", "--domain", domain,
             "--epochs", str(epochs), "--target_modules", targets,
             "--run_name", f"ablation_targets__{label}__{domain}"])


def stage_cross_domain(epochs):
    for method in ["full", "lora", "qlora"]:
        # train telugu, test codemixed
        run([sys.executable, TRAIN_PY,
             "--model", PEFT_BACKBONE, "--method", method, "--domain", "telugu",
             "--eval_domain", "codemixed", "--epochs", str(epochs),
             "--run_name", f"crossdomain__{method}__telugu_to_codemixed"])
        # train codemixed, test telugu
        run([sys.executable, TRAIN_PY,
             "--model", PEFT_BACKBONE, "--method", method, "--domain", "codemixed",
             "--eval_domain", "telugu", "--epochs", str(epochs),
             "--run_name", f"crossdomain__{method}__codemixed_to_telugu"])


def aggregate_results(output_root="results", out_csv="results/all_results.csv"):
    import pandas as pd
    rows = []
    for metrics_path in glob.glob(os.path.join(output_root, "*", "metrics.json")):
        with open(metrics_path) as f:
            rows.append(json.load(f))
    if not rows:
        print("No results found yet.")
        return
    df = pd.json_normalize(rows)
    df.to_csv(out_csv, index=False)
    print(f"Aggregated {len(rows)} runs -> {out_csv}")
    print(df[["run_name", "model", "method", "domain", "test_accuracy", "test_macro_f1",
              "trainable_pct", "training_time_sec", "peak_gpu_memory_mb"]].to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                    choices=["baselines", "peft", "seeds", "ablation_rank", "ablation_quant",
                             "ablation_targets", "cross_domain", "aggregate", "all"])
    p.add_argument("--domain", default="telugu", choices=["telugu", "codemixed", "combined"])
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()

    if args.stage in ("baselines", "all"):
        stage_baselines(args.domain, args.epochs)
    if args.stage in ("peft", "all"):
        stage_peft(args.domain, args.epochs)
    if args.stage in ("seeds", "all"):
        stage_seeds(args.domain, args.epochs)
    if args.stage in ("ablation_rank", "all"):
        stage_ablation_rank(args.domain, args.epochs)
    if args.stage in ("ablation_quant", "all"):
        stage_ablation_quant(args.domain, args.epochs)
    if args.stage in ("ablation_targets", "all"):
        stage_ablation_targets(args.domain, args.epochs)
    if args.stage in ("cross_domain", "all"):
        stage_cross_domain(args.epochs)
    if args.stage in ("aggregate", "all"):
        aggregate_results()


if __name__ == "__main__":
    main()
