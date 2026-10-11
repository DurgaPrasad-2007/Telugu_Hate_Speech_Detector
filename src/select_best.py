"""Select the best PEFT config by validation macro-F1 only and copy its artifacts to models/best/.

Reads results/study/study_results.csv and results/study/<run>/metrics.json.
Writes results/study/selection.json and results/study/selection.md.
Test macro-F1 is reported but never used for selection.
"""
import csv
import json
import re
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "study"
OUT = ROOT / "models" / "best"
MIN_SEEDS = 3

RULE = (
    "1. Group runs by config key (model, method, lr, epochs, and lora_r+lora_alpha only when method != full). "
    "2. Keep configs with >= 3 seeds. "
    "3. For each config: mean and sample SD (ddof=1) of val and test macro-F1; mean trainable params, peak GPU MB, train time. "
    "4. top = config with highest mean val macro-F1. "
    "5. Tie set = all configs with mean val >= top.mean_val - top.std_val. "
    "6. Winner = tie-set member with the FEWEST trainable params (ties: higher mean val). "
    "7. Winner run = winner config's seed with the highest val macro-F1. "
    "Selection uses validation only; test is never used."
)


def config_key(r):
    full = r["method"] == "full"
    return (
        r["model"],
        r["method"],
        float(r["lr"]),
        int(r["epochs"]),
        None if full else int(r["lora_r"]),
        None if full else int(r["lora_alpha"]),
    )


def label(c):
    model = c["model"].split("/")[-1]
    peft = f"r{c['lora_r']}a{c['lora_alpha']}" if c["method"] != "full" else "full"
    return f"{model} / {c['method']} / lr{c['lr']:g} / e{c['epochs']} / {peft}"


def summarize(runs):
    groups = {}
    for r in runs:
        groups.setdefault(config_key(r), []).append(r)

    kept, excluded = [], []
    for key, rs in groups.items():
        model, method, lr, epochs, r_, a_ = key
        if len(rs) < MIN_SEEDS:
            excluded.append({"config": label({"model": model, "method": method, "lr": lr, "epochs": epochs,
                                              "lora_r": r_, "lora_alpha": a_}), "n_seeds": len(rs),
                             "runs": [x["run_name"] for x in rs]})
            continue
        val = [float(x["val_macro_f1"]) for x in rs]
        test = [float(x["test_macro_f1"]) for x in rs]
        c = {
            "model": model, "method": method, "lr": lr, "epochs": epochs,
            "lora_r": r_, "lora_alpha": a_,
            "n_seeds": len(rs),
            "seeds": sorted(int(x["seed"]) for x in rs),
            "val_mean": statistics.mean(val),
            "val_std": statistics.stdev(val),
            "test_mean": statistics.mean(test),
            "test_std": statistics.stdev(test),
            "trainable_params_mean": statistics.mean(float(x["trainable_params"]) for x in rs),
            "peak_gpu_mb_mean": statistics.mean(float(x["peak_gpu_memory_mb"]) for x in rs),
            "train_time_s_mean": statistics.mean(float(x["training_time_sec"]) for x in rs),
            "runs": rs,
        }
        c["label"] = label(c)
        kept.append(c)
    return sorted(kept, key=lambda c: -c["val_mean"]), excluded


def copy_winner_artifacts(run_name, method):
    src = STUDY / run_name / "model"
    OUT.mkdir(parents=True, exist_ok=True)
    weights = "model.safetensors" if method == "full" else "adapter_model.safetensors"
    cfg = "config.json" if method == "full" else "adapter_config.json"
    required = [weights, cfg]
    optional = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "vocab.txt", "sentencepiece.bpe.model"]
    for name in required:
        assert (src / name).exists(), f"missing {src / name}"
    copied = []
    for name in required + optional:
        if (src / name).exists():
            shutil.copy2(src / name, OUT / name)
            copied.append(name)
    shutil.copy2(STUDY / run_name / "metrics.json", OUT / "metrics.json")
    return copied


def write_markdown(res, path):
    w = res["winner"]
    lines = [
        "# PEFT configuration selection",
        "",
        "## Rule (validation macro-F1 only)",
        "",
        *[f"- {step}" for step in re.split(r"(?<=\.) (?=\d\. )", RULE)],
        "",
        f"- Top config: **{res['top']['label']}** (val {res['top']['val_mean']:.4f} ± {res['top']['val_std']:.4f})",
        f"- Tie threshold: {res['tie_threshold']:.4f}",
        "",
        "## Winner",
        "",
        f"- Config: **{w['label']}**",
        f"- Winner run: `{w['run_name']}`",
        f"- Run val macro-F1: {w['run_val_macro_f1']:.4f}; run test macro-F1: {w['run_test_macro_f1']:.4f}",
        f"- Config val: {w['val_mean']:.4f} ± {w['val_std']:.4f}; test: {w['test_mean']:.4f} ± {w['test_std']:.4f} (n={w['n_seeds']})",
        f"- Trainable params: {w['trainable_params_mean']:,.0f}",
        "",
        "## Ranking (all configs with >= 3 seeds, by mean val macro-F1)",
        "",
        "| Rank | Config | n | Val F1 mean ± SD | Test F1 mean ± SD | Trainable params | Peak GPU MB | Train time s | In tie set |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for i, c in enumerate(res["ranking"], 1):
        mark = "yes" if c["in_tie_set"] else ""
        lines.append(
            f"| {i} | {c['config']} | {c['n_seeds']} | {c['val_mean']:.4f} ± {c['val_std']:.4f} | "
            f"{c['test_mean']:.4f} ± {c['test_std']:.4f} | {c['trainable_params_mean']:,.0f} | "
            f"{c['peak_gpu_mb_mean']:.0f} | {c['train_time_s_mean']:.0f} | {mark} |"
        )
    if res["excluded_configs"]:
        lines += ["", "## Excluded (< 3 seeds)", ""]
        lines += [f"- {e['config']}: {e['n_seeds']} seed(s)" for e in res["excluded_configs"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    with open(STUDY / "study_results.csv", newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))

    ranking, excluded = summarize(runs)
    top = ranking[0]
    threshold = top["val_mean"] - top["val_std"]
    tie = [c for c in ranking if c["val_mean"] >= threshold]
    winner = min(tie, key=lambda c: (c["trainable_params_mean"], -c["val_mean"]))
    best_run = max(winner["runs"], key=lambda r: float(r["val_macro_f1"]))
    run_name = best_run["run_name"]
    metrics = json.loads((STUDY / run_name / "metrics.json").read_text(encoding="utf-8"))

    tie_labels = {c["label"] for c in tie}
    ranking_out = []
    for i, c in enumerate(ranking, 1):
        ranking_out.append({
            "rank": i, "config": c["label"], "model": c["model"], "method": c["method"],
            "lr": c["lr"], "epochs": c["epochs"], "lora_r": c["lora_r"], "lora_alpha": c["lora_alpha"],
            "n_seeds": c["n_seeds"], "seeds": c["seeds"],
            "val_mean": c["val_mean"], "val_std": c["val_std"],
            "test_mean": c["test_mean"], "test_std": c["test_std"],
            "trainable_params_mean": c["trainable_params_mean"],
            "peak_gpu_mb_mean": c["peak_gpu_mb_mean"],
            "train_time_s_mean": c["train_time_s_mean"],
            "in_tie_set": c["label"] in tie_labels,
        })

    res = {
        "rule": RULE,
        "selection_metric": "val_macro_f1 (test never used for selection)",
        "min_seeds": MIN_SEEDS,
        "top": {"label": top["label"], "val_mean": top["val_mean"], "val_std": top["val_std"]},
        "tie_threshold": threshold,
        "tie_set": [c["label"] for c in tie],
        "ranking": ranking_out,
        "excluded_configs": excluded,
        "winner": {
            "label": winner["label"], "model": winner["model"], "method": winner["method"],
            "lr": winner["lr"], "epochs": winner["epochs"],
            "lora_r": winner["lora_r"], "lora_alpha": winner["lora_alpha"],
            "n_seeds": winner["n_seeds"],
            "val_mean": winner["val_mean"], "val_std": winner["val_std"],
            "test_mean": winner["test_mean"], "test_std": winner["test_std"],
            "trainable_params_mean": winner["trainable_params_mean"],
            "run_name": run_name,
            "run_val_macro_f1": float(best_run["val_macro_f1"]),
            "run_test_macro_f1": float(best_run["test_macro_f1"]),
            "run_confusion_matrix": metrics.get("confusion_matrix"),
        },
    }

    (STUDY / "selection.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_markdown(res, STUDY / "selection.md")

    copied = copy_winner_artifacts(run_name, winner["method"])
    shutil.copy2(STUDY / "selection.json", OUT / "selection.json")

    print(f"Winner config: {winner['label']}")
    print(f"Winner run:    {run_name} (val {float(best_run['val_macro_f1']):.4f}, "
          f"test {float(best_run['test_macro_f1']):.4f})")
    print(f"Top config:    {top['label']} (threshold {threshold:.4f})")
    print("Tie set:")
    for c in tie:
        print(f"  - {c['label']}: val {c['val_mean']:.4f} ± {c['val_std']:.4f}, "
              f"params {c['trainable_params_mean']:,.0f}")
    print(f"Copied to {OUT}: {', '.join(copied + ['metrics.json', 'selection.json'])}")


if __name__ == "__main__":
    main()
