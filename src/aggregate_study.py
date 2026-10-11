"""
Aggregate results/study/study_results.csv into results/study/summary.md.

Usage:
  python src/aggregate_study.py

Selection is done on validation macro-F1 only; test macro-F1 is reported
for reference and is never used to pick a configuration.
"""
import csv
import json
import os
import statistics

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STUDY_DIR = os.path.join(ROOT, "results", "study")
CSV_PATH = os.path.join(STUDY_DIR, "study_results.csv")
OUT_PATH = os.path.join(STUDY_DIR, "summary.md")
PROF_CSV_PATH = os.path.join(STUDY_DIR, "prof_table.csv")

MURIL, XLMR, MBERT = "MuRIL", "xlm-roberta-base", "bert-base-multilingual-cased"
INDIC = "ai4bharat/IndicBERTv2-MLM-only"
SEEDS = [42, 123, 2024]
LORA_LR, FULL_LR = 2e-4, 2e-5


def backbone(model):
    m = model.lower()
    if "muril" in m:
        return MURIL
    if "xlm-roberta" in m:
        return XLMR
    if "bert-base-multilingual" in m:
        return MBERT
    return model


def load():
    rows = []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "run": r["run_name"],
                "method": r["method"],
                "backbone": backbone(r["model"]),
                "lr": round(float(r["lr"]), 10),
                "epochs": int(r["epochs"]),
                "lora_r": int(r["lora_r"]),
                "lora_alpha": int(r["lora_alpha"]),
                "seed": int(r["seed"]),
                "val": float(r["val_macro_f1"]),
                "test": float(r["test_macro_f1"]),
                "params": int(float(r["trainable_params"])),
                "pct": float(r["trainable_pct"]),
                "time": float(r["training_time_sec"]),
                "mem": float(r["peak_gpu_memory_mb"]),
            })
    return rows


def where(rows, **kw):
    if "lr" in kw:
        kw["lr"] = round(kw["lr"], 10)
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def one(rows, **kw):
    hits = where(rows, **kw)
    return hits[0] if hits else None


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def mean_std(vals):
    if not vals:
        return "-"
    m = statistics.mean(vals)
    if len(vals) < 2:
        return f"{m:.4f} ± n/a"
    return f"{m:.4f} ± {statistics.stdev(vals):.4f}"  # sample std, ddof=1


def safe_div(a, b):
    return a / b if b else 0.0


def f1_of(p, r):
    return safe_div(2 * p * r, p + r)


def run_metrics(run):
    """Metrics recomputed from metrics.json's confusion matrix; None if the run has no metrics.json."""
    d = os.path.join(STUDY_DIR, run)
    path = os.path.join(d, "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    (tn, fp), (fn, tp) = m["confusion_matrix"]  # rows = actual, class 1 = Abusive
    n = tn + fp + fn + tp
    p1, r1 = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    p0, r0 = safe_div(tn, tn + fn), safe_div(tn, tn + fp)
    f1_1, f1_0 = f1_of(p1, r1), f1_of(p0, r0)
    s0, s1 = tn + fp, fn + tp
    size = None
    for name in ("adapter_model.safetensors", "model.safetensors"):
        p = os.path.join(d, "model", name)
        if os.path.isfile(p):
            size = os.path.getsize(p) / (1024 * 1024)
            break
    return {
        "acc": safe_div(tn + tp, n), "p1": p1, "r1": r1, "f1_1": f1_1, "f1_0": f1_0,
        "macro": (f1_0 + f1_1) / 2, "wf1": safe_div(f1_0 * s0 + f1_1 * s1, n),
        "params": int(m["trainable_params"]), "pct": float(m["trainable_pct"]),
        "time": float(m["training_time_sec"]), "mem": float(m["peak_gpu_memory_mb"]),
        "size": size, "cm": [[tn, fp], [fn, tp]],
    }


def group_runs(rows, **kw):
    out = []
    for r in where(rows, **kw):
        m = run_metrics(r["run"])
        if m is None:
            continue
        m.update(run=r["run"], seed=r["seed"], val=r["val"], rank=r["lora_r"], alpha=r["lora_alpha"],
                 method=r["method"], backbone=r["backbone"], lr=r["lr"], epochs=r["epochs"])
        out.append(m)
    return sorted(out, key=lambda m: m["seed"])


def ms(vals):
    if not vals:
        return "-"
    m = statistics.mean(vals)
    if len(vals) < 2:
        return f"{m:.4f} ± -"
    return f"{m:.4f} ± {statistics.stdev(vals):.4f}"


def avg(vals, spec):
    return format(statistics.mean(vals), spec) if vals else "-"


PROF_HEADERS = ["Exp", "Setup", "n", "Seeds", "Test acc", "Abusive P", "Abusive R", "Abusive F1",
                "Non-abusive F1", "Macro-F1", "Weighted-F1", "Val macro-F1", "Trainable params",
                "Trainable %", "Train time (s)", "Peak GPU (MB)", "Size (MB)"]
PROF_METRICS = ["acc", "p1", "r1", "f1_1", "f1_0", "macro", "wf1", "val"]
PROF_CSV_FIELDS = ["group", "run_name", "method", "backbone", "lr", "epochs", "lora_r", "lora_alpha", "seed",
                   "accuracy", "abusive_precision", "abusive_recall", "abusive_f1", "nonabusive_f1",
                   "macro_f1", "weighted_f1", "val_macro_f1", "trainable_params", "trainable_pct",
                   "train_time_sec", "peak_gpu_mb", "model_size_mb", "tn", "fp", "fn", "tp"]
LORA_KW = dict(lr=LORA_LR, epochs=10, lora_r=8, lora_alpha=64)
FULL_KW = dict(lr=FULL_LR, epochs=5)
E_SPECS = [
    ("E1", "FacebookAI/xlm-roberta-base full", dict(method="full", backbone=XLMR, **FULL_KW)),
    ("E2", "xlm-roberta-base LoRA", dict(method="lora", backbone=XLMR, **LORA_KW)),
    ("E4", "xlm-roberta-base AdaLoRA", dict(method="adalora", backbone=XLMR, **LORA_KW)),
    ("E5", "xlm-roberta-base DoRA", dict(method="dora", backbone=XLMR, **LORA_KW)),
    ("E6", "bert-base-multilingual-cased LoRA", dict(method="lora", backbone=MBERT, **LORA_KW)),
    ("E7", "IndicBERTv2-MLM-only LoRA", dict(method="lora", backbone=INDIC, **LORA_KW)),
]
MURIL_SPECS = [
    ("MuRIL full", "MuRIL full", dict(method="full", backbone=MURIL, **FULL_KW)),
    ("MuRIL LoRA", "MuRIL LoRA", dict(method="lora", backbone=MURIL, **LORA_KW)),
    ("MuRIL AdaLoRA", "MuRIL AdaLoRA", dict(method="adalora", backbone=MURIL, **LORA_KW)),
    ("MuRIL DoRA", "MuRIL DoRA", dict(method="dora", backbone=MURIL, **LORA_KW)),
]


def prof_row(label, desc, runs):
    if not runs:
        return [label, desc, 0, "-"] + ["not run"] * 13
    cells = [ms([r[k] for r in runs]) for k in PROF_METRICS]
    cells += [avg([r["params"] for r in runs], ",.0f"), avg([r["pct"] for r in runs], ".3f"),
              avg([r["time"] for r in runs], ".1f"), avg([r["mem"] for r in runs], ".0f"),
              avg([r["size"] for r in runs if r["size"] is not None], ".2f")]
    return [label, desc, len(runs), ",".join(str(r["seed"]) for r in runs)] + cells


def csv_row(group, m):
    lora = m["method"] != "full"  # full runs carry a placeholder rank/alpha in the CSV
    return {
        "group": group, "run_name": m["run"], "method": m["method"], "backbone": m["backbone"],
        "lr": m["lr"], "epochs": m["epochs"], "lora_r": m["rank"] if lora else "",
        "lora_alpha": m["alpha"] if lora else "", "seed": m["seed"],
        "accuracy": m["acc"], "abusive_precision": m["p1"], "abusive_recall": m["r1"],
        "abusive_f1": m["f1_1"], "nonabusive_f1": m["f1_0"], "macro_f1": m["macro"],
        "weighted_f1": m["wf1"], "val_macro_f1": m["val"], "trainable_params": m["params"],
        "trainable_pct": m["pct"], "train_time_sec": m["time"], "peak_gpu_mb": m["mem"],
        "model_size_mb": "" if m["size"] is None else f"{m['size']:.2f}",
        "tn": m["cm"][0][0], "fp": m["cm"][0][1], "fn": m["cm"][1][0], "tp": m["cm"][1][1],
    }


def cm_text(label, desc, m):
    if m is None:
        return f"- **{label}** {desc}: not run"
    return f"- **{label}** {desc} (seed 42, `{m['run']}`): `{m['cm']}`"


def prof_section(rows):
    csv_out, out = [], []
    out += ["## (f) Professor's E1-E7 framework", "",
            "Metrics are recomputed from each run's `metrics.json` confusion matrix (class 1 = Abusive); "
            "val macro-F1 comes from `study_results.csv`. Cells are mean ± sample std (ddof=1); "
            "std shows `-` when n < 2. Experiments with no completed run show `not run`.", ""]

    body, cms = [], []
    out += ["### (f1) Experiments E1-E7 (LoRA family: r8 a64 lr 2e-4 e10; full: lr 2e-5 e5)", ""]
    for label, desc, spec in E_SPECS:
        runs = group_runs(rows, **spec)
        body.append(prof_row(label, desc, runs))
        csv_out += [csv_row(label, m) for m in runs]
        s42 = [m for m in runs if m["seed"] == 42]
        cms.append(cm_text(label, desc, s42[0] if s42 else None))
    out += [table(PROF_HEADERS, body), ""]

    out += ["### (f2) MuRIL block (same definitions)", ""]
    body = []
    for label, desc, spec in MURIL_SPECS:
        runs = group_runs(rows, **spec)
        body.append(prof_row(label, desc, runs))
        csv_out += [csv_row(label, m) for m in runs]
        s42 = [m for m in runs if m["seed"] == 42]
        cms.append(cm_text(label, desc, s42[0] if s42 else None))
    out += [table(PROF_HEADERS, body), ""]

    out += ["### (f3) E3 rank study (MuRIL LoRA, lr 2e-4, e10, seed 42; best alpha per rank by val)", ""]
    e3 = group_runs(rows, method="lora", backbone=MURIL, lr=LORA_LR, epochs=10, seed=42)
    csv_out += [csv_row("E3 rank", m) for m in e3]
    body, cms3 = [], []
    for rk in sorted({m["rank"] for m in e3}):
        best = max((m for m in e3 if m["rank"] == rk), key=lambda m: m["val"])
        body.append([f"r{rk}", f"a{best['alpha']}", f"{best['val']:.4f}", f"{best['macro']:.4f}",
                     f"{best['params']:,}", f"{best['pct']:.3f}", f"{best['time']:.1f}"])
        cms3.append(cm_text(f"E3 r{rk} a{best['alpha']}", "MuRIL LoRA", best))
    if not body:
        body = [["not run"] * 7]
    out += [table(["Rank", "Best alpha (by val)", "Val macro-F1", "Test macro-F1", "Trainable params",
                   "Trainable %", "Train time (s)"], body), ""]

    out += ["### (f4) Per-run confusion matrices, seed 42 "
            "(rows = actual [non-abusive, abusive]; cols = predicted; `[[TN, FP], [FN, TP]]`)", ""]
    out += cms + [""] + cms3 + [""]

    skipped = sorted(d for d in os.listdir(STUDY_DIR)
                     if os.path.isdir(os.path.join(STUDY_DIR, d))
                     and not os.path.isfile(os.path.join(STUDY_DIR, d, "metrics.json")))
    if skipped:
        out += [f"Run folders without `metrics.json` (skipped): {', '.join(skipped)}", ""]
    return out, csv_out


def write_prof_csv(csv_out):
    with open(PROF_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROF_CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_out)


def main():
    rows = load()
    out = ["# Study summary", "",
           "Source: `results/study/study_results.csv`. Model selection used **validation macro-F1 only**; "
           "test macro-F1 is shown for reporting and was not used to choose any configuration.", ""]

    # (a) rank x alpha pivot, MuRIL LoRA lr 2e-4 seed 42
    base_a = dict(method="lora", backbone=MURIL, lr=LORA_LR, seed=42)
    ranks = sorted({r["lora_r"] for r in where(rows, **base_a)})
    alphas = sorted({r["lora_alpha"] for r in where(rows, **base_a)})
    out += ["## (a) LoRA rank x alpha (MuRIL, lr 2e-4, seed 42)", ""]
    for key, name in [("val", "Validation macro-F1"), ("test", "Test macro-F1 (reporting only)")]:
        body = []
        for rk in ranks:
            cells = []
            for al in alphas:
                r = one(rows, lora_r=rk, lora_alpha=al, **base_a)
                cells.append(fmt(r[key] if r else None))
            body.append([f"r{rk}"] + cells)
        out += [f"**{name}**", "", table(["rank \\ alpha"] + [f"a{a}" for a in alphas], body), ""]

    # (b) learning-rate table, MuRIL LoRA r8 a64 seed 42
    out += ["## (b) Learning rate (MuRIL, LoRA r8 a64, seed 42)", ""]
    lr_rows = where(rows, method="lora", backbone=MURIL, lora_r=8, lora_alpha=64, seed=42)
    body = [[f"{r['lr']:g}", r["epochs"], fmt(r["val"]), fmt(r["test"]), f"{r['time']:.0f}"]
            for r in sorted(lr_rows, key=lambda r: r["lr"])]
    out += [table(["LR", "Epochs", "Val F1", "Test F1 (reporting)", "Train time (s)"], body), ""]

    # (c) method comparison, MuRIL r8 a64 seed 42
    out += ["## (c) Method comparison (MuRIL, r8 a64, seed 42)", ""]
    body = []
    for method, lr in [("lora", LORA_LR), ("dora", LORA_LR), ("adalora", LORA_LR), ("full", FULL_LR)]:
        r = one(rows, method=method, backbone=MURIL, lora_r=8, lora_alpha=64, seed=42, lr=lr)
        if r is None:
            body.append([method, f"{lr:g}"] + ["-"] * 7)
            continue
        body.append([method, f"{lr:g}", r["epochs"], fmt(r["val"]), fmt(r["test"]),
                     f"{r['params']:,}", f"{r['pct']:.3f}", f"{r['time']:.0f}", f"{r['mem']:.0f}"])
    out += [table(["Method", "LR", "Epochs", "Val F1", "Test F1 (reporting)", "Trainable params",
                   "Trainable %", "Train time (s)", "Peak GPU (MB)"], body), ""]

    # (d) backbone table, LoRA and DoRA, r8 a64 lr 2e-4 seed 42
    out += ["## (d) Backbone (r8 a64, lr 2e-4, seed 42)", ""]
    body = []
    for method in ["lora", "dora"]:
        for bb in [MURIL, XLMR, MBERT]:
            r = one(rows, method=method, backbone=bb, lora_r=8, lora_alpha=64, lr=LORA_LR, seed=42)
            if r is None:
                body.append([method, bb, "-", "-", "-", "-"])
            else:
                body.append([method, bb, fmt(r["val"]), fmt(r["test"]), f"{r['params']:,}", f"{r['time']:.0f}"])
    out += [table(["Method", "Backbone", "Val F1", "Test F1 (reporting)", "Trainable params",
                   "Train time (s)"], body), ""]

    # (e) multi-seed, MuRIL lora / dora / full
    out += ["## (e) Multi-seed (MuRIL; mean ± sample std, ddof=1)", ""]
    body = []
    for method, lr in [("lora", LORA_LR), ("dora", LORA_LR), ("full", FULL_LR)]:
        runs = [one(rows, method=method, backbone=MURIL, lora_r=8, lora_alpha=64, lr=lr, seed=s)
                for s in SEEDS]
        runs = [r for r in runs if r is not None]
        seeds_present = ",".join(str(r["seed"]) for r in runs) or "-"
        body.append([method, f"{lr:g}", len(runs), seeds_present,
                     mean_std([r["val"] for r in runs]), mean_std([r["test"] for r in runs])])
    out += [table(["Method", "LR", "n", "Seeds present", "Val F1 mean ± std",
                   "Test F1 mean ± std (reporting)"], body), ""]
    out += [f"Seeds requested: {', '.join(map(str, SEEDS))}. `n` is the number of seeds found in the CSV.", ""]

    prof_lines, prof_csv = prof_section(rows)
    out += prof_lines

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    write_prof_csv(prof_csv)
    print(f"wrote {os.path.abspath(OUT_PATH)}")
    print(f"wrote {os.path.abspath(PROF_CSV_PATH)} ({len(prof_csv)} rows)")


if __name__ == "__main__":
    main()
