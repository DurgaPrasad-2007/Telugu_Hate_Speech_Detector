"""Make the PEFT study figures in docs/figures/ from results/study/study_results.csv.

Colour-blind-safe Okabe-Ito palette, no seaborn. Run: .venv\\Scripts\\python -I src\\make_figures.py
"""
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "study"
FIG = ROOT / "docs" / "figures"
DPI = 160

MURIL = "google/muril-base-cased"
XLMR = "FacebookAI/xlm-roberta-base"
INDIC = "ai4bharat/IndicBERTv2-MLM-only"
MBERT = "bert-base-multilingual-cased"
BACKBONE_NAME = {MURIL: "MuRIL", XLMR: "XLM-R", INDIC: "IndicBERTv2-MLM-only", MBERT: "mBERT"}
METHODS = ["full", "lora", "dora", "adalora"]
METHOD_NAME = {"full": "Full FT", "lora": "LoRA", "dora": "DoRA", "adalora": "AdaLoRA"}
METHOD_COLOR = {"full": "#000000", "lora": "#0072B2", "dora": "#E69F00", "adalora": "#009E73"}
BACKBONE_COLOR = {MURIL: "#0072B2", XLMR: "#D55E00", INDIC: "#009E73", MBERT: "#CC79A7"}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})


def load():
    with open(STUDY / "study_results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("lr", "val_macro_f1", "test_macro_f1", "trainable_params", "peak_gpu_memory_mb",
                  "training_time_sec"):
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
        r["epochs"] = int(r["epochs"])
        r["lora_r"] = int(r["lora_r"])
        r["lora_alpha"] = int(r["lora_alpha"])
    return rows


def main_config(r, model, method):
    """Main setting per method: full uses its own lr/epochs; PEFT uses lr 2e-4, r8, a64."""
    if r["model"] != model or r["method"] != method:
        return False
    if method == "full":
        return True
    return r["lr"] == 2e-4 and r["lora_r"] == 8 and r["lora_alpha"] == 64


def mean_sd(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)
    print("wrote", FIG / name)


def fig_rank_alpha_heatmap(rows):
    ranks, alphas = [4, 8, 16, 32], [16, 32, 64]
    grid = [[None] * len(alphas) for _ in ranks]
    for r in rows:
        if (r["model"] == MURIL and r["method"] == "lora" and r["lr"] == 2e-4 and r["seed"] == 42
                and r["lora_r"] in ranks and r["lora_alpha"] in alphas):
            grid[ranks.index(r["lora_r"])][alphas.index(r["lora_alpha"])] = r["val_macro_f1"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vals = [v for row in grid for v in row if v is not None]
    im = ax.imshow(grid, cmap="cividis", vmin=min(vals) - 0.002, vmax=max(vals) + 0.002)
    for i, rk in enumerate(ranks):
        for j, al in enumerate(alphas):
            v = grid[i][j]
            txt = "n/a" if v is None else f"{v:.4f}"
            color = "white" if v is not None and v < (min(vals) + max(vals)) / 2 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(len(alphas)), [f"α={a}" for a in alphas])
    ax.set_yticks(range(len(ranks)), [f"r={r}" for r in ranks])
    ax.set_xlabel("LoRA alpha (α, scaling numerator)")
    ax.set_ylabel("LoRA rank (r)")
    ax.set_title("Validation macro-F1 (unitless): MuRIL LoRA rank × alpha\n(lr 2e-4, 10 epochs, seed 42)")
    fig.colorbar(im, ax=ax, label="Validation macro-F1 (unitless, 0-1)")
    save(fig, "rank_alpha_heatmap.png")


def fig_method_comparison(rows):
    backbones = [MURIL, XLMR]
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for bi, bb in enumerate(backbones):
        means, sds = [], []
        for m in METHODS:
            vals = [r["test_macro_f1"] for r in rows if main_config(r, bb, m)]
            mu, sd = mean_sd(vals)
            means.append(mu)
            sds.append(sd)
        xs = [i + (bi - 0.5) * width for i in range(len(METHODS))]
        ax.bar(xs, means, width, yerr=sds, capsize=4, color=BACKBONE_COLOR[bb],
               label=f"{BACKBONE_NAME[bb]} (n=3 seeds)")
        for x, mu in zip(xs, means):
            ax.text(x, 0.851, f"{mu:.3f}", ha="center", va="bottom", fontsize=8, rotation=90, color="white")
    ax.set_ylim(0.85, 0.93)
    ax.set_xticks(range(len(METHODS)), [METHOD_NAME[m] for m in METHODS])
    ax.set_ylabel("Test macro-F1 (unitless; mean ± SD over 3 seeds)\ny-axis starts at 0.85, not 0")
    ax.set_title("Test macro-F1 by fine-tuning method and backbone (mean ± SD, 3 seeds)")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "method_comparison.png")


def fig_efficiency_scatter(rows):
    points = []
    for bb in [MURIL, XLMR, INDIC, MBERT]:
        for m in METHODS:
            rs = [r for r in rows if r["model"] == bb and r["method"] == m and main_config(r, bb, m)]
            if len(rs) < 3:
                continue
            points.append((bb, m, rs[0]["trainable_params"],
                           statistics.mean(r["val_macro_f1"] for r in rs)))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for bb, m, params, val in points:
        ax.scatter(params, val, s=110, color=METHOD_COLOR[m], marker="o" if bb == MURIL else
                   ("s" if bb == XLMR else ("^" if bb == INDIC else "D")),
                   edgecolor="black", linewidth=0.6, zorder=3)
        # stagger labels that share the same x (e.g. three 296k-param LoRA runs)
        # the three 296k-param LoRA points sit within 0.003 of each other: put mBERT on the left
        if bb == MBERT or (bb == MURIL and m == "lora"):
            offset, ha = (0, 9), "center"
        elif bb == INDIC:
            offset, ha = (12, -12), "left"
        else:
            offset, ha = (10, -4), "left"
        ax.annotate(f"{BACKBONE_NAME[bb]} {METHOD_NAME[m]}", (params, val), xytext=offset,
                    textcoords="offset points", fontsize=8, ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(1.5e5, 6e8)
    ax.set_xlabel("Trainable parameters (count, log scale)")
    ax.set_ylabel("Mean validation macro-F1 (unitless; 3 seeds)")
    ax.set_title("Parameter efficiency: validation macro-F1 vs trainable parameters\n"
                 "marker = backbone (circle MuRIL, square XLM-R, triangle IndicBERT, diamond mBERT)",
                 fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=METHOD_COLOR[m], label=METHOD_NAME[m])
               for m in METHODS]
    ax.legend(handles=handles, title="Method (colour)", frameon=False, loc="center right")
    ax.grid(alpha=0.3, zorder=0)
    save(fig, "efficiency_scatter.png")


def fig_memory_time(rows):
    means_mem, sd_mem, means_t, sd_t = [], [], [], []
    for m in METHODS:
        rs = [r for r in rows if main_config(r, MURIL, m)]
        mm, ms = mean_sd([r["peak_gpu_memory_mb"] for r in rs])
        tm, ts = mean_sd([r["training_time_sec"] for r in rs])
        means_mem.append(mm); sd_mem.append(ms); means_t.append(tm); sd_t.append(ts)
    colors = [METHOD_COLOR[m] for m in METHODS]
    names = [METHOD_NAME[m] for m in METHODS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.3))
    a1.bar(names, means_mem, yerr=sd_mem, capsize=4, color=colors)
    a1.set_ylabel("Peak GPU memory (MB)")
    a1.set_title("Peak GPU memory (MB), MuRIL, mean ± SD over seeds")
    a2.bar(names, means_t, yerr=sd_t, capsize=4, color=colors)
    a2.set_ylabel("Training time (s)")
    a2.set_title("Training time (s), MuRIL, mean ± SD over seeds")
    for ax, vals in ((a1, means_mem), (a2, means_t)):
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.text(0.5, 0.01, "Note: training times were measured with other GPU load present, so they are "
             "indicative, not a clean benchmark.", ha="center", fontsize=8, style="italic")
    save(fig, "memory_time.png")


def fig_backbone_comparison(rows):
    bbs = [MURIL, XLMR, INDIC, MBERT]
    means, sds = [], []
    for bb in bbs:
        rs = [r["test_macro_f1"] for r in rows if main_config(r, bb, "lora")]
        mu, sd = mean_sd(rs)
        means.append(mu); sds.append(sd)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [BACKBONE_NAME[b] for b in bbs]
    ax.bar(names, means, yerr=sds, capsize=4, color=[BACKBONE_COLOR[b] for b in bbs])
    for i, mu in enumerate(means):
        ax.text(i, 0.851, f"{mu:.4f}", ha="center", va="bottom", fontsize=8, color="white", rotation=90)
    ax.set_ylim(0.85, 0.92)
    ax.set_ylabel("Test macro-F1 (unitless; mean ± SD, 3 seeds)\ny-axis starts at 0.85, not 0")
    ax.set_title("LoRA (r8, α64, lr 2e-4) test macro-F1 by backbone (mean ± SD, 3 seeds)")
    save(fig, "backbone_comparison.png")


def fig_lr_ablation(rows):
    lrs = [1e-4, 2e-4, 5e-4]
    pick = {r["lr"]: r for r in rows if r["model"] == MURIL and r["method"] == "lora"
            and r["lora_r"] == 8 and r["lora_alpha"] == 64 and r["seed"] == 42 and r["epochs"] == 10}
    xs = list(range(len(lrs)))
    val = [pick[lr]["val_macro_f1"] for lr in lrs]
    test = [pick[lr]["test_macro_f1"] for lr in lrs]
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(xs, val, marker="o", color="#0072B2", label="Validation macro-F1")
    ax.plot(xs, test, marker="s", color="#D55E00", linestyle="--", label="Test macro-F1 (reporting only)")
    for x, v, t in zip(xs, val, test):
        ax.annotate(f"{v:.4f}", (x, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{t:.4f}", (x, t), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(xs, [f"{lr:g}" for lr in lrs])
    ax.set_xlabel("Learning rate (unitless; 10 epochs, cosine schedule)")
    ax.set_ylabel("Macro-F1 (unitless, 0-1)")
    ax.set_title("MuRIL LoRA (r8, α64, seed 42): macro-F1 vs learning rate")
    ax.set_ylim(0.82, 0.93)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "lr_ablation.png")


def main():
    rows = load()
    fig_rank_alpha_heatmap(rows)
    fig_method_comparison(rows)
    fig_efficiency_scatter(rows)
    fig_memory_time(rows)
    fig_backbone_comparison(rows)
    fig_lr_ablation(rows)


if __name__ == "__main__":
    main()
