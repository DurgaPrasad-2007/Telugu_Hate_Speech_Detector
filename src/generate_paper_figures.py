"""Paper figures (PDF + 300 dpi PNG) in docs/figures/, plotted from real results.

Run: .venv\\Scripts\\python.exe src/generate_paper_figures.py
Sources are named per figure. Numbers copied from committed reports are marked
"report" (ROBUSTNESS.md, POLICY_REPORT.md, SEEDS_RELABEL_REPORT.md,
PEFT_STUDY_REPORT.md 7b); everything else is read from results/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colorblind-safe palette
BLUE, ORANGE, GREEN, VERMIL, SKY, PURPLE, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#7F7F7F")
COL1, COL2 = 3.25, 6.8  # inches: ACL/IEEE 1- and 2-column widths

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def fig1():
    # single seed: report 2.1 / SEEDS (seed 42). 3 seeds: MuRIL & ensemble from
    # results/seeds/f1_by_seed.csv (clean); XLM-R from PEFT report 7b (E2, seeds 42/123/2024,
    # LoRA r8 a64 lr2e-4 -- a different recipe from the lp8 single-seed run). TF-IDF is deterministic.
    s = pd.read_csv(os.path.join(ROOT, "results", "seeds", "f1_by_seed.csv"))
    c = s[s.cond == "clean"]
    single = {"TF-IDF": 0.8960, "MuRIL": float(c[c.seed == 42].muril.iloc[0]),
              "XLM-R": 0.8944, "Ensemble": float(c[c.seed == 42].ens.iloc[0])}
    multi = {"TF-IDF": (0.8960, 0.0), "MuRIL": (c.muril.mean(), c.muril.std(ddof=1)),
             "XLM-R": (0.9035, 0.0033), "Ensemble": (c.ens.mean(), c.ens.std(ddof=1))}
    names = list(single)
    x, w = np.arange(len(names)), 0.36
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    ax.bar(x - w / 2, [single[n] for n in names], w, color=SKY, label="Single seed (42)")
    m = [multi[n][0] for n in names]
    ax.bar(x + w / 2, m, w, yerr=[multi[n][1] for n in names], capsize=3, color=BLUE,
           error_kw={"lw": 1}, label=r"3 seeds, mean $\pm$ sd")
    for i, n in enumerate(names):
        ax.text(i - w / 2, single[n] + 0.0006, f"{single[n]:.4f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, m[i] + multi[n][1] + 0.0006, f"{m[i]:.4f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, names)
    ax.set_ylim(0.88, 0.915)
    ax.set_ylabel("Test macro-F1")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig1_backbone_f1_comparison")


def fig2():
    # report: ROBUSTNESS.md "Brutal obfuscation stress test", MuRIL, 300 abusive test comments
    attacks = ["Spaced\nletters", "Dots", "Dashes", "Repetition", "Combined"]
    before, after = [1, 15, 2, 54, 55], [66, 99, 98, 92, 90]
    x, w = np.arange(len(attacks)), 0.36
    fig, ax = plt.subplots(figsize=(COL1 * 1.6, 2.6))
    ax.bar(x - w / 2, before, w, color=VERMIL, label="No canonicalization")
    ax.bar(x + w / 2, after, w, color=GREEN, label="With canonicalization")
    for i in range(len(attacks)):
        ax.text(i - w / 2, before[i] + 1.5, f"{before[i]}", ha="center", fontsize=7)
        ax.text(i + w / 2, after[i] + 1.5, f"{after[i]}", ha="center", fontsize=7)
    ax.set_xticks(x, attacks)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Abusive comments still\nblocked or reviewed (%)")
    ax.legend(loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0, 1.12))
    save(fig, "fig2_obfuscation_defense")


def fig3():
    cal = json.load(open(os.path.join(ROOT, "results", "calibration.json")))
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0), sharey=True)
    for ax, key, title, col in ((axes[0], "before", "Before (T = 1)", VERMIL),
                                (axes[1], "after", f"After (T = {cal['temperature']:.3f})", GREEN)):
        bins = [b for b in cal[key]["reliability_bins"] if b["n"]]
        conf, acc, n = (np.array([b[k] for b in bins]) for k in ("conf", "acc", "n"))
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Perfect")
        ax.plot(conf, acc, "o-", color=col, lw=1.5, ms=4 + 0, label="Model")
        for cx, ax_, nn in zip(conf, acc, n):
            ax.annotate(f"n={nn}", (cx, ax_), textcoords="offset points", xytext=(4, -10), fontsize=6.5)
        ax.set_title(f"{title}: ECE = {cal[key]['ece']:.4f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("Fraction of positives")
    save(fig, "fig3_calibration_reliability")


def fig4():
    # (a) report POLICY_REPORT.md conformal rows (test, alpha=2% wrongful blocks)
    rows = {"TF-IDF": [(24.6, 4.72), (15.2, 9.30), (5.3, 18.97)],
            "Transformer": [(23.0, 5.47), (17.0, 8.10), (5.4, 18.59)],
            "Ensemble": [(21.8, 4.57), (13.0, 8.92), (3.2, 19.34)]}
    style = {"TF-IDF": (ORANGE, "s"), "Transformer": (BLUE, "^"), "Ensemble": (GREEN, "o")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.9))
    for k, pts in rows.items():
        col, mk = style[k]
        a.plot(*zip(*pts), marker=mk, color=col, lw=1.2, ms=5, label=k)
    a.set_xlabel("Human review load (% of traffic)")
    a.set_ylabel("Realized abusive miss rate (%)")
    a.set_title(r"(a) Conformal policy, false-block $\approx$ 2%")
    a.legend(frameon=False)
    # (b) results/threshold_sweep.csv: validation sweep of the deployed model; Pareto frontier
    sw = pd.read_csv(os.path.join(ROOT, "results", "threshold_sweep.csv"))
    sw["review"] = sw.review_pct * 100
    sc = b.scatter(sw.review, sw.block_recall * 100, c=sw.block_precision * 100, cmap="viridis",
                   s=7, alpha=0.7, linewidths=0)
    cb = fig.colorbar(sc, ax=b, pad=0.02)
    cb.set_label("Block precision (%)", fontsize=8)
    b.set_xlabel("Human review load (% of traffic)")
    b.set_ylabel("Block recall (%)")
    b.set_title("(b) Threshold sweep (validation)")
    fig.tight_layout()
    save(fig, "fig4_conformal_review_tradeoff")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
    print("wrote figures to", OUT)
