"""Final figures -> docs/figures/ (dpi 160, Okabe-Ito palette).

Inputs: results/study/significance.json, results/study/selection.json,
        results/study/baseline_classical.json, results/final_best/export_metrics.json,
        results/study/<run>/model adapter/weights files (for sizes).

Run: .venv\\Scripts\\python -I src\\make_final_figures.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
STUDY = os.path.join(ROOT, "results", "study")
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
SELECTED = OI["vermillion"]
OTHER = OI["blue"]
BASE = OI["orange"]
GREY = "#7f7f7f"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def jload(p):
    with open(os.path.join(p), encoding="utf-8") as f:
        return json.load(f)


sig = jload(os.path.join(STUDY, "significance.json"))
sel = jload(os.path.join(STUDY, "selection.json"))
base = jload(os.path.join(STUDY, "baseline_classical.json"))
export = jload(os.path.join(ROOT, "results", "final_best", "export_metrics.json"))
ranking = sel["ranking"]


def size_mb(path):
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else float("nan")


# ---------------------------------------------------------------- (1) forest
def short_label(c):
    return c["comparison"]


comps = [c for c in sig["comparisons"] if "ci95" in c]
fig, ax = plt.subplots(figsize=(9, 0.55 * len(comps) + 2.2))
ys = np.arange(len(comps))[::-1]
for y, c in zip(ys, comps):
    d = c["delta_macro_f1_A_minus_other"]
    lo, hi = c["ci95"]
    col = OTHER if (lo <= 0 <= hi) else (SELECTED if lo > 0 else OI["green"])
    ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ecolor=col,
                capsize=3, markersize=6, linewidth=1.5)
    ax.text(1.01, y, f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={c['mcnemar_p']:.3g}",
            transform=ax.get_yaxis_transform(), va="center", fontsize=8)
ax.axvline(0, color="black", linewidth=1)
ax.set_yticks(ys)
ax.set_yticklabels([short_label(c) for c in comps], fontsize=9)
ax.set_xlabel("Macro-F1 difference (selected model minus comparator), 95% paired bootstrap CI\n"
              "right of zero = selected better, left of zero = comparator better")
ax.set_title(f"Forest plot: selected XLM-R LoRA vs each comparator (TEST, n={sig['test_n']}, "
             f"10,000 resamples)", fontsize=11, loc="left")
ax.set_xlim(-0.06, 0.06)
ax.grid(axis="x", color="#e5e5e5", linewidth=0.7)
ax.set_axisbelow(True)
fig.subplots_adjust(left=0.36, right=0.72)
fig.text(0.735, 0.97, "delta [95% CI]   McNemar p", fontsize=8, va="top")
fig.savefig(os.path.join(FIG, "final_forest.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- (2) frontier
def cfg_label(r):
    m = r["model"].split("/")[-1].replace("xlm-roberta-base", "XLM-R").replace(
        "muril-base-cased", "MuRIL").replace("bert-base-multilingual-cased", "mBERT").replace(
        "IndicBERTv2-MLM-only", "IndicBERTv2")
    if r["method"] == "full":
        return f"{m} full"
    return f"{m} {r['method'].upper()} r{r['lora_r']}/e{r['epochs']}"


sel_key = "xlm-roberta-base / lora / lr0.0002 / e10 / r8a64"
tfidf_f1 = base["test"]["macro_f1"]
tfidf_val = base["val"]["macro_f1"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, xkey, xlabel, log in [
    (axes[0], "trainable_params_mean", "Trainable parameters (log scale)", True),
    (axes[1], "peak_gpu_mb_mean", "Peak GPU memory (MB, mean over 3 seeds)", False),
]:
    for r in ranking:
        x = r[xkey]
        y = r["test_mean"]
        is_sel = r["config"] == sel_key
        col = SELECTED if is_sel else OTHER
        ax.errorbar(x, y, yerr=r["test_std"], fmt="o", color=col, ecolor=col, capsize=3,
                    markersize=10 if is_sel else 6, zorder=3 if is_sel else 2)
    ax.axhline(tfidf_f1, color=OI["black"], linestyle="--", linewidth=1)
    ax.text(0.99, tfidf_f1, f" TF-IDF baseline {tfidf_f1:.4f}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8)
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log")
    ax.grid(color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Test macro-F1 (mean ± SD, 3 seeds)")
axes[0].set_ylim(0.88, 0.915)
# label points (left panel only, to avoid clutter)
label_offsets = {  # small manual offsets to avoid overlaps
    "xlm-roberta-base / full / lr2e-05 / e5 / full": (8, 4),
    "xlm-roberta-base / lora / lr0.0002 / e10 / r8a64": (10, -22),
    "muril-base-cased / full / lr2e-05 / e5 / full": (8, 4),
    "xlm-roberta-base / dora / lr0.0002 / e10 / r8a64": (-4, 14),
    "muril-base-cased / lora / lr0.0002 / e10 / r8a64": (-6, 12),
    "bert-base-multilingual-cased / lora / lr0.0002 / e10 / r8a64": (-14, -14),
    "IndicBERTv2-MLM-only / lora / lr0.0002 / e10 / r8a64": (10, -4),
    "muril-base-cased / dora / lr0.0002 / e10 / r8a64": (10, 6),
    "xlm-roberta-base / lora / lr0.0002 / e5 / r4a32": (-10, -16),
    "xlm-roberta-base / lora / lr0.0002 / e5 / r16a32": (8, 2),
    "muril-base-cased / adalora / lr0.0002 / e10 / r8a64": (12, -14),
    "xlm-roberta-base / adalora / lr0.0002 / e10 / r8a64": (-6, -16),
}
key_lines = []
for i, r in enumerate(ranking, start=1):
    name = cfg_label(r) + (" (selected)" if r["config"] == sel_key else "")
    key_lines.append(f"{i}. {name}")
    axes[0].annotate(str(i), (r["trainable_params_mean"], r["test_mean"]), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8,
                     color=SELECTED if r["config"] == sel_key else "black")
fig.text(0.01, -0.02, "Point numbers: " + "   ".join(key_lines[:6]) + "\n"
         + "   ".join(key_lines[6:]), fontsize=7.5, va="top")
axes[0].set_title("Trainable parameters vs test macro-F1", loc="left", fontsize=10)
axes[1].set_title("Peak GPU memory vs test macro-F1", loc="left", fontsize=10)
fig.suptitle("Accuracy-efficiency frontier: 12 configs (3 seeds each), selected = XLM-R LoRA r8 a64 (orange-red); "
             "dashed = TF-IDF", fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0.06, 1, 0.94])
fig.savefig(os.path.join(FIG, "final_frontier.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- (3) scorecard
pr = sig["per_run_test"]


def recall_mean(names):
    vals = [pr[n]["abusive_recall"] for n in names]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def row_for(key):
    return next(r for r in ranking if r["config"] == key)


full_r = row_for("xlm-roberta-base / full / lr2e-05 / e5 / full")
lora_r = row_for(sel_key)
dora_r = row_for("xlm-roberta-base / dora / lr0.0002 / e10 / r8a64")

full_rec = recall_mean(["B_xlmr_full_s2024", "B_xlmr_full_s42", "B_xlmr_full_s123"])
lora_rec = recall_mean(["A_xlmr_lora_s2024", "X_xlmr_lora_s42", "X_xlmr_lora_s123"])
dora_rec = recall_mean(["C_xlmr_dora_s2024", "X_xlmr_dora_s42", "X_xlmr_dora_s123"])

study_dir = lambda d: os.path.join(STUDY, d, "model")
full_mb = size_mb(os.path.join(study_dir("xlm-roberta-base_full_lr2e-5_e5_s2024"), "model.safetensors"))
lora_mb = size_mb(os.path.join(study_dir("xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024"), "adapter_model.safetensors"))
dora_mb = size_mb(os.path.join(study_dir("xlm-roberta-base_dora_r8_a64_lr2e-4_e10_s2024"), "adapter_model.safetensors"))
int8_mb = export["int8_onnx_size_mb"]

def pct(v):
    return f"{100 * v:.1f}%"

cols = ["Model / method", "Test macro-F1\n(mean ± SD, 3 seeds)", "Abusive recall\n(mean over 3 seeds)",
        "Trainable params", "Peak GPU (MB)", "Adapter / weights size (MB)", "INT8 size (MB)"]
rows = [
    ["TF-IDF baseline (LinearSVC, CPU)", f"{base['test']['macro_f1']:.4f} (1 run)",
     pct(base["test"]["abusive_recall"]), "n/a (sparse, CPU)", "n/a (CPU)",
     f"{base['model_size_mb_joblib']:.1f} (joblib, whole model)", "n/a"],
    ["XLM-R full fine-tune", f"{full_r['test_mean']:.4f} ± {full_r['test_std']:.4f}", pct(full_rec[0]),
     f"{full_r['trainable_params_mean'] / 1e6:.1f} M (100%)", f"{full_r['peak_gpu_mb_mean']:.0f}",
     f"{full_mb:.0f} (full checkpoint)", "n/a"],
    ["XLM-R LoRA r8 a64 (SELECTED)", f"{lora_r['test_mean']:.4f} ± {lora_r['test_std']:.4f}", pct(lora_rec[0]),
     f"{lora_r['trainable_params_mean'] / 1e6:.3f} M ({100 * lora_r['trainable_params_mean'] / 278045186:.2f}%)",
     f"{lora_r['peak_gpu_mb_mean']:.0f}", f"{lora_mb:.1f} (adapter)", f"{int8_mb:.1f} (ONNX)"],
    ["XLM-R DoRA r8 a64", f"{dora_r['test_mean']:.4f} ± {dora_r['test_std']:.4f}", pct(dora_rec[0]),
     f"{dora_r['trainable_params_mean'] / 1e6:.3f} M ({100 * dora_r['trainable_params_mean'] / 278045186:.2f}%)",
     f"{dora_r['peak_gpu_mb_mean']:.0f}", f"{dora_mb:.1f} (adapter)", "n/a"],
]

fig, ax = plt.subplots(figsize=(18, 3.8))
ax.axis("off")
col_w = [0.19, 0.17, 0.12, 0.16, 0.09, 0.17, 0.10]  # fractions of axes width, sums to 1.0
tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colLoc="center",
               colWidths=col_w)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 2.1)
for (r_, c_), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r_ == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")
    elif r_ == 3:  # selected row
        cell.set_facecolor("#fbe3d2")
    if c_ == 0 and r_ > 0:
        cell._loc = "left"
        cell.set_text_props(ha="left")
ax.set_title("Final scorecard (TEST set, n=2,997): macro-F1, abusive recall, size and memory per model",
             loc="left", fontsize=12, pad=12)
fig.text(0.01, 0.02, "Peak GPU = peak_gpu_mb_mean from selection.json. Adapter size for full FT is the full "
         "checkpoint. INT8 = ONNX quantized model from results/final_best/export_metrics.json.",
         fontsize=8, color=GREY)
fig.savefig(os.path.join(FIG, "final_scorecard.png"), dpi=160, bbox_inches="tight")
plt.close(fig)
print("figures written:", sorted(f for f in os.listdir(FIG) if f.startswith("final_")))
