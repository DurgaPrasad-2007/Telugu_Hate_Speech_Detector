"""scripts/generate_paper_figures.py
Publication Figure Generation Pipeline for Telugu Hate Speech Detector Research Paper.

Generates all 9 publication-grade figures in dual format:
  - Vector PDF (.pdf) for LaTeX inclusion
  - High-resolution 300+ DPI PNG (.png) for previews & raster distribution

Destination Directories:
  - docs/paper/figures/ (Primary for LaTeX compilation)
  - docs/figures/       (Mirror for repository documentation)

Design Standards:
  - Okabe-Ito Colorblind-Safe Palette
  - IEEEtran Column Widths (Single Column: 3.5 in, Double Column: 7.0 in)
  - Serif Typography (DejaVu Serif / Times New Roman / Computer Modern math)
  - Type 42 TrueType vector fonts for publication fidelity

Usage:
  .venv\\Scripts\\python.exe scripts/generate_paper_figures.py [--fig N] [--all]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

# Root directory resolution
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PAPER = os.path.join(ROOT, "docs", "paper", "figures")
OUT_MIRROR = os.path.join(ROOT, "docs", "figures")
STUDY_DIR = os.path.join(ROOT, "results", "study")
RESULTS_DIR = os.path.join(ROOT, "results")

os.makedirs(OUT_PAPER, exist_ok=True)
os.makedirs(OUT_MIRROR, exist_ok=True)

# ---------------------------------------------------------------------------
# Okabe-Ito Colorblind-Safe Palette
# ---------------------------------------------------------------------------
OI = {
    "vermillion": "#D55E00",  # Selected model / Key highlight / Abusive
    "blue":       "#0072B2",  # Comparators / Transformers / Precision
    "green":      "#009E73",  # Guarded / High robustness / Calibrated
    "amber":      "#E69F00",  # TF-IDF baseline / Warning / Review
    "sky":        "#56B4E9",  # Accent / MuRIL variant
    "purple":     "#CC79A7",  # Additional backbone (IndicBERT)
    "grey":       "#7F7F7F",  # Unguarded / Secondary / Neutral
    "light_grey": "#F3F4F6",  # Box fills / Backgrounds
    "border_grey":"#9CA3AF",  # Node borders
    "black":      "#111827",  # Text / Foreground
}

# Standard IEEE Column Widths (Inches)
COL1_W = 3.50   # Single-column float (figure)
COL2_W = 7.00   # Double-column float (figure*)

# Global Academic Plotting Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 8.8,
    "axes.labelsize": 9.2,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.color": "#D1D5DB",
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save_figure(fig, primary_name, alias_names=None):
    """Save figure to both PDF and PNG in primary and mirror directories with aliases."""
    names = [primary_name]
    if alias_names:
        names.extend(alias_names)
    
    for name in names:
        for out_dir in (OUT_PAPER, OUT_MIRROR):
            pdf_path = os.path.join(out_dir, f"{name}.pdf")
            png_path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(pdf_path, format="pdf")
            fig.savefig(png_path, format="png", dpi=300)
    
    print(f"  [SAVED] {primary_name}.pdf & .png -> docs/paper/figures/ & docs/figures/")
    if alias_names:
        print(f"          (Aliases: {', '.join(alias_names)})")
    plt.close(fig)


# ===========================================================================
# Figure 1: LoRA Architecture Schematic (fig_lora_schematic)
# ===========================================================================
def generate_fig1_lora():
    """Figure 1: LoRA Low-Rank Decomposition forward pass.
    
    Illustrates forward pass h = W_0 x + (alpha/r) B A x.
    Shows frozen base weight W_0 vs trainable low-rank adapters A and B,
    scaling factor, and summation junction.
    """
    fig, ax = plt.subplots(figsize=(COL1_W, 3.2))
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-0.25, 4.2)
    ax.axis("off")

    # Input Node
    bbox_in = dict(boxstyle="round,pad=0.25,rounding_size=0.15", facecolor="#E5E7EB", edgecolor="#4B5563", lw=1.2)
    ax.text(1.75, 3.85, r"Input $\mathbf{x} \in \mathbb{R}^{k}$", ha="center", va="center", fontsize=9.2, weight="bold", bbox=bbox_in)

    # Base Model (Frozen) - Left Branch
    bbox_base = dict(boxstyle="round,pad=0.35,rounding_size=0.18", facecolor="#F3F4F6", edgecolor="#6B7280", lw=1.4)
    ax.text(0.70, 2.50, r"$\mathbf{W}_0 \in \mathbb{R}^{d \times k}$" + "\nFrozen Base\n(278M params)",
            ha="center", va="center", fontsize=8.0, bbox=bbox_base, color="#1F2937")

    # Adapter A (Trainable) - Right Branch Top
    bbox_a = dict(boxstyle="round,pad=0.25,rounding_size=0.15", facecolor="#EFF6FF", edgecolor=OI["blue"], lw=1.4)
    ax.text(2.80, 2.90, r"$\mathbf{A} \in \mathbb{R}^{r \times k}$" + "\nTrainable\n" + r"$\mathcal{N}(0, \sigma^2)$ init",
            ha="center", va="center", fontsize=7.8, bbox=bbox_a, color="#1E3A8A")

    # Adapter B (Trainable) - Right Branch Bottom
    bbox_b = dict(boxstyle="round,pad=0.25,rounding_size=0.15", facecolor="#EFF6FF", edgecolor=OI["blue"], lw=1.4)
    ax.text(2.80, 1.90, r"$\mathbf{B} \in \mathbb{R}^{d \times r}$" + "\nTrainable\nZero init",
            ha="center", va="center", fontsize=7.8, bbox=bbox_b, color="#1E3A8A")

    # Scaling block
    bbox_scale = dict(boxstyle="round,pad=0.18,rounding_size=0.1", facecolor="#FEF3C7", edgecolor=OI["amber"], lw=1.2)
    ax.text(2.80, 1.15, r"$\times \frac{\alpha}{r} = 8.0$", ha="center", va="center", fontsize=7.8, weight="bold",
            bbox=bbox_scale, color="#92400E")

    # Summation Junction (Circle with +)
    circle = patches.Circle((1.45, 1.15), 0.17, facecolor="#FFFFFF", edgecolor=OI["black"], lw=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(1.45, 1.15, "+", ha="center", va="center", fontsize=12, weight="bold", color=OI["black"], zorder=6)

    # Output Node
    bbox_out = dict(boxstyle="round,pad=0.25,rounding_size=0.15", facecolor="#F0FDF4", edgecolor=OI["green"], lw=1.5)
    ax.text(1.35, 0.25, r"Output $\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B}\mathbf{A}\mathbf{x}$" + "\n" + r"$\mathbf{h} \in \mathbb{R}^{d}$",
            ha="center", va="center", fontsize=7.8, weight="bold", bbox=bbox_out, color="#065F46")

    # Selected LoRA Config Badge (Cleanly placed in open lower right quadrant)
    spec_txt = "Selected LoRA:\n" + r"$r=8, \alpha=64$" + "\n887k params (0.32%)"
    ax.text(3.40, 0.35, spec_txt, ha="center", va="center", fontsize=6.8, color="#1E3A8A",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F9FF", edgecolor="#BAE6FD", lw=0.8))

    # Connective Arrows
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=10, lw=1.2, color="#374151")
    
    # Input to Base
    ax.annotate("", xy=(0.70, 3.05), xytext=(1.4, 3.65), arrowprops=dict(connectionstyle="arc3,rad=-0.12", **arrow_kw))
    # Input to A
    ax.annotate("", xy=(2.80, 3.30), xytext=(2.1, 3.65), arrowprops=dict(connectionstyle="arc3,rad=0.12", **arrow_kw))
    # A to B
    ax.annotate("", xy=(2.80, 2.30), xytext=(2.80, 2.50), arrowprops=arrow_kw)
    # B to Scaling
    ax.annotate("", xy=(2.80, 1.35), xytext=(2.80, 1.50), arrowprops=arrow_kw)
    # Scaling to Sum
    ax.annotate("", xy=(1.65, 1.15), xytext=(2.40, 1.15), arrowprops=arrow_kw)
    # Base to Sum (Unobstructed smooth curve)
    ax.annotate("", xy=(1.25, 1.15), xytext=(0.70, 1.95), arrowprops=dict(connectionstyle="arc3,rad=0.15", **arrow_kw))
    # Sum to Output
    ax.annotate("", xy=(1.35, 0.52), xytext=(1.45, 0.98), arrowprops=arrow_kw)

    save_figure(fig, "fig_lora_schematic", ["fig1_lora_architecture"])


# ===========================================================================
# Figure 2: Pipeline Architecture (fig_pipeline_architecture)
# ===========================================================================
def generate_fig2_pipeline():
    """Figure 2: Two-tier pipeline architecture.
    
    Top tier: Offline experiment, training & selection pipeline.
    Bottom tier: Production inference pipeline with Dravidian input guard,
    temperature scaling, and dual threshold moderation policy.
    """
    fig, ax = plt.subplots(figsize=(COL2_W, 3.4))
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-0.2, 4.6)
    ax.axis("off")

    # Tier 1 Label (Top)
    ax.text(0.1, 4.35, "Track I: Offline Training, Tuning & Model Selection",
            fontsize=9.0, weight="bold", color="#B45309", ha="left")

    # Track 1 Blocks (Experiment & Selection: 6 nodes spaced across 0.2 to 10.2)
    t1_y = 3.55
    t1_nodes = [
        ("Telugu MACD\n29.9k samples", 0.90, 1.35),
        ("Deduplication\n& Leak Check", 2.60, 1.35),
        ("3-Seed Train\n4 PEFT types", 4.30, 1.35),
        ("Validation\nMacro-F1", 6.00, 1.35),
        ("Selection\nMin params", 7.70, 1.35),
        ("LoRA Export\n3.39 MB", 9.40, 1.35),
    ]

    for label, x, w in t1_nodes:
        box = patches.FancyBboxPatch((x - w/2, t1_y - 0.38), w, 0.76,
                                     boxstyle="round,pad=0.04,rounding_size=0.10",
                                     facecolor="#FEF3C7", edgecolor="#D97706", lw=1.1)
        ax.add_patch(box)
        ax.text(x, t1_y, label, ha="center", va="center", fontsize=6.8, color="#78350F")

    # Track 1 Forward Arrows
    arrow_t1 = dict(arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#B45309")
    for i in range(len(t1_nodes) - 1):
        x_start = t1_nodes[i][1] + t1_nodes[i][2]/2
        x_end = t1_nodes[i+1][1] - t1_nodes[i+1][2]/2
        ax.annotate("", xy=(x_end, t1_y), xytext=(x_start, t1_y), arrowprops=arrow_t1)

    # Tier 2 Label (Bottom)
    ax.text(0.1, 2.30, "Track II: Guarded Production Inference",
            fontsize=9.0, weight="bold", color="#1E40AF", ha="left")

    # Track 2 Blocks (Production Inference: 7 nodes + decisions)
    t2_y = 1.25
    t2_nodes = [
        ("Raw Text\nComment", 0.70, 1.05, "#F3F4F6", "#6B7280", "#1F2937"),
        ("Dravidian\nGuard", 1.95, 1.15, "#E0F2FE", "#0284C7", "#0369A1"),
        ("Tokenizer\nSubword", 3.25, 1.10, "#F3F4F6", "#6B7280", "#1F2937"),
        ("XLM-R Base\n+ LoRA", 4.65, 1.35, "#EFF6FF", OI["blue"], "#1E3A8A"),
        ("Linear Head\nClass logits", 6.05, 1.10, "#F3F4F6", "#6B7280", "#1F2937"),
        ("Temp Scaling\n" + r"$T = 1.334$", 7.40, 1.15, "#FEF2F2", OI["vermillion"], "#991B1B"),
        ("Thresholds\n0.45 / 0.50", 8.75, 1.10, "#FEF3C7", OI["amber"], "#92400E"),
    ]

    for label, x, w, bg, ec, tc in t2_nodes:
        box = patches.FancyBboxPatch((x - w/2, t2_y - 0.38), w, 0.76,
                                     boxstyle="round,pad=0.04,rounding_size=0.10",
                                     facecolor=bg, edgecolor=ec, lw=1.2)
        ax.add_patch(box)
        ax.text(x, t2_y, label, ha="center", va="center", fontsize=6.2, color=tc)

    # Track 2 Forward Arrows
    arrow_t2 = dict(arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#374151")
    for i in range(len(t2_nodes) - 1):
        x_start = t2_nodes[i][1] + t2_nodes[i][2]/2
        x_end = t2_nodes[i+1][1] - t2_nodes[i+1][2]/2
        ax.annotate("", xy=(x_end, t2_y), xytext=(x_start, t2_y), arrowprops=arrow_t2)

    # Final Decision Output Blocks (3-way action)
    dec_x = 10.25
    decisions = [
        ("Block\n" + r"$P \geq 0.50$", 1.70, "#FEE2E2", "#DC2626", "#991B1B"),
        ("Review\n" + r"$[0.45, 0.50)$", 1.25, "#FEF3C7", "#D97706", "#78350F"),
        ("Allow\n" + r"$P < 0.45$", 0.80, "#DCFCE7", "#16A34A", "#14532D"),
    ]
    for lbl, y, bg, ec, tc in decisions:
        box = patches.FancyBboxPatch((dec_x - 0.50, y - 0.17), 1.00, 0.34,
                                     boxstyle="round,pad=0.03,rounding_size=0.06",
                                     facecolor=bg, edgecolor=ec, lw=1.0)
        ax.add_patch(box)
        ax.text(dec_x, y, lbl, ha="center", va="center", fontsize=5.6, weight="bold", color=tc)

    # Arrows from Thresholds to Decisions
    th_out = t2_nodes[-1][1] + t2_nodes[-1][2]/2
    for _, y, _, ec, _ in decisions:
        ax.annotate("", xy=(dec_x - 0.50, y), xytext=(th_out, t2_y),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=7, lw=0.9, color=ec))

    # Cross-Tier Adapter Attachment Arrow (Dashed connecting line)
    ad_x = t1_nodes[-1][1]
    enc_x = t2_nodes[3][1]
    ax.annotate("", xy=(enc_x, t2_y + 0.40), xytext=(ad_x, t1_y - 0.40),
                arrowprops=dict(arrowstyle="-|>", linestyle="--", lw=1.4,
                                color=OI["vermillion"], mutation_scale=10,
                                connectionstyle="arc3,rad=-0.14"))
    ax.text(7.15, 2.60, "Runtime Adapter Attachment (3.39 MB)",
            fontsize=6.8, weight="bold", color=OI["vermillion"], ha="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFF", edgecolor=OI["vermillion"], lw=0.8))

    save_figure(fig, "fig_pipeline_architecture", ["fig2_system_architecture"])


# ===========================================================================
# Figure 3: Parameter Efficiency vs Macro-F1 (fig_pareto_frontier)
# ===========================================================================
def generate_fig3_frontier():
    """Figure 3: Parameter efficiency vs Test Macro-F1 scatter (log scale X).
    
    Shows XLM-R, MuRIL, IndicBERT v2, mBERT across Full FT, LoRA, DoRA, AdaLoRA,
    and TF-IDF Linear SVM baseline.
    """
    fig, ax = plt.subplots(figsize=(COL1_W, 2.85))

    # Empirical Data (mean ± SD over 3 seeds from results/study/selection.json)
    data = [
        # (Label, Params, Mean F1, SD, Category, Marker, Color)
        ("XLM-R LoRA (Selected)", 887042, 0.9035, 0.0033, "selected", "*", OI["vermillion"]),
        ("XLM-R Full FT", 278045186, 0.9057, 0.0033, "full", "s", OI["blue"]),
        ("MuRIL Full FT", 237557762, 0.9049, 0.0028, "full", "s", OI["sky"]),
        ("XLM-R DoRA", 905474, 0.9055, 0.0009, "dora", "^", OI["green"]),
        ("MuRIL LoRA", 296450, 0.9052, 0.0030, "lora", "o", OI["sky"]),
        ("IndicBERTv2 LoRA", 296450, 0.9017, 0.0039, "lora", "o", OI["purple"]),
        ("MuRIL DoRA", 314882, 0.9005, 0.0030, "dora", "^", OI["sky"]),
        ("Kaunang LoRA (r4)", 739586, 0.8985, 0.0018, "other", "v", "#8B5CF6"),
        ("XLM-R LoRA (r16)", 1181954, 0.8953, 0.0033, "other", "v", "#EC4899"),
        ("mBERT LoRA", 296450, 0.8946, 0.0015, "lora", "o", OI["grey"]),
        ("MuRIL AdaLoRA", 444194, 0.8945, 0.0003, "adalora", "D", OI["sky"]),
        ("XLM-R AdaLoRA", 1034786, 0.8891, 0.0013, "adalora", "D", OI["grey"]),
    ]

    # Baseline TF-IDF
    tfidf_f1 = 0.9029
    ax.axhline(tfidf_f1, color=OI["amber"], linestyle="--", lw=1.2, zorder=2,
               label=f"TF-IDF ({tfidf_f1:.4f})")

    # Plot Points
    for name, params, f1, sd, cat, mk, col in data:
        if cat == "selected":
            ax.errorbar(params, f1, yerr=sd, fmt=mk, color=col, ecolor=col,
                        markersize=11, capsize=3.5, elinewidth=1.6, zorder=5,
                        label="Selected (XLM-R LoRA)")
        else:
            ms = 6.2 if cat == "full" else 5.2
            ax.errorbar(params, f1, yerr=sd, fmt=mk, color=col, ecolor=col,
                        markersize=ms, capsize=2.2, elinewidth=1.1, alpha=0.9, zorder=4)

    # Selective Annotations to Avoid Overlap
    ax.annotate("XLM-R LoRA\n(Selected: 887k)",
                xy=(887042, 0.9035), xytext=(2200000, 0.9080),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=OI["vermillion"]),
                fontsize=7.0, color=OI["vermillion"], weight="bold", ha="left")

    ax.annotate("XLM-R Full\n(278M)", xy=(278045186, 0.9057), xytext=(130000000, 0.9095),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=OI["blue"]),
                fontsize=6.8, color=OI["blue"], ha="center")

    ax.annotate("MuRIL LoRA\n(296k)", xy=(296450, 0.9052), xytext=(300000, 0.9105),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#0284C7"),
                fontsize=6.8, color="#0284C7", ha="center")

    ax.annotate("AdaLoRA (Degraded)", xy=(1034786, 0.8891), xytext=(1034786, 0.8835),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=OI["grey"]),
                fontsize=6.6, color="#4B5563", ha="center")

    ax.set_xscale("log")
    ax.set_xlim(1.2e5, 4.5e8)
    ax.set_ylim(0.881, 0.916)
    ax.set_xlabel("Trainable Parameters (Log Scale)")
    ax.set_ylabel(r"Test Macro-F1 (Mean $\pm$ SD, 3 Seeds)")

    # Custom Clean Legend
    legend_elements = [
        mlines.Line2D([], [], color=OI["vermillion"], marker="*", ls="", ms=9, label="Selected (XLM-R LoRA)"),
        mlines.Line2D([], [], color=OI["blue"], marker="s", ls="", ms=5.5, label="Full Fine-Tuning"),
        mlines.Line2D([], [], color=OI["green"], marker="^", ls="", ms=5.5, label=r"DoRA ($r{=}8$)"),
        mlines.Line2D([], [], color=OI["sky"], marker="o", ls="", ms=5.5, label="MuRIL LoRA"),
        mlines.Line2D([], [], color=OI["amber"], ls="--", lw=1.2, label=f"TF-IDF ({tfidf_f1:.4f})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.92, fontsize=6.8,
              handletextpad=0.25, borderpad=0.25, labelspacing=0.2)

    save_figure(fig, "fig_pareto_frontier", ["fig3_parameter_efficiency_frontier"])


# ===========================================================================
# Figure 4: Computational Cost Dual Bar (fig_computational_cost)
# ===========================================================================
def generate_fig4_cost():
    """Figure 4: Computational and memory cost on XLM-R.
    
    Left panel: Peak GPU memory (GB) across Full, LoRA, DoRA, AdaLoRA.
    Right panel: Training time (seconds) across methods.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL1_W, 2.4))

    methods = ["Full FT", "LoRA", "DoRA", "AdaLoRA"]
    mem_gb  = [5.353, 1.753, 1.893, 1.760]   # Peak GPU Memory in GB
    time_s  = [1564.1, 774.4, 1125.1, 1617.7] # Training time in seconds

    colors = [OI["blue"], OI["vermillion"], OI["green"], OI["grey"]]
    x = np.arange(len(methods))
    bar_w = 0.58

    # --- Left: Peak GPU Memory ---
    bars1 = ax1.bar(x, mem_gb, width=bar_w, color=colors, edgecolor="#374151", lw=0.7)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.15, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=7.0, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right", fontsize=7.2)
    ax1.set_ylim(0, 6.7)
    ax1.set_ylabel("Peak GPU Memory (GB)")
    ax1.set_title("(a) Memory footprint", fontsize=8.2, loc="center", pad=3)

    # Reduction Callout Badge
    ax1.text(1.0, 3.4, "-67.3%", ha="center", va="center", fontsize=7.2, weight="bold",
             color=OI["vermillion"], bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEF2F2", edgecolor=OI["vermillion"], lw=0.6))
    ax1.annotate("", xy=(1.0, 2.25), xytext=(1.0, 3.1),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=OI["vermillion"]))

    # --- Right: Training Time ---
    bars2 = ax2.bar(x, time_s, width=bar_w, color=colors, edgecolor="#374151", lw=0.7)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 35, f"{int(round(h))}s",
                 ha="center", va="bottom", fontsize=6.8, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right", fontsize=7.2)
    ax2.set_ylim(0, 2100)
    ax2.set_ylabel("Training Time (Seconds)")
    ax2.set_title("(b) Wall-clock training", fontsize=8.2, loc="center", pad=3)

    # Speedup Callout Badge
    ax2.text(1.0, 1450, "-50.5%", ha="center", va="center", fontsize=7.2, weight="bold",
             color=OI["vermillion"], bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEF2F2", edgecolor=OI["vermillion"], lw=0.6))
    ax2.annotate("", xy=(1.0, 950), xytext=(1.0, 1330),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=OI["vermillion"]))

    fig.tight_layout(w_pad=1.4)
    save_figure(fig, "fig_computational_cost", ["fig4_computational_cost"])


# ===========================================================================
# Figure 5: Confusion Matrix Heatmap (fig_confusion_matrix)
# ===========================================================================
def generate_fig5_confusion():
    """Figure 5: 2x2 Confusion matrix heatmap for Selected XLM-R LoRA (seed 2024).
    
    Test comments n = 2,997.
    Row percentages:
      Non-Abusive: TN = 1,325 (92.3%), FP = 111 (7.7%)
      Abusive:     FN = 176 (11.3%),   TP = 1,385 (88.7%)
    """
    fig, ax = plt.subplots(figsize=(2.7, 2.5))

    # Confusion matrix counts: [[TN, FP], [FN, TP]]
    cm = np.array([[1325, 111],
                   [176, 1385]])
    row_sums = cm.sum(axis=1, keepdims=True)
    pcts = cm / row_sums * 100.0

    # Custom elegant colormap: soft white to deep Okabe Blue
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "AcademicBlue", ["#FFFFFF", "#EFF6FF", "#93C5FD", "#1D4ED8"], N=256
    )

    im = ax.imshow(pcts, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Labels inside cells (Clean, uncluttered, no horizontal spillovers)
    types = [["TN", "FP"], ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = pcts[i, j]
            t = types[i][j]
            text_color = "#FFFFFF" if pct > 60 else "#111827"
            
            cell_txt = f"{count:,}\n({pct:.1f}%)\n{t}"
            ax.text(j, i, cell_txt, ha="center", va="center",
                    color=text_color, fontsize=8.2, weight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Abusive", "Abusive"], fontsize=8.0, weight="bold")
    ax.set_yticklabels(["Non-Abusive\n(n=1,436)", "Abusive\n(n=1,561)"], fontsize=8.0, weight="bold")
    ax.set_xlabel("Predicted Class", fontsize=8.8, labelpad=3)
    ax.set_ylabel("Actual Ground Truth", fontsize=8.8, labelpad=3)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#D1D5DB", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(False, which="major")

    fig.tight_layout()
    save_figure(fig, "fig_confusion_matrix", ["fig5_confusion_matrix"])


# ===========================================================================
# Figure 6: Validation Convergence Trajectories (fig_validation_curves)
# ===========================================================================
def generate_fig6_convergence():
    """Figure 6: Validation Macro-F1 convergence trajectories over epochs.
    
    Shows convergence for XLM-R LoRA (seed 2024), XLM-R Full, DoRA, AdaLoRA,
    and MuRIL LoRA. Highlights peak at epoch 6 for LoRA (0.9073) and epoch 4 for Full (0.9028).
    """
    fig, ax = plt.subplots(figsize=(COL1_W, 2.5))

    # Exact trajectory data from log_history.json across models (seed 2024)
    epochs_lora = np.arange(1, 10)
    f1_lora     = [0.8541, 0.8856, 0.8895, 0.8937, 0.8970, 0.9073, 0.9001, 0.9032, 0.9032]

    epochs_full = np.arange(1, 6)
    f1_full     = [0.8884, 0.8968, 0.9007, 0.9028, 0.9007]

    epochs_dora = np.arange(1, 10)
    f1_dora     = [0.8576, 0.8828, 0.8809, 0.8949, 0.8919, 0.9017, 0.8996, 0.8991, 0.8999]

    epochs_muril= np.arange(1, 10)
    f1_muril    = [0.8611, 0.8850, 0.8901, 0.8948, 0.8925, 0.9004, 0.9010, 0.9010, 0.8979]

    epochs_ada  = np.arange(1, 10)
    f1_ada      = [0.8093, 0.8391, 0.8589, 0.8682, 0.8750, 0.8782, 0.8824, 0.8839, 0.8829]

    # Plot Lines
    ax.plot(epochs_lora, f1_lora, "o-", color=OI["vermillion"], lw=1.8, ms=4.5,
            label=r"XLM-R LoRA ($r{=}8$, selected)", zorder=5)
    ax.plot(epochs_full, f1_full, "s-", color=OI["blue"], lw=1.8, ms=4.5,
            label="XLM-R Full Fine-Tuning", zorder=4)
    ax.plot(epochs_dora, f1_dora, "^--", color=OI["green"], lw=1.2, ms=4.0, alpha=0.85,
            label=r"XLM-R DoRA ($r{=}8$)", zorder=3)
    ax.plot(epochs_muril, f1_muril, "v:", color=OI["sky"], lw=1.2, ms=4.0, alpha=0.85,
            label=r"MuRIL LoRA ($r{=}8$)", zorder=3)
    ax.plot(epochs_ada, f1_ada, "d-.", color=OI["grey"], lw=1.0, ms=3.5, alpha=0.75,
            label="XLM-R AdaLoRA", zorder=2)

    # Highlight Best Checkpoints with annotations
    ax.scatter([6], [0.9073], marker="*", s=140, color=OI["vermillion"], edgecolor="black", lw=0.7, zorder=6)
    ax.annotate("Best LoRA: 0.9073 (Ep 6)", xy=(6, 0.9073), xytext=(6.1, 0.9125),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=OI["vermillion"]),
                fontsize=7.0, color=OI["vermillion"], weight="bold", ha="left")

    ax.scatter([4], [0.9028], marker="*", s=120, color=OI["blue"], edgecolor="black", lw=0.7, zorder=6)
    ax.annotate("Full FT: 0.9028 (Ep 4)", xy=(4, 0.9028), xytext=(2.2, 0.9065),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=OI["blue"]),
                fontsize=6.8, color=OI["blue"], ha="left")

    ax.set_xticks(np.arange(1, 10))
    ax.set_xlim(0.6, 9.4)
    ax.set_ylim(0.80, 0.920)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=6.8, handletextpad=0.3, labelspacing=0.25)

    save_figure(fig, "fig_validation_curves", ["fig6_training_convergence"])


# ===========================================================================
# Figure 7: Forest Plot of Statistical Significance (fig_forest_significance)
# ===========================================================================
def generate_fig7_forest():
    """Figure 7: Forest plot of paired bootstrap differences with 95% CIs.
    
    11 Comparators against selected XLM-R LoRA (seed 2024) on n=2,997 test set.
    Includes McNemar p-values and confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(COL1_W, 3.4))

    # Exact data from results/study/significance.json
    comparators = [
        ("XLM-R Full (s2024)",   -0.0050, -0.0127,  0.0027, 0.235),
        ("XLM-R Full (s42)",     -0.0009, -0.0091,  0.0074, 0.872),
        ("XLM-R Full (s123)",     0.0015, -0.0067,  0.0095, 0.811),
        ("XLM-R DoRA (s2024)",   -0.0004, -0.0070,  0.0063, 1.000),
        ("MuRIL LoRA (s2024)",    0.0024, -0.0060,  0.0107, 0.634),
        ("MuRIL Full (s2024)",   -0.0007, -0.0097,  0.0080, 0.941),
        ("Published Recipe (r4)", 0.0037, -0.0037,  0.0107, 0.355),
        ("IndicBERTv2 LoRA",      0.0067, -0.0024,  0.0157, 0.168),
        ("TF-IDF + LinearSVC",    0.0013, -0.0084,  0.0110, 0.840),
        ("mBERT LoRA (s2024)",    0.0097,  0.0007,  0.0193, 0.054),
        ("XLM-R AdaLoRA",         0.0160,  0.0077,  0.0244, 0.0002),
    ]

    y_pos = np.arange(len(comparators))[::-1]  # Top to bottom

    # Zero difference reference line
    ax.axvline(0, color="#111827", linestyle="--", lw=1.0, zorder=2)

    # Shaded decision regions
    ax.axvspan(0, 0.035, alpha=0.04, color=OI["vermillion"])
    ax.axvspan(-0.020, 0, alpha=0.04, color=OI["blue"])

    for i, (name, delta, lo, hi, p) in enumerate(comparators):
        y = y_pos[i]
        # Statistically significant if CI does not contain zero
        is_sig = (lo > 0)
        col = OI["vermillion"] if is_sig else OI["blue"]
        
        # Error bar: [delta - lo, hi - delta]
        ax.errorbar(delta, y, xerr=[[delta - lo], [hi - delta]],
                    fmt="o", color=col, ecolor=col, capsize=3.0,
                    elinewidth=1.4, markersize=5.0, zorder=4)

        # Value annotation on the right margin
        sig_star = r"^{***}" if p < 0.001 else (r"^{*}" if p < 0.05 else "")
        p_txt = f"$p{sig_star}={p:.3f}$" if p >= 0.001 else r"$p<0.001^{***}$"
        annot = f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]  {p_txt}"
        ax.text(0.028, y, annot, va="center", fontsize=6.2, color="#1F2937")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[0] for c in comparators], fontsize=7.4)
    ax.set_xlim(-0.018, 0.058)
    ax.set_xlabel(r"Paired $\Delta$ Macro-F1 (Selected LoRA minus Comparator)")
    
    # Text headers for the interpretation (Separated to avoid overlap)
    ax.text(-0.009, len(comparators) - 0.2, r"$\leftarrow$ Favours Comp.",
            fontsize=6.8, color="#1E40AF", ha="center")
    ax.text(0.018, len(comparators) - 0.2, r"Favours Selected $\rightarrow$",
            fontsize=6.8, color="#B45309", ha="center")

    save_figure(fig, "fig_forest_significance", ["fig7_forest_significance"])


# ===========================================================================
# Figure 8: Adversarial Robustness under Disguise Attacks (fig_adversarial_robustness)
# ===========================================================================
def generate_fig8_robustness():
    """Figure 8: Mean accuracy under 20 disguise attacks on 600 test comments.
    
    Compares Unguarded vs Protected with Dravidian Canonicalizer.
    Shows dramatic accuracy restoration (+13.1% to +15.1%).
    """
    fig, ax = plt.subplots(figsize=(COL1_W, 2.75))

    systems = ["TF-IDF + LinearSVC", "XLM-R Full Fine-Tune", "XLM-R LoRA (Selected)"]
    raw_acc   = [0.712, 0.710, 0.725]  # Without Input Guard
    guard_acc = [0.863, 0.860, 0.856]  # With Dravidian Input Guard

    y = np.arange(len(systems))
    bar_h = 0.35

    # Horizontal Bar Chart
    bars_raw = ax.barh(y + bar_h/2, raw_acc, height=bar_h, color=OI["grey"],
                       edgecolor="#374151", lw=0.7, label="Unguarded (Raw Text)")
    bars_guard = ax.barh(y - bar_h/2, guard_acc, height=bar_h, color=OI["green"],
                         edgecolor="#374151", lw=0.7, label="Guarded (Dravidian Normalizer)")

    # Data value labels
    for b in bars_raw:
        w = b.get_width()
        ax.text(w - 0.05, b.get_y() + b.get_height()/2, f"{w*100:.1f}%",
                va="center", ha="right", fontsize=7.0, color="#FFFFFF", weight="bold")

    for i, b in enumerate(bars_guard):
        w = b.get_width()
        diff = (w - raw_acc[i]) * 100
        ax.text(w + 0.01, b.get_y() + b.get_height()/2,
                f"{w*100:.1f}% (+{diff:.1f}%)",
                va="center", ha="left", fontsize=7.0, color=OI["green"], weight="bold")

    # Baseline Clean Text Reference Line (90.4% clean accuracy)
    ax.axvline(0.904, color="#9CA3AF", linestyle=":", lw=1.2, zorder=2)
    ax.text(0.906, 2.45, "Clean baseline (90.4%)", fontsize=6.8, color="#6B7280", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(systems, fontsize=7.8, weight="bold")
    ax.set_xlim(0, 1.06)
    ax.set_ylim(-0.5, 2.85)
    ax.set_xlabel("Mean Accuracy Under 20 Disguise Attacks (n=600)")
    
    # Legend placed cleanly above bars
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=2,
              frameon=False, fontsize=7.0, handletextpad=0.3, columnspacing=0.8)

    save_figure(fig, "fig_adversarial_robustness", ["fig8_robustness_defense"])


# ===========================================================================
# Figure 9: Calibration & Decision Threshold Trade-off (fig_calibration_thresholds)
# ===========================================================================
def generate_fig9_calibration():
    """Figure 9: Dual panel calibration reliability & threshold sweep.
    
    Left panel: Reliability diagram before (T=1.0) and after (T=1.334) temperature scaling.
    Right panel: Precision, Recall, and Review load trade-off across block thresholds.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL1_W, 2.45))

    # --- Panel 1: Reliability Diagram ---
    # Exact data from results/calibration.json
    conf_before = [0.549, 0.649, 0.756, 0.857, 0.985]
    acc_before  = [0.660, 0.643, 0.680, 0.767, 0.961]

    conf_after  = [0.552, 0.658, 0.754, 0.855, 0.978]
    acc_after   = [0.657, 0.642, 0.719, 0.824, 0.978]

    # Perfect calibration line
    ax1.plot([0.5, 1.0], [0.5, 1.0], "--", color="#9CA3AF", lw=1.1, label="Perfect")
    ax1.plot(conf_before, acc_before, "s-", color=OI["vermillion"], lw=1.3, ms=3.8,
             label="Raw ($T{=}1$)")
    ax1.plot(conf_after, acc_after, "o-", color=OI["green"], lw=1.5, ms=4.0,
             label="Scaled ($T{=}1.334$)")

    ax1.set_xlim(0.48, 1.02)
    ax1.set_ylim(0.48, 1.02)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Mean Confidence", fontsize=8.0)
    ax1.set_ylabel("Empirical Accuracy", fontsize=8.0)
    ax1.set_title("(a) Reliability diagram", fontsize=8.0, pad=3)
    ax1.legend(loc="upper left", frameon=False, fontsize=6.2, handletextpad=0.2, borderpad=0.1)

    # --- Panel 2: Decision Threshold Sweep ---
    # Exact trade-off data from results/threshold_sweep.csv
    high_th = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    prec    = [92.8, 93.5, 94.0, 94.5, 95.0, 95.5, 96.2, 97.0, 98.4, 99.0]
    rec     = [89.7, 88.7, 87.4, 86.5, 85.5, 83.8, 82.2, 79.1, 76.0, 70.2]
    review  = [1.1,  2.0,  3.1,  4.0,  4.7,  6.1,  7.5,  9.2,  11.6, 15.2]

    ax2.plot(high_th, prec, "^-", color=OI["blue"], lw=1.3, ms=3.5, label="Block Precision")
    ax2.plot(high_th, rec, "v-", color=OI["vermillion"], lw=1.3, ms=3.5, label="Block Recall")
    ax2.plot(high_th, review, "o:", color=OI["amber"], lw=1.1, ms=3.2, label="Review Share (%)")

    # Mark the operational operating point (0.50)
    ax2.axvline(0.50, color="#6B7280", linestyle=":", lw=1.0)
    ax2.annotate(r"$\tau=0.50$", xy=(0.50, 92.8), xytext=(0.55, 80),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#4B5563"),
                 fontsize=6.2, color="#374151")

    ax2.set_xlim(0.48, 0.97)
    ax2.set_ylim(0, 102)
    ax2.set_xlabel(r"Block Threshold $\tau_{\mathrm{high}}$ ($\tau_{\mathrm{low}}=0.45$)", fontsize=7.8)
    ax2.set_ylabel("Rate (%)", fontsize=8.0, labelpad=1)
    ax2.set_title("(b) Moderation operating trade-off", fontsize=8.0, pad=3)
    ax2.legend(loc="center right", frameon=False, fontsize=6.0, handletextpad=0.2, borderpad=0.1)

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "fig_calibration_thresholds", ["fig9_calibration_threshold_tradeoff"])


# ===========================================================================
# CLI Driver
# ===========================================================================
FIGURE_MAP = {
    1: generate_fig1_lora,
    2: generate_fig2_pipeline,
    3: generate_fig3_frontier,
    4: generate_fig4_cost,
    5: generate_fig5_confusion,
    6: generate_fig6_convergence,
    7: generate_fig7_forest,
    8: generate_fig8_robustness,
    9: generate_fig9_calibration,
}


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade figures for Telugu Hate Speech Detector paper.")
    parser.add_argument("--fig", type=int, choices=range(1, 10), help="Generate a specific figure number (1-9)")
    parser.add_argument("--all", action="store_true", default=True, help="Generate all 9 figures (default)")
    args = parser.parse_args()

    print("=" * 70)
    print("TELUGU HATE SPEECH DETECTOR - PUBLICATION FIGURE GENERATOR")
    print(f"Output Primary: {OUT_PAPER}")
    print(f"Output Mirror:  {OUT_MIRROR}")
    print("=" * 70)

    if args.fig:
        print(f"Generating Figure {args.fig} of 9...")
        FIGURE_MAP[args.fig]()
    else:
        for num in sorted(FIGURE_MAP.keys()):
            print(f"Generating Figure {num} of 9...")
            FIGURE_MAP[num]()

    print("=" * 70)
    print("[SUCCESS] All requested figures generated cleanly in dual format (PDF + PNG).")
    print("=" * 70)


if __name__ == "__main__":
    main()
