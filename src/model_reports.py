"""Generate per-model markdown reports, figures and INDEX from results/study artifacts.

Usage (CPU only, idempotent, overwrites docs/models/*):
    .venv\\Scripts\\python -I src/model_reports.py

Every number is read from results/study/* (study_results.csv, <run>/metrics.json,
<run>/log_history.json, <run>/model/adapter_config.json, <run>/model/training_args.bin)
or from the HF config cache (AutoConfig, local_files_only). Hard-coded values are only
the paper / professor reference settings (REFERENCE) and method descriptions (GOAL).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: never touch CUDA

import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from transformers import AutoConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "results" / "study"
OUT = ROOT / "docs" / "models"
FIG = OUT / "figures"
TRAIN_PY = ROOT / "src" / "train.py"

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
CM_CMAP = LinearSegmentedColormap.from_list("okabe_blue", ["#FFFFFF", "#56B4E9", "#0072B2"])

XLMR, MURIL, MBERT, INDIC = (
    "FacebookAI/xlm-roberta-base",
    "google/muril-base-cased",
    "bert-base-multilingual-cased",
    "ai4bharat/IndicBERTv2-MLM-only",
)
BACKBONE = {  # CSV model id -> (slug prefix, display name)
    XLMR: ("xlmr", "xlm-roberta-base"),
    MURIL: ("muril", "MuRIL base cased"),
    MBERT: ("mbert", "bert-base-multilingual-cased (mBERT)"),
    INDIC: ("indicbertv2", "IndicBERTv2-MLM-only"),
}
EXP_ID = {"xlmr_full": "E1", "xlmr_lora": "E2", "xlmr_adalora": "E4", "xlmr_dora": "E5",
          "mbert_lora": "E6", "indicbertv2_lora": "E7"}
ORDER = ["xlmr_full", "xlmr_lora", "xlmr_adalora", "xlmr_dora", "mbert_lora", "indicbertv2_lora",
         "muril_full", "muril_lora", "muril_adalora", "muril_dora", "mbert_dora",
         "paper_xlmr_lora_r4", "prof_xlmr_lora_r16"]
REF_LORA = (2e-4, 10, 8, 64)  # MuRIL reference cell (lr, epochs, r, alpha); its 3 seeds form muril_lora

# Reference settings from the project spec (NOT run artifacts; used only for the alignment table).
REFERENCE = {
    "paper": {"lora_r": 4, "lora_alpha": 32, "lora_dropout": 0.1, "train_bs": 8, "max_length": 128,
              "epochs": 5, "lr": None, "optimizer": "AdamW", "fp16": True,
              "target_modules": "query,value", "weight_decay": None},
    "prof": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.1, "train_bs": 16, "max_length": 128,
             "epochs": 5, "lr": 2e-4, "optimizer": None, "fp16": None,
             "target_modules": None, "weight_decay": 0.01},
}
PENDING = {  # slug -> (CSV-independent config from REFERENCE, backbone)
    "paper_xlmr_lora_r4": ("paper", XLMR),
    "prof_xlmr_lora_r16": ("prof", XLMR),
}
GOAL = {
    "full": ("Full fine-tuning is the baseline: every backbone weight and the classification head are "
             "updated by AdamW. The other methods are compared against it on the same backbone."),
    "lora": ("LoRA freezes W and learns a low-rank update W + BA on the query and value projections "
             "(B in R^{d_out x r}, A in R^{r x d_in}); the effective update is scaled by alpha/r."),
    "dora": ("DoRA decomposes each adapted weight into magnitude and direction, W' = m * (W + BA) / "
             "||W + BA||, training a magnitude vector m plus the LoRA direction BA (use_dora=True in PEFT)."),
    "adalora": ("AdaLoRA parametrises each update as P diag(E) Q (SVD-like) and reallocates rank across "
                "matrices during training, from init_r towards target_r (adaptive rank budget)."),
}
METHOD_LABEL = {"full": "Full FT", "lora": "LoRA", "dora": "DoRA", "adalora": "AdaLoRA"}
NUM_LABELS = 2  # train.py load_model default


# ---------------------------------------------------------------- loading
def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_training_args(p):
    """training_args.bin is the pickled TrainingArguments written by the HF Trainer."""
    if not p.exists():
        return None
    try:
        import torch

        a = torch.load(p, weights_only=False, map_location="cpu")
    except Exception as exc:  # report and continue; fields become 'not recorded'
        print(f"  training_args.bin unreadable ({p}): {exc}")
        return None

    def val(x):
        return getattr(x, "value", x)

    return {"fp16": a.fp16, "optim": val(a.optim), "weight_decay": a.weight_decay,
            "warmup_steps": a.warmup_steps, "lr_scheduler": val(a.lr_scheduler_type),
            "eval_bs": a.per_device_eval_batch_size, "train_bs": a.per_device_train_batch_size,
            "epochs": a.num_train_epochs, "learning_rate": a.learning_rate,
            "max_grad_norm": a.max_grad_norm}


def load_runs():
    runs, notes = [], []
    with open(STUDY / "study_results.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = STUDY / row["run_name"]
            if not (d / "metrics.json").exists():
                notes.append(f"skip {row['run_name']}: no metrics.json")
                continue
            m = jload(d / "metrics.json")
            hist = jload(d / "log_history.json") if (d / "log_history.json").exists() else []
            ad = d / "model" / "adapter_config.json"
            full = row["method"] == "full"
            runs.append({
                "run": row["run_name"], "dir": d, "csv": row, "m": m, "hist": hist,
                "adapter": jload(ad) if ad.exists() else None,
                "targs": load_training_args(d / "model" / "training_args.bin"),
                "method": row["method"], "model": row["model"], "seed": int(row["seed"]),
                "lr": float(row["lr"]), "epochs": int(row["epochs"]),
                "r": None if full else int(row["lora_r"]),
                "alpha": None if full else int(row["lora_alpha"]),
            })
            # CSV must agree with the run's own metrics.json (test/val macro-F1, params)
            for k_csv, k_m in [("test_macro_f1", "test_macro_f1"), ("val_macro_f1", "val_macro_f1"),
                               ("trainable_params", "trainable_params")]:
                if abs(float(row[k_csv]) - float(m[k_m])) > 1e-9:
                    notes.append(f"CSV/metrics.json MISMATCH {row['run_name']} {k_csv}")
    for n in notes:
        print(n)
    return runs


def build_groups(runs):
    """Group runs by (backbone+method slug, lr, epochs, r, alpha). MuRIL LoRA grid cells go to E3."""
    e3_rows, groups = [], {}
    for r in runs:
        name = r["run"]
        if name.startswith("paper_") or name.startswith("prof_"):
            base = "paper_xlmr_lora_r4" if name.startswith("paper_") else "prof_xlmr_lora_r16"
        else:
            prefix = BACKBONE[r["model"]][0]
            if prefix == "muril" and r["method"] == "lora":
                if r["seed"] == 42:
                    e3_rows.append(r)
                if (r["lr"], r["epochs"], r["r"], r["alpha"]) != REF_LORA:
                    continue  # grid cell, reported in E3 only
            base = f"{prefix}_{r['method']}"
        key = (base, r["lr"], r["epochs"], r["r"], r["alpha"])
        groups.setdefault(key, {"base": base, "runs": []})["runs"].append(r)

    out, seen = [], {}
    for key, g in groups.items():
        base = g["base"]
        n = seen.get(base, 0)
        seen[base] = n + 1
        slug = base if n == 0 else f"{base}_lr{key[1]:g}_e{key[2]}_r{key[3]}_a{key[4]}"
        g["runs"].sort(key=lambda x: x["seed"])
        out.append({"slug": slug, "base": base, "runs": g["runs"]})
    for slug, (refname, model) in PENDING.items():
        if slug not in {g["slug"] for g in out}:
            out.append({"slug": slug, "base": slug, "runs": [], "pending_ref": refname, "pending_model": model})
    out.sort(key=lambda g: ORDER.index(g["slug"]) if g["slug"] in ORDER else 99)
    return out, e3_rows


def group_cfg(g):
    """Config of the group: from the first run's metrics.json hyperparams, or REFERENCE for pending."""
    if not g["runs"]:
        ref = REFERENCE[g["pending_ref"]]
        return {"model": g["pending_model"], "method": "lora", "lr": ref["lr"], "epochs": ref["epochs"],
                "lora_r": ref["lora_r"], "lora_alpha": ref["lora_alpha"],
                "lora_dropout": ref["lora_dropout"], "train_bs": ref["train_bs"],
                "max_length": ref["max_length"], "target_modules": ref["target_modules"] or "query,value",
                "lr_scheduler": None, "early_stop_patience": None}
    r0 = g["runs"][0]
    hp = dict(r0["m"]["hyperparams"])
    hp.update({"model": r0["model"], "method": r0["method"], "epochs_cfg": r0["epochs"], "lr_cfg": r0["lr"]})
    return hp


def get_arch(model_id):
    try:
        c = AutoConfig.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:
        return None, f"not recorded: HF cache has no local config for {model_id} ({type(exc).__name__})"
    return {"model_type": c.model_type, "layers": c.num_hidden_layers, "hidden": c.hidden_size,
            "heads": c.num_attention_heads, "ffn": c.intermediate_size, "vocab": c.vocab_size,
            "max_pos": c.max_position_embeddings}, None


# ---------------------------------------------------------------- maths
def safe_div(a, b):
    return a / b if b else 0.0


def cm_metrics(cm):
    (tn, fp), (fn, tp) = cm
    n = tn + fp + fn + tp
    ap, ar = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    af = safe_div(2 * ap * ar, ap + ar)
    np_, nr = safe_div(tn, tn + fn), safe_div(tn, tn + fp)
    nf = safe_div(2 * np_ * nr, np_ + nr)
    return {"n": n, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": safe_div(tn + tp, n), "ab_p": ap, "ab_r": ar, "ab_f1": af,
            "nab_f1": nf, "macro_f1": (af + nf) / 2,
            "weighted_f1": safe_div(nf * (tn + fp) + af * (fn + tp), n)}


def mstd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    m = statistics.fmean(vals)
    return m, (statistics.stdev(vals) if len(vals) > 1 else None), len(vals)


def fmt_ms(vals, nd=4):
    m, s, n = mstd(vals)
    if m is None:
        return "not recorded"
    if s is None:
        return f"{m:.{nd}f} (n=1, std not recorded)"
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def fnum(x, nd=4):
    return "not recorded" if x is None else f"{x:.{nd}f}"


def head_params(a):
    h = a["hidden"]
    if a["model_type"] in ("xlm-roberta", "roberta"):  # RobertaClassificationHead: dense + out_proj
        return (h * h + h) + (h * NUM_LABELS + NUM_LABELS)
    return h * NUM_LABELS + NUM_LABELS  # BERT: classifier Linear on pooler output


def predicted(method, a, r, n_targets, init_r=None):
    """Formula breakdown (see report). q/v in BERT/RoBERTa encoders have d_in = d_out = hidden."""
    h, L = a["hidden"], a["layers"]
    if method == "full":
        return None
    if method == "lora":
        per = r * (2 * h)
    elif method == "dora":
        per = r * (2 * h) + h  # + magnitude vector of length d_out
    elif method == "adalora":
        init_r = init_r if init_r is not None else r * 3 // 2  # train.py: init_r = lora_r*3//2
        per = init_r * (2 * h + 1)  # P (d_out x k), E (k), Q (k x d_in)
    else:
        raise ValueError(method)
    adapter = L * n_targets * per
    return adapter, head_params(a)


def targets_of(cfg):
    return len((cfg.get("target_modules") or "query,value").split(","))


# ---------------------------------------------------------------- figures
def plot_curves(slug, runs, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, r in enumerate(runs):
        c = OKABE[[0, 4, 2, 5, 6][i % 5]]
        ev = [(e["epoch"], e["eval_macro_f1"]) for e in r["hist"] if "eval_macro_f1" in e]
        tl = [(e["epoch"], e["loss"]) for e in r["hist"] if "loss" in e and "eval_loss" not in e]
        if ev:
            ax1.plot(*zip(*ev), marker="o", color=c, label=f"seed {r['seed']}")
        if tl:
            ax2.plot(*zip(*tl), color=c, label=f"seed {r['seed']}")
    ax1.set_title("Validation macro-F1 per epoch (Trainer eval)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("macro-F1 (validation)")
    ax2.set_title("Training loss (logged every 50 steps)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("train loss")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle(slug)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cm(slug, cm, seed, path):
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(arr, cmap=CM_CMAP)
    thr = arr.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{arr[i, j]}", ha="center", va="center",
                    color="white" if arr[i, j] > thr else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Non-abusive (0)", "Abusive (1)"])
    ax.set_yticks([0, 1], ["Non-abusive (0)", "Abusive (1)"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{slug}, seed {seed} (best val F1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


SHORT_ADAPTER = {"lora": "LoRA A,B", "dora": "LoRA A,B + magnitude m", "adalora": "AdaLoRA P,E,Q"}


def ascii_flow(method, a, cfg):
    h, L, tgt = a["hidden"], a["layers"], cfg.get("target_modules") or "query,value"
    T = "TRAINABLE"
    fr = "frozen"
    if method == "full":
        q = v = k = o = ffn = f"{T}"
        head_desc = f"{T} (all weights are trainable)"
    else:
        q = v = f"{fr} W + {T} {SHORT_ADAPTER[method]}"
        k = o = ffn = fr
        head_desc = f"{T} (modules_to_save)"
    if a["model_type"] in ("xlm-roberta", "roberta"):
        pool = "<s> token (no pooler)"
        head = f"head: dense({h}->{h}) + out_proj({h}->2)  [{head_desc}]"
    else:
        pool = "[CLS] -> pooler dense+tanh (pooler: " + ("trainable" if method == "full" else "frozen") + ")"
        head = f"head: classifier Linear({h}->2)  [{head_desc}]"
    return "\n".join([
        "Telugu text (raw)",
        "  |  normalize_text('normalized'): NFC -> mask URLs & @mentions -> collapse spaces  [src/data_utils.py]",
        "  v",
        f"Tokenizer, truncation to max_length={cfg.get('max_length')}  [src/train.py]",
        "  |",
        f"  v",
        f"Embeddings: word + position ({a['vocab']:,} x {h})      [{'TRAINABLE' if method == 'full' else 'frozen'}]",
        "  |",
        f"  v",
        f"Encoder layer x{L}:",
        f"    self-attn query (q):  {q}  (targets: {tgt})",
        f"    self-attn key   (k):  {k}",
        f"    self-attn value (v):  {v}",
        f"    self-attn out   (o):  {o}",
        f"    FFN {h}->{a['ffn']}->{h}:    {ffn}",
        f"    LayerNorm:            {'TRAINABLE' if method == 'full' else 'frozen'}",
        "  |",
        f"  v",
        pool,
        f"  |",
        f"  v",
        head,
        "  |",
        "  v",
        "2 logits -> softmax -> argmax -> Non-abusive (0) / Abusive (1)",
    ])


def group_report(g, groups_by_slug, figs_ok):
    slug, runs, cfg = g["slug"], g["runs"], group_cfg(g)
    pending = not runs
    method = cfg["method"]
    prefix, bb_name = BACKBONE[cfg["model"]]
    exp = "extra"
    if not g["base"].startswith(("paper_", "prof_")):
        exp = EXP_ID.get(g["base"], "extra")
    seeds = [r["seed"] for r in runs]
    status = "pending - no completed runs" if pending else (
        f"complete - {len(runs)} seed(s) {seeds}" + (" (single seed: no std)" if len(runs) == 1 else ""))
    L = []
    title = {"paper_xlmr_lora_r4": "paper_xlmr_lora_r4 (paper Table 2 config)",
             "prof_xlmr_lora_r16": "prof_xlmr_lora_r16 (professor's recommended start config)"}.get(
        slug, f"{slug}")
    L += [f"# {title}", "",
          "| Field | Value |", "|---|---|",
          f"| Experiment | {exp} |",
          f"| Status | {status} |",
          f"| n seeds | {len(runs)} |",
          f"| Backbone | `{cfg['model']}` ({bb_name}) |",
          f"| Method | {METHOD_LABEL[method]} |",
          f"| Config | lr={cfg.get('lr_cfg', cfg.get('lr')) if cfg.get('lr_cfg', cfg.get('lr')) is not None else 'not specified'}, "
          f"epochs={cfg.get('epochs_cfg', cfg.get('epochs'))}"
          + ("" if method == "full" else f", r={cfg.get('lora_r')}, alpha={cfg.get('lora_alpha')}")
          + " |", ""]
    if pending:
        L += ["> **pending - no completed runs.** The matching training is still running; this report "
              "is regenerated from artifacts once `results/study/study_results.csv` has rows for it. "
              "Values below are reference settings or formula predictions, not measurements.", ""]

    # 2. Goal
    L += ["## 2. Goal", "", GOAL[method], ""]
    if method == "full":
        L += [f"Question answered: what test macro-F1 does full fine-tuning of {bb_name} reach at "
              f"lr={cfg.get('lr_cfg', cfg.get('lr'))}, {cfg.get('epochs_cfg', cfg.get('epochs'))} epochs?", ""]
    elif method == "lora":
        L += [f"Question answered: how much macro-F1 does a LoRA adapter with r={cfg.get('lora_r')}, "
              f"alpha={cfg.get('lora_alpha')} on query/value keep relative to full fine-tuning?", ""]
    else:
        L += [f"Question answered: does {METHOD_LABEL[method]} (r={cfg.get('lora_r')}, "
              f"alpha={cfg.get('lora_alpha')}) differ from plain LoRA and full fine-tuning?", ""]

    # 3. Architecture
    a, aerr = get_arch(cfg["model"])
    L += ["## 3. Architecture end-to-end", ""]
    init_r = None
    if runs and runs[0]["adapter"] and "init_r" in runs[0]["adapter"]:
        init_r = runs[0]["adapter"]["init_r"]
    if a is None:
        L += [aerr, ""]
    else:
        L += ["```text", ascii_flow(method, a, cfg), "```", ""]
        n_t = targets_of(cfg)
        if method == "full":
            L += ["Full fine-tuning: no adapters; all weights and the head are trainable.", ""]
        else:
            L += [f"Adapter formula (q,v with d_in = d_out = hidden = {a['hidden']}):",
                  "- LoRA: `layers x targets x r x (d_in + d_out)`",
                  "- DoRA: LoRA + `layers x targets x d_out` (magnitude vector)",
                  "- AdaLoRA: `layers x targets x init_r x (d_in + d_out + 1)`, init_r = r*3//2 "
                  "(train.py) or adapter_config.json init_r",
                  "- Head (modules_to_save): RoBERTa-type `h*h+h + h*2+2`; BERT-type `h*2+2`; num_labels=2", ""]
        rows = [("layers", a["layers"], "AutoConfig.num_hidden_layers"),
                ("hidden size", a["hidden"], "AutoConfig.hidden_size"),
                ("attention heads", a["heads"], "AutoConfig.num_attention_heads"),
                ("FFN intermediate", a["ffn"], "AutoConfig.intermediate_size"),
                ("vocab size", f"{a['vocab']:,}", "AutoConfig.vocab_size"),
                ("max positions", a["max_pos"], "AutoConfig.max_position_embeddings")]
        L += md_table(["Architecture", "Value", "Source"], rows) + [""]
        if runs:
            tot = {r["m"]["total_params"] for r in runs}
            trn = {r["m"]["trainable_params"] for r in runs}
            pct = runs[0]["m"]["trainable_pct"]
            L += [f"Parameter counts across seeds identical: {'OK' if len(tot) == 1 and len(trn) == 1 else 'MISMATCH'}.", ""]
            print(f"[{slug}] param counts identical across seeds: {'OK' if len(tot)==1 and len(trn)==1 else 'MISMATCH'}")
            rec_total, rec_tr = runs[0]["m"]["total_params"], runs[0]["m"]["trainable_params"]
            rows = [("total params (metrics.json total_params)", f"{rec_total:,}", "metrics.json"),
                    ("trainable params", f"{rec_tr:,}", "metrics.json"),
                    ("trainable %", f"{pct:.4f}", "metrics.json"),
                    ("frozen = total - trainable", f"{rec_total - rec_tr:,}", "derived")]
            if method != "full":
                ad_p, hd_p = predicted(method, a, cfg["lora_r"], n_t, init_r)
                exp_tr = ad_p + hd_p
                ok = "OK" if exp_tr == rec_tr else f"MISMATCH (recorded {rec_tr:,} vs formula {exp_tr:,})"
                print(f"[{slug}] trainable check: {ok}")
                rows += [("adapter params (formula)", f"{ad_p:,}", "formula above"),
                         ("head params (formula)", f"{hd_p:,}", "formula above"),
                         ("adapter + head vs trainable", ok, "formula check")]
            else:
                ok = "OK (all params trainable)" if rec_tr == rec_total else "MISMATCH"
                print(f"[{slug}] trainable check: {ok}")
                rows += [("adapter + head vs trainable", ok, "formula check")]
            L += md_table(["Parameters", "Value", "Source"], rows) + [""]
        else:
            if method != "full":
                ad_p, hd_p = predicted(method, a, cfg["lora_r"], n_t)
                L += md_table(["Parameters", "Value", "Source"], [
                    ("trainable params", "pending", "metrics.json not yet written"),
                    ("adapter params (predicted from spec config)", f"{ad_p:,}", "formula, not measured"),
                    ("head params (predicted)", f"{hd_p:,}", "formula, not measured"),
                    ("predicted trainable", f"{ad_p + hd_p:,}", "formula, not measured")]) + [""]

    # 4. Training setup
    L += ["## 4. Training setup", ""]
    t0 = runs[0]["targs"] if runs and runs[0]["targs"] else None
    tt = (lambda k: t0[k]) if t0 else (lambda k: "not recorded")
    setup = [
        ("learning rate", cfg.get("lr_cfg", cfg.get("lr")) if runs else cfg.get("lr"), "metrics.json hyperparams.lr"),
        ("epochs (configured)", cfg.get("epochs_cfg", cfg.get("epochs")), "CSV / metrics.json"),
        ("train batch size", cfg.get("train_bs"), "metrics.json hyperparams.train_bs"),
        ("eval batch size", tt("eval_bs"), "training_args.bin"),
        ("max_length", cfg.get("max_length"), "metrics.json hyperparams.max_length"),
        ("optimizer", tt("optim"), "training_args.bin (HF Trainer default; train.py does not set it)"),
        ("weight decay", tt("weight_decay"), "training_args.bin (train.py sets 0.01)"),
        ("LR schedule", tt("lr_scheduler") if t0 else cfg.get("lr_scheduler"), "training_args.bin / metrics.json"),
        ("warmup steps", tt("warmup_steps"), f"training_args.bin (train.py: 10% of total steps, see below)"),
        ("fp16", tt("fp16"), "training_args.bin (train.py: fp16 = CUDA available unless --fp16)"),
        ("early stopping patience", cfg.get("early_stop_patience") if not pending else "not recorded",
         "metrics.json hyperparams"),
        ("grad clip (max_grad_norm)", tt("max_grad_norm"), "training_args.bin"),
    ]
    if method == "full":
        setup += [("LoRA r / alpha / dropout / targets", "n/a (full FT)", "")]
    else:
        setup += [("LoRA r / alpha / dropout", f"{cfg.get('lora_r')} / {cfg.get('lora_alpha')} / "
                   f"{cfg.get('lora_dropout')}", "metrics.json hyperparams"),
                  ("target modules", cfg.get("target_modules"), "metrics.json hyperparams.target_modules")]
    setup += [("preprocessing", "not recorded in metrics.json", "train.py default --preprocess normalized")]
    L += md_table(["Setting", "Value", "Source"], [(a_, b_, c_) for a_, b_, c_ in setup]) + [""]
    if t0 is not None and runs:
        L += [f"Warmup and schedule code: `src/train.py` line ~208 sets warmup_steps = max(1, int(0.1 * total_steps)); "
              f"line 237 sets weight_decay=0.01; line 239 uses lr_scheduler_type=args.lr_scheduler (default cosine).", ""]
        print(f"[{slug}] schedule: warmup_steps recorded {t0['warmup_steps']}, weight_decay {t0['weight_decay']}")

    # 5. What happened
    L += ["## 5. What happened", ""]
    if pending:
        L += ["pending - no completed runs.", ""]
    else:
        rows, stopped = [], []
        for r in runs:
            ev = [e["epoch"] for e in r["hist"] if "eval_macro_f1" in e]
            ran = int(round(max(ev))) if ev else None
            cfg_ep = r["epochs"]
            early = "yes" if (ran is not None and ran < cfg_ep) else "no"
            stopped.append(early == "yes")
            rows.append((r["seed"], int(round(r["m"]["hyperparams"]["best_epoch"])),
                         f"{ran} / {cfg_ep}" if ran is not None else "not recorded", early,
                         fnum(r["m"]["val_macro_f1"]), fnum(r["m"]["test_macro_f1"]),
                         f"{r['m']['training_time_sec']:.1f}", f"{r['m']['peak_gpu_memory_mb']:.1f}"))
        L += md_table(["seed", "best epoch", "epochs run / configured", "early stop", "val macro-F1",
                       "test macro-F1", "train time (s)", "peak GPU (MB)"], rows) + [""]
        L += [f"Early stopping (epochs run < configured) triggered in {sum(stopped)} of {len(runs)} seeds. "
              "Epochs run is the last epoch with a Trainer eval entry in log_history.json.", ""]
        if figs_ok:
            L += [f"![eval macro-F1 and training loss per epoch](figures/{slug}_curves.png)", ""]
            L += ["Left: validation macro-F1 per epoch (Trainer eval entries, all seeds). "
                  "Right: training loss from log_history.json (logged every 50 steps), x-axis in epochs.", ""]

    # 6. Results
    L += ["## 6. Results", ""]
    cms = {r["seed"]: cm_metrics(r["m"]["confusion_matrix"]) for r in runs}
    best = max(runs, key=lambda r: r["m"]["val_macro_f1"]) if runs else None
    if pending:
        L += ["pending - no completed runs.", ""]
    else:
        def col(k):
            return [cms[r["seed"]][k] for r in runs]
        rows = [("accuracy", fmt_ms(col("accuracy"))),
                ("abusive precision", fmt_ms(col("ab_p"))),
                ("abusive recall", fmt_ms(col("ab_r"))),
                ("abusive F1", fmt_ms(col("ab_f1"))),
                ("non-abusive F1", fmt_ms(col("nab_f1"))),
                ("macro-F1 (from CM)", fmt_ms(col("macro_f1"))),
                ("weighted-F1 (from CM)", fmt_ms(col("weighted_f1")))]
        L += md_table(["Metric (test set, mean ± sample std over seeds)", "Value"], rows) + [""]
        cons = []
        for r in runs:
            d = abs(cms[r["seed"]]["macro_f1"] - r["m"]["test_macro_f1"])
            cons.append("OK" if d < 1e-6 else f"MISMATCH({d:.2e})")
        L += [f"Consistency: CM-derived macro-F1 vs metrics.json test_macro_f1 per seed: {', '.join(cons)}.", ""]
        print(f"[{slug}] CM macro-F1 vs metrics.json: {', '.join(cons)}")
        rows = []
        for r in runs:
            c = cms[r["seed"]]
            rows.append((r["seed"], c["n"], c["tn"], c["fp"], c["fn"], c["tp"],
                         f"{safe_div(c['fp'], c['fn']):.3f}"))
        L += md_table(["seed", "test n", "TN", "FP (pred Abusive, true not)", "FN (pred not, true Abusive)",
                       "TP", "FP/FN"], rows) + [""]
        fp_m, fn_m = mstd([cms[r["seed"]]["fp"] for r in runs])[0], mstd([cms[r["seed"]]["fn"] for r in runs])[0]
        L += [f"Error counts (mean over seeds): FP = {fp_m:.1f}, FN = {fn_m:.1f}, "
              f"FP/FN = {safe_div(fp_m, fn_m):.3f}. Class 1 = Abusive, confusion matrix layout [[TN,FP],[FN,TP]].", ""]
        if figs_ok and best is not None:
            L += [f"Confusion matrix, seed {best['seed']} (highest validation macro-F1 = "
                  f"{best['m']['val_macro_f1']:.4f}):", "", f"![confusion matrix](figures/{slug}_cm.png)", ""]

    # 7. Compared with baseline
    L += ["## 7. Compared with baseline", ""]
    if method == "full":
        L += ["This group is the same-backbone full fine-tune baseline.", ""]
    else:
        full_slug = f"{prefix}_full"
        fg = groups_by_slug.get(full_slug)
        if fg is None or not fg["runs"]:
            L += [f"No same-backbone full fine-tune is recorded in results/study for {bb_name}, so no baseline "
                  "difference is computed.", ""]
        elif pending or not runs:
            L += ["pending - no completed runs.", ""]
        else:
            rows = []
            for name, key in [("test macro-F1", "test_macro_f1"), ("val macro-F1", "val_macro_f1")]:
                a_m, a_s, _ = mstd([r["m"][key] for r in runs])
                b_m, b_s, _ = mstd([r["m"][key] for r in fg["runs"]])
                if a_m is None or b_m is None:
                    rows.append((name, "not recorded", "", "", "not recorded"))
                    continue
                diff = a_m - b_m
                if a_s is None or b_s is None:
                    verdict = "not computable (n=1 on one side)"
                else:
                    verdict = "within seed noise" if abs(diff) < max(a_s, b_s) else "outside seed noise"
                rows.append((name, fmt_ms([r["m"][key] for r in runs]),
                             fmt_ms([r["m"][key] for r in fg["runs"]]),
                             f"{diff:+.4f}", verdict))
            L += md_table(["metric", f"{METHOD_LABEL[method]} (this)", "full FT (same backbone)",
                           "diff (this - full)", "|diff| vs max(std)"], rows) + [""]
            fr_tr = mstd([r["m"]["trainable_params"] for r in fg["runs"]])[0]
            tr = mstd([r["m"]["trainable_params"] for r in runs])[0]
            mem_a = mstd([r["m"]["peak_gpu_memory_mb"] for r in runs])[0]
            mem_b = mstd([r["m"]["peak_gpu_memory_mb"] for r in fg["runs"]])[0]
            time_a = mstd([r["m"]["training_time_sec"] for r in runs])[0]
            time_b = mstd([r["m"]["training_time_sec"] for r in fg["runs"]])[0]
            L += md_table(["ratio", "value"], [
                ("trainable params: this / full", f"{tr / fr_tr:.4f} ({tr:,.0f} / {fr_tr:,.0f})"),
                ("peak GPU memory: this / full", f"{mem_a / mem_b:.3f} ({mem_a:.0f} MB / {mem_b:.0f} MB)"),
                ("training time: this / full (mean)", f"{time_a / time_b:.3f} ({time_a:.0f} s / {time_b:.0f} s)"),
            ]) + [""]
            L += ["Std-based noise rule: 'within seed noise' if |mean diff| < max(std_this, std_full). "
                  "Sample std (n-1).", ""]

    # 8. Interpretation
    L += ["## 8. Interpretation", ""]
    if pending:
        L += ["- pending - no completed runs.", ""]
    else:
        t_m, t_s, _ = mstd([r["m"]["test_macro_f1"] for r in runs])
        L += [f"- Test macro-F1 mean {t_m:.4f} over {len(runs)} seed(s)"
              + (f" (std {t_s:.4f})." if t_s is not None else " (std not recorded: n=1)."),
              f"- Seed with the highest validation macro-F1: {best['seed']} (val {best['m']['val_macro_f1']:.4f}, "
              f"test {best['m']['test_macro_f1']:.4f}).",
              f"- Best epochs by seed: {', '.join(str(int(round(r['m']['hyperparams']['best_epoch']))) for r in runs)}."]
        if method != "full":
            L += ["- Hypothesis (not tested): the adapter capacity limits how closely LoRA-type runs track "
                  "full fine-tuning on this data; the diffs in section 7 are the only evidence here."]
        if method == "adalora":
            L += ["- Hypothesis (not tested): the AdaLoRA rank pruning schedule (tinit=10%, tfinal=20% of steps) "
                  "may not have pruned to target_r in this run; final ranks are not recorded."]
        if method == "dora":
            L += ["- Hypothesis (not tested): the magnitude term adds little over LoRA at this rank; "
                  "compare with section 7 and the LoRA group."]
        L += [""]

    # 9. Spec alignment
    L += ["## 9. Spec alignment", ""]
    ours = {
        "LoRA rank r": cfg.get("lora_r") if method != "full" else "n/a (full FT)",
        "LoRA alpha": cfg.get("lora_alpha") if method != "full" else "n/a (full FT)",
        "LoRA dropout": cfg.get("lora_dropout") if method != "full" else "n/a (full FT)",
        "train batch size": cfg.get("train_bs"),
        "max_length": cfg.get("max_length"),
        "epochs (configured)": cfg.get("epochs_cfg", cfg.get("epochs")),
        "learning rate": cfg.get("lr_cfg", cfg.get("lr")),
        "optimizer": tt("optim"),
        "fp16": tt("fp16"),
        "target modules": cfg.get("target_modules") if method != "full" else "all weights",
        "weight decay": tt("weight_decay"),
    }
    if pending:
        ours = {k: "pending" for k in ours}
    rows = []
    for k, v in ours.items():
        pv, fv = REFERENCE["paper"], REFERENCE["prof"]
        key = {"LoRA rank r": "lora_r", "LoRA alpha": "lora_alpha", "LoRA dropout": "lora_dropout",
               "train batch size": "train_bs", "max_length": "max_length", "epochs (configured)": "epochs",
               "learning rate": "lr", "optimizer": "optimizer", "fp16": "fp16",
               "target modules": "target_modules", "weight decay": "weight_decay"}[k]
        pref, fref = pv[key], fv[key]

        def show(x):
            if x is None:
                return "not specified"
            return str(x)

        def match(spec):
            if pending:
                return "pending"
            if spec is None:
                return "n/a (not in spec)"
            if isinstance(v, str) and v.startswith("n/a"):
                return "n/a"
            if key == "optimizer":
                return "yes" if "adamw" in str(v).lower() else "no"
            if key == "target_modules":
                return "yes" if str(v).replace("+", ",").replace(" ", "") == str(spec).replace("+", ",") else "no"
            if isinstance(spec, bool):
                return "yes" if bool(v) == spec else "no"
            if v is None:
                return "not recorded"
            try:
                return "yes" if abs(float(v) - float(spec)) < 1e-12 else "no"
            except (TypeError, ValueError):
                return "yes" if str(v) == str(spec) else "no"
        rows.append((k, "pending" if pending else show(v), show(pref), show(fref),
                     f"paper: {match(pref)}; prof: {match(fref)}"))
    L += md_table(["parameter", "ours", "paper Table 2", "professor start", "match?"], rows)
    L += ["", "Paper and professor values come from the project spec, not from run artifacts. "
          "Paper lr is not specified in the spec.", ""]

    # 10. Limitations
    L += ["## 10. Limitations and artifact paths", ""]
    lim = [f"- Seeds: {len(runs)} ({', '.join(str(s) for s in seeds) if seeds else 'none yet'}). "
           "Std is over these seeds only; n=1 means no std.",
           "- Single dataset split from train.py / data_utils.py; test size is the sum of the CM (see section 6).",
           "- 'Peak GPU MB' and training time are from one GPU environment; hardware is not recorded in the artifacts.",
           "- Preprocessing level is not recorded in metrics.json; the train.py default is 'normalized'.",
           "- Parameter counts: trainable_params is the count at initialisation for AdaLoRA (final pruned ranks "
           "are not recorded).",
           "- Regenerate with `.venv\\Scripts\\python -I src/model_reports.py`."]
    if method == "full" and pending:
        lim = lim[:1] + lim[2:]
    L += lim + [""]
    if runs:
        L += ["Artifacts:", ""]
        for r in runs:
            L.append(f"- `results/study/{r['run']}/` (metrics.json, log_history.json, model/)")
        L.append("- `results/study/study_results.csv`")
        if figs_ok:
            L += [f"- `docs/models/figures/{slug}_curves.png`, `docs/models/figures/{slug}_cm.png`"]
        L.append("")
    else:
        L += ["Artifacts: none yet (pending).", ""]
    return "\n".join(L) + "\n", status, exp


def make_figs(g):
    runs = g["runs"]
    if not runs:
        return False
    plot_curves(g["slug"], runs, FIG / f"{g['slug']}_curves.png")
    best = max(runs, key=lambda r: r["m"]["val_macro_f1"])
    plot_cm(g["slug"], best["m"]["confusion_matrix"], best["seed"], FIG / f"{g['slug']}_cm.png")
    return True


# ---------------------------------------------------------------- E3
def e3(rows):
    """MuRIL LoRA rank x alpha grid (seed 42) and lr ablation at r8/alpha64."""
    if not rows:
        return "", None
    rows = sorted(rows, key=lambda r: (r["r"], r["alpha"], r["lr"]))
    grid = [r for r in rows if r["lr"] == REF_LORA[0] and r["epochs"] == REF_LORA[1]]
    lr_abl = [r for r in rows if r["r"] == 8 and r["alpha"] == 64]
    ranks, alphas = [4, 8, 16, 32], [16, 32, 64]
    cell = {(r["r"], r["alpha"]): r for r in grid}
    vmat = np.full((len(ranks), len(alphas)), np.nan)
    tmat = np.full_like(vmat, np.nan)
    for i, rk in enumerate(ranks):
        for j, al in enumerate(alphas):
            if (rk, al) in cell:
                vmat[i, j] = cell[(rk, al)]["m"]["val_macro_f1"]
                tmat[i, j] = cell[(rk, al)]["m"]["test_macro_f1"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, name in [(axes[0], vmat, "validation macro-F1"), (axes[1], tmat, "test macro-F1")]:
        im = ax.imshow(mat, cmap=CM_CMAP, vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        for i in range(len(ranks)):
            for j in range(len(alphas)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(alphas)), [f"alpha={a}" for a in alphas])
        ax.set_yticks(range(len(ranks)), [f"r={r}" for r in ranks])
        ax.set_title(f"MuRIL LoRA, lr=2e-4, seed 42: {name}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "e3_heatmap.png", dpi=150)
    plt.close(fig)

    return grid, lr_abl, vmat, tmat


def e3_report(rows, ref_runs):
    res = e3(rows)
    if not rows:
        return "# E3 - MuRIL LoRA rank x alpha grid\n\nNo grid rows found.\n"
    grid, lr_abl, vmat, tmat = res
    L = ["# E3 - MuRIL LoRA rank x alpha grid and lr ablation", "",
         "| Field | Value |", "|---|---|",
         "| Experiment | E3 |",
         f"| Status | complete - {len(rows)} runs (seed 42 only) |",
         "| n seeds | 1 per cell (seed 42); the reference cell (r=8, alpha=64, lr=2e-4) also has seeds 123 and 2024 in muril_lora |",
         "| Backbone | `google/muril-base-cased` (MuRIL base cased) |",
         f"| Method | LoRA on {grid[0]['m']['hyperparams']['target_modules']}, dropout "
         f"{grid[0]['m']['hyperparams']['lora_dropout']}, {grid[0]['epochs']} epochs |", ""]
    L += ["## Grid: rank x alpha (lr = 2e-4)", "", "![E3 heatmap](figures/e3_heatmap.png)", ""]
    rows_v = []
    for i, rk in enumerate([4, 8, 16, 32]):
        rows_v.append((f"r={rk}",) + tuple(
            ("not recorded" if np.isnan(vmat[i, j]) else f"{vmat[i, j]:.4f} / {tmat[i, j]:.4f}")
            for j in range(3)))
    L += md_table(["rank", "alpha=16 (val / test)", "alpha=32 (val / test)", "alpha=64 (val / test)"], rows_v) + [""]

    L += ["## Learning-rate ablation (r=8, alpha=64)", ""]
    lr_rows = []
    for r in sorted(lr_abl, key=lambda x: x["lr"]):
        m = r["m"]
        lr_rows.append((f"{r['lr']:g}", f"{m['val_macro_f1']:.4f}", f"{m['test_macro_f1']:.4f}",
                        f"{m['hyperparams']['best_epoch']:.0f}", r["run"]))
    L += md_table(["lr", "val macro-F1", "test macro-F1", "best epoch", "run"], lr_rows) + [""]

    L += ["## Trainable parameters per rank", ""]
    pr = {}
    for r in grid:
        pr.setdefault(r["r"], r["m"])
    prow = []
    for rk in sorted(pr):
        m = pr[rk]
        prow.append((rk, f"{m['trainable_params']:,}", f"{m['trainable_pct']:.4f}",
                     f"{m['total_params']:,}"))
    L += md_table(["rank r", "trainable params", "trainable %", "total params"], prow) + [""]
    ind ={rk: len({r["m"]["trainable_params"] for r in grid if r["r"] == rk}) == 1 for rk in pr}
    print(f"[e3] trainable params identical across alpha per rank: {all(ind.values())}")
    L += [f"Identical across alpha within each rank: {'OK' if all(ind.values()) else 'MISMATCH'}.", ""]

    # adapter formula check, MuRIL: 12 layers, hidden 768, q+v, head = classifier 768*2+2
    a, _ = get_arch(MURIL)
    if a is not None:
        chk = []
        for rk in sorted(pr):
            ad_p, hd_p = predicted("lora", a, rk, 2)
            ok = "OK" if ad_p + hd_p == pr[rk]["trainable_params"] else "MISMATCH"
            chk.append(f"r={rk}: formula {ad_p + hd_p:,} vs recorded {pr[rk]['trainable_params']:,} -> {ok}")
            print(f"[e3] r={rk} formula check: {ok}")
        L += ["Adapter formula check (`layers x 2 x r x 2*hidden` + classifier `2*hidden+2`):", ""] + \
             [f"- {c}" for c in chk] + [""]

    # best cell by validation
    best_grid = max(grid, key=lambda r: r["m"]["val_macro_f1"])
    best_all = max(rows, key=lambda r: r["m"]["val_macro_f1"])
    L += ["## Best cell by validation", "",
          f"- Best rank x alpha cell (lr=2e-4): r={best_grid['r']}, alpha={best_grid['alpha']} "
          f"(val {best_grid['m']['val_macro_f1']:.4f}, test {best_grid['m']['test_macro_f1']:.4f}).",
          f"- Best of all E3 rows including lr ablation: `{best_all['run']}` "
          f"(val {best_all['m']['val_macro_f1']:.4f}, test {best_all['m']['test_macro_f1']:.4f}).", ""]
    if ref_runs:
        vs = [r["m"]["val_macro_f1"] for r in ref_runs]
        ts = [r["m"]["test_macro_f1"] for r in ref_runs]
        L += [f"Seed spread of the 3-seed reference cell (muril_lora), mean ± sample std: val macro-F1 "
              f"{fmt_ms(vs)}; test macro-F1 {fmt_ms(ts)}.", "",
              "Caveat: each cell is one seed. Differences smaller than the seed spread above are not "
              "distinguishable from seed noise (heuristic, not a significance test).", ""]
    L += ["Hypothesis (not tested): alpha/r scaling (alpha = 4r..16r here) is what moves the grid more than rank alone.",
          ""]
    L += ["## Artifacts", "", "- `results/study/study_results.csv` (rows with model=google/muril-base-cased, method=lora, seed=42)",
          "- `results/study/lora_r*_a*_lr*_e10_s42/`, `results/study/lora_r8_a64_lr*_e10_s42/`", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- index
def index_md(group_rows, e3_status):
    L = ["# Model reports index", "",
         "Generated by `src/model_reports.py` from `results/study/`. Values are mean ± sample std over seeds "
         "(test and validation macro-F1).", "",
         "| Slug | Exp | Backbone | Method | Status | Seeds | Val macro-F1 | Test macro-F1 | Trainable params | Report | Figures |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for slug, runs, exp, status, cfg, method, bb in group_rows:
        if not runs:
            val = test = tp = "pending"
            figs = "not generated"
        else:
            val = fmt_ms([r["m"]["val_macro_f1"] for r in runs])
            test = fmt_ms([r["m"]["test_macro_f1"] for r in runs])
            tp = f"{runs[0]['m']['trainable_params']:,}"
            figs = f"[curves](figures/{slug}_curves.png), [cm](figures/{slug}_cm.png)"
        L.append(f"| {slug} | {exp} | {bb} | {METHOD_LABEL[method]} | {status} | {len(runs)} | {val} | {test} | "
                 f"{tp} | [{slug}.md]({slug}.md) | {figs} |")
    L.append(f"| e3_rank_study | E3 | MuRIL base cased | LoRA grid | {e3_status} | 1 per cell (seed 42) | "
             f"see report | see report | see report | [e3_rank_study.md](e3_rank_study.md) | "
             f"[heatmap](figures/e3_heatmap.png) |")
    L += ["", "Experiment ids: E1 xlmr full, E2 xlmr lora, E3 MuRIL LoRA grid, E4 xlmr adalora, E5 xlmr dora, "
          "E6 mbert lora, E7 indicbertv2 lora; others are 'extra'.", ""]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    groups, e3_rows = build_groups(runs)
    by_slug = {g["slug"]: g for g in groups}
    index_rows = []
    for g in groups:
        figs_ok = make_figs(g)
        md, status, exp = group_report(g, by_slug, figs_ok)
        (OUT / f"{g['slug']}.md").write_text(md, encoding="utf-8")
        cfg = group_cfg(g)
        prefix, bb = BACKBONE[cfg["model"]]
        index_rows.append((g["slug"], g["runs"], exp, status, cfg, cfg["method"], bb))
        print(f"wrote {g['slug']}.md ({len(g['runs'])} runs, {status})")
    ref_runs = by_slug.get("muril_lora", {}).get("runs", [])
    e3_md = e3_report(e3_rows, ref_runs)
    (OUT / "e3_rank_study.md").write_text(e3_md, encoding="utf-8")
    print(f"wrote e3_rank_study.md ({len(e3_rows)} rows)")
    e3_status = f"complete ({len(e3_rows)} cells)" if e3_rows else "pending"
    (OUT / "INDEX.md").write_text(index_md(index_rows, e3_status), encoding="utf-8")
    print("wrote INDEX.md")


if __name__ == "__main__":
    main()
