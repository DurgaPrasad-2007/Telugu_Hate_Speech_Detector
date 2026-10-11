"""Expectation vs outcome for every model tested.

Numbers come from results/study/{selection,significance}.json and baseline_classical.json,
the expectations are the stated hypotheses from the paper / supervisor plan / common practice.
Used by app.py (Models tab) and `python src/expectations.py` (writes docs/EXPECTATIONS.md).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _j(rel):
    p = os.path.join(ROOT, rel)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}


def _find(rank, model, method, r=None, a=None, ep=None):
    for x in rank:
        if model not in str(x.get("model", "")) or x.get("method") != method:
            continue
        if r is not None and int(x.get("lora_r") or -1) != r:
            continue
        if a is not None and int(x.get("lora_alpha") or -1) != a:
            continue
        if ep is not None and int(x.get("epochs") or -1) != ep:
            continue
        return x
    return None


def _f(x, k="test_mean"):
    return f"{x[k]:.4f}" if x else "n/a"


def _sig(sig, name):
    for c in sig.get("comparisons", []):
        if name in c.get("comparison", "") or name in c.get("run", ""):
            return c
    return None


def _cfg():
    a = _j("models/best/adapter_config.json")
    h = _j("results/study/xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024/metrics.json").get("hyperparams", {})
    return {
        "r": a.get("r", "?"), "alpha": a.get("lora_alpha", "?"), "do": a.get("lora_dropout", "?"),
        "mods": "/".join(sorted(a.get("target_modules", []))) or "query/value",
        "lr": h.get("lr", "?"), "ep": h.get("epochs", "?"), "bs": h.get("train_bs", "?"),
        "ml": h.get("max_length", "?"), "pat": h.get("early_stop_patience", "?"),
    }


def _usage(c):
    common = (f"Binary head (abusive / non-abusive) on NFC-normalised Telugu text, max length {c['ml']}, batch {c['bs']}, "
              f"AdamW, cosine schedule, early stopping (patience {c['pat']}) on validation macro-F1.")
    lora = (f"Adapters on {c['mods']} attention matrices (r={c['r']}, alpha={c['alpha']}, dropout {c['do']}), "
            f"classifier head trained, base frozen, lr {c['lr']}, up to {c['ep']} epochs. ")
    return {
        "XLM-R full fine-tune (E1)": (
            "General multilingual encoder (100 languages, CC100) for classification, tagging, QA.",
            "All weights trained with lr 2e-5 for up to 5 epochs. " + common),
        "XLM-R LoRA (E2, selected)": (
            "Same encoder, adapted cheaply: one frozen base shared by many tasks, a few MB per task.",
            lora + common + " Shipped as a 3.4 MB adapter in models/best, with the input guard, temperature scaling "
            "and allow/review/block thresholds on top."),
        "LoRA rank study (E3, MuRIL grid)": (
            "Choosing adapter capacity: rank sets how much the adapter can change the base model.",
            "12-cell grid, rank 4/8/16/32 x alpha 16/32/64, lr 2e-4, one seed on MuRIL; best cell by validation reused elsewhere."),
        "XLM-R AdaLoRA (E4)": (
            "Spends a fixed rank budget unevenly: prunes unimportant weight matrices, keeps important ones.",
            f"Start rank {int(c['r']) * 3 // 2}, target rank {c['r']}, query/value, rank-allocation step each optimizer step "
            "(fixed after the first run collapsed), same data and schedule as LoRA."),
        "XLM-R DoRA (E5)": (
            "LoRA variant that splits each weight into magnitude and direction to train closer to full fine-tuning.",
            lora.replace("(r=", "(DoRA, r=") + common),
        "mBERT LoRA (E6)": (
            "Original multilingual BERT (104 Wikipedia languages): the classic cross-lingual baseline.",
            lora + common),
        "IndicBERTv2 LoRA (E7)": (
            "BERT pretrained on Indic-language corpora (23 languages incl. Telugu) for Indian-language tasks.",
            "Original IndicBERT is gated, so IndicBERTv2-MLM-only was used. " + lora + common),
        "MuRIL full fine-tune / LoRA": (
            "Google's BERT pretrained on 17 Indian languages incl. transliterated text; the natural choice for Telugu.",
            "Run both as full fine-tuning (lr 2e-5) and as LoRA with the same settings as XLM-R. " + common),
        "MuRIL DoRA / AdaLoRA": (
            "Same variants as above, on the Indian-language backbone.",
            "Same r/alpha/lr as MuRIL LoRA; used to check the method ranking holds on a second backbone."),
        "Paper config (r4, alpha32, dropout 0.1, bs 8, len 128, 5 ep)": (
            "The Indonesian paper's published recipe (IndoBERT + LoRA, r=4, alpha=32).",
            "Applied unchanged to XLM-R on Telugu to test whether the paper's settings transfer."),
        "Supervisor start (r16, alpha32, dropout 0.1, 5 ep)": (
            "The supervisor's suggested starting configuration for XLM-R + LoRA.",
            "Applied as written (r=16, alpha=32, dropout 0.1, length 128, 5 epochs) on the Telugu data."),
        "TF-IDF + LinearSVC (added baseline)": (
            "Classical text classifier: sparse character and word n-grams with a linear SVM; tiny and CPU-only.",
            "Character 2-5 grams plus word 1-2 grams, C chosen on validation, the same input guard, Platt-scaled scores "
            "and thresholds fitted on validation; lives in models/tfidf and is selectable in the app."),
        "QLoRA and IA3 (earlier session)": (
            "QLoRA: LoRA on a 4-bit quantised base to cut memory. IA3: learns tiny scaling vectors instead of matrices.",
            "Run once with the earlier schedule (lr 1e-4, 3 epochs); not repeated with the corrected setup."),
        "INT8 ONNX export": (
            "Runs the model on ordinary CPUs without PyTorch or a GPU, with smaller weights.",
            "Merged adapter into the base, exported to ONNX, dynamic per-channel INT8 (AVX2); accuracy checked against fp32 "
            "on the same val/test sets."),
    }


def build_rows():
    sel, sig, bl = _j("results/study/selection.json"), _j("results/study/significance.json"), _j("results/study/baseline_classical.json")
    rk = sel.get("ranking", [])
    xf, xl = _find(rk, "xlm-roberta", "full"), _find(rk, "xlm-roberta", "lora", 8, 64, 10)
    xd, xa = _find(rk, "xlm-roberta", "dora"), _find(rk, "xlm-roberta", "adalora")
    mf, ml = _find(rk, "muril", "full"), _find(rk, "muril", "lora", 8, 64, 10)
    md, ma = _find(rk, "muril", "dora"), _find(rk, "muril", "adalora")
    mb, iv = _find(rk, "multilingual-cased", "lora"), _find(rk, "IndicBERTv2", "lora")
    pc, ps = _find(rk, "xlm-roberta", "lora", 4, 32, 5), _find(rk, "xlm-roberta", "lora", 16, 32, 5)
    tf = (bl.get("test") or {}).get("macro_f1")
    s_ada, s_tf = _sig(sig, "AdaLoRA"), _sig(sig, "TF-IDF")
    s_mb = _sig(sig, "mBERT")
    pct = (100 * xl["trainable_params_mean"] / xf["trainable_params_mean"]) if xl and xf else None
    gpu = (xf["peak_gpu_mb_mean"] / xl["peak_gpu_mb_mean"]) if xl and xf else None
    R = []

    def add(model, expected, happened, verdict):
        R.append({"Model": model, "What we expected": expected, "What happened": happened, "Verdict": verdict})

    add("XLM-R full fine-tune (E1)",
        "Strong baseline, the accuracy ceiling the adapters are measured against.",
        f"Test macro-F1 {_f(xf)}: top group, but tied with several cheaper models. Costs "
        f"{xf['trainable_params_mean']/1e6:.0f} M trained weights and {xf['peak_gpu_mb_mean']/1000:.1f} GB GPU." if xf else "n/a",
        "Met as baseline; not meaningfully better than the rest")
    add("XLM-R LoRA (E2, selected)",
        "Paper/supervisor: near full-FT accuracy while training only 1-2% of the weights.",
        f"Test {_f(xl)}, within noise of full FT, training {pct:.2f}% of the weights with {gpu:.1f}x less GPU memory. "
        "Lower abusive recall than full FT (more cautious)." if xl and pct else "n/a",
        "Met (and cheaper than the paper's 1-2%)")
    add("LoRA rank study (E3, MuRIL grid)",
        "A clear optimal rank among 4/8/16/32 (supervisor: 'find optimal rank').",
        "r=8 is best on validation but the whole grid spans about one point on one seed; alpha=64 helped at every rank. "
        "No rank is distinguishable from the others.",
        "Partly failed: rank barely matters, alpha does")
    add("XLM-R AdaLoRA (E4)",
        "Adaptive rank allocation should beat uniform LoRA (supervisor: 'particularly interesting').",
        f"Test {_f(xa)}, worse than LoRA by {xl['test_mean']-xa['test_mean']:.3f}"
        + (f" (McNemar p={s_ada['mcnemar_p']:.4f})" if s_ada else "")
        + f"; about {xa['train_time_s_mean']/xl['train_time_s_mean']:.1f}x the training time. An earlier run collapsed to 0.34 "
        "before the rank-allocation step was fixed." if xa and xl else "n/a",
        "Failed expectation")
    add("XLM-R DoRA (E5)",
        "Magnitude/direction split should improve on LoRA, especially at low rank.",
        f"Test {_f(xd)} vs LoRA {_f(xl)}: a tie on XLM-R; on MuRIL it was lower ({_f(md)} vs {_f(ml)}). "
        f"About {xd['train_time_s_mean']/xl['train_time_s_mean']:.1f}x the training time." if xd and xl else "n/a",
        "Failed expectation: no gain over LoRA")
    add("mBERT LoRA (E6)",
        "Older multilingual model; expected to trail XLM-R.",
        f"Test {_f(mb)}, the lowest of the backbones"
        + (f"; selected model ahead by {s_mb['delta_macro_f1_A_minus_other']:.3f}, McNemar p={s_mb['mcnemar_p']:.3f} (borderline)." if s_mb else "."),
        "Met")
    add("IndicBERTv2 LoRA (E7)",
        "Indic-specialised model should beat general multilingual ones on Telugu.",
        f"Test {_f(iv)} vs XLM-R LoRA {_f(xl)}: tied, not better. Original IndicBERT was gated, so v2 was used.",
        "Failed expectation: no Indic advantage")
    add("MuRIL full fine-tune / LoRA",
        "Indian-language pretraining should outperform XLM-R on Telugu.",
        f"Full {_f(mf)}, LoRA {_f(ml)} vs XLM-R {_f(xf)} / {_f(xl)}: a tie. MuRIL LoRA missed the selection tie set by 0.0004.",
        "Failed expectation: no gain over XLM-R")
    add("MuRIL DoRA / AdaLoRA",
        "Same as the XLM-R variants.", f"DoRA {_f(md)}, AdaLoRA {_f(ma)}: both below MuRIL LoRA.", "Failed expectation")
    add("Paper config (r4, alpha32, dropout 0.1, bs 8, len 128, 5 ep)",
        "The paper's tuned settings should transfer and be near-optimal.",
        f"Test {_f(pc)} vs our tuned LoRA {_f(xl)}: slightly lower. Cause not isolated (several settings differ).",
        "Partly failed: close, not best")
    add("Supervisor start (r16, alpha32, dropout 0.1, 5 ep)",
        "A sensible starting point from the supervisor's plan.",
        f"Test {_f(ps)}; two of three seeds peaked on the last allowed epoch, so the 5-epoch cap probably cost accuracy.",
        "Partly failed: under-trained")
    add("TF-IDF + LinearSVC (added baseline)",
        "A classical floor well below the transformers.",
        f"Test {tf:.4f}"
        + (f", no detectable difference from the selected model (p={s_tf['mcnemar_p']:.2f})" if s_tf else "")
        + "; 30 MB and about 1.7 ms per comment on CPU." if tf else "n/a",
        "Failed expectation, and the most important result")
    add("QLoRA and IA3 (earlier session)",
        "Cheaper still than LoRA.",
        "Run only with the earlier under-trained settings (QLoRA 0.870, IA3 0.726); not re-tested with the corrected schedule, "
        "so no conclusion is drawn.",
        "Inconclusive")
    add("INT8 ONNX export",
        "A small accuracy loss for a 4x smaller CPU model.",
        "Per-tensor INT8 collapsed (0.63 on the selected model; 0.37 on the previously committed MuRIL file). "
        "Per-channel INT8 costs about 1.2 points of test macro-F1.",
        "Failed first, fixed after")
    use = _usage(_cfg())
    for r in R:
        u, cst = use.get(r["Model"], ("-", "-"))
        r["Typical use"], r["How we customised it"] = u, cst
    return R


def main():
    rows = build_rows()
    out = ["# Expectation vs outcome, per model", "",
           "Generated by `python src/expectations.py` from `results/study/*.json`. Expectations are the hypotheses "
           "stated by the paper, the supervisor's plan or common practice; outcomes are measured.", "",
           "| Model | Typical use | How we customised it | What we expected | What happened | Verdict |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[k]).replace("|", "/") for k in
                   ("Model", "Typical use", "How we customised it", "What we expected", "What happened", "Verdict")) + " |")
    open(os.path.join(ROOT, "docs", "EXPECTATIONS.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"wrote docs/EXPECTATIONS.md ({len(rows)} rows)")


if __name__ == "__main__":
    main()
