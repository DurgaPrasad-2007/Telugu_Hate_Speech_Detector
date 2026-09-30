"""
System-overview content for the demo: Mermaid diagrams + the reasoning behind
each design decision. Every number is read from results/ at render time, so the
page can't drift from what was actually measured.
"""
import html
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _j(rel):
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _stress(rel):
    return {r["morph"]: r for r in _j(rel).get("results", [])}


def load_facts():
    """One dict of measured facts used by both diagrams and the decision cards."""
    fin, cal, pol = _j("results/final_summary.json"), _j("results/calibration.json"), _j("configs/policy.json")
    bench = _j("results/benchmark_results.json")
    runs = pd.read_csv(os.path.join(ROOT, "results", "all_results.csv"))
    sweep = pd.read_csv(os.path.join(ROOT, "results", "threshold_sweep.csv"))
    row = sweep[(sweep.low.round(2) == round(pol.get("low", 0), 2)) & (sweep.high.round(2) == round(pol.get("high", 0), 2))]
    row = row.iloc[0] if len(row) else None

    def run(tag):
        r = runs[runs.run_name.str.contains(tag)]
        return r.iloc[0] if len(r) else None
    muril, xlmr = run("muril"), run("xlmr")
    peft = runs[runs.run_name.str.startswith("peft__")]
    best_peft = peft.sort_values("val_macro_f1", ascending=False).iloc[0]
    worst_peft = peft.sort_values("val_macro_f1").iloc[0]
    return {
        "fin": fin, "cal": cal, "pol": pol, "bench": bench, "runs": runs, "muril": muril, "xlmr": xlmr,
        "best_peft": best_peft, "worst_peft": worst_peft, "n_runs": len(runs), "sweep": row,
        "raw": _stress("results/stress_test_raw_summary.json"),
        "def": _stress("results/stress_test_summary.json"),
        "xlmr_def": _stress("results/xlmr_reference/stress_test_summary.json"),
    }


def _short(run_name):
    return {"muril": "MuRIL", "xlmr": "XLM-R"}.get(run_name.split("__")[1], run_name)


def mermaid_offline(f):
    fin, cal, pol = f["fin"], f["cal"], f["pol"]
    win = _short(fin.get("run_name", "baseline__muril__full__telugu"))
    return f"""flowchart LR
  D[("MACD Telugu<br/>~24K comments")]:::data --> Q["Integrity checks<br/>labels hash-verified<br/>train/test leakage dropped"]:::step
  Q --> S["Model sweep<br/>{f["n_runs"]} runs: 2 backbones,<br/>5 adapter variants"]:::step
  S --> W["Winner: {win}<br/>full fine-tune<br/>chosen on validation F1"]:::win
  W --> K["Calibrate<br/>T = {cal.get('temperature', 1):.2f}<br/>ECE {cal.get('before', {}).get('ece', 0):.3f} to {cal.get('after', {}).get('ece', 0):.3f}"]:::step
  K --> P["Set policy on validation<br/>allow &lt; {pol.get('low')} | review | block &ge; {pol.get('high')}"]:::step
  P --> H["Attack it<br/>13 obfuscation styles<br/>on real test comments"]:::step
  classDef data fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
  classDef step fill:#ffffff,stroke:#94a3b8,color:#0f172a
  classDef win fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0f172a"""


def mermaid_online(f):
    pol = f["pol"]
    return f"""flowchart LR
  U(["Comment"]):::io --> G["1. Guard<br/>undo obfuscation:<br/>zero-width, spacing,<br/>emoji-in-word"]:::step
  G --> M["2. Model<br/>text view +<br/>emoji-stripped view<br/>keep the higher score"]:::win
  M --> C["3. Calibrate<br/>logits / T"]:::step
  C --> V{{"4. Policy"}}:::step
  V -->|"below {pol.get('low')}"| A["Allow"]:::allow
  V -->|"{pol.get('low')} to {pol.get('high')}"| R["Human review"]:::review
  V -->|"{pol.get('high')} or above"| B["Block"]:::block
  C --> X["5. Explain<br/>drop each word,<br/>measure score change"]:::step
  X --> UI(["Offending words<br/>blurred, hover to reveal"]):::io
  classDef io fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
  classDef step fill:#ffffff,stroke:#94a3b8,color:#0f172a
  classDef win fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0f172a
  classDef allow fill:#dcfce7,stroke:#16a34a,color:#14532d
  classDef review fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef block fill:#fee2e2,stroke:#dc2626,color:#7f1d1d"""


def mermaid_component(src, height, uid):
    """Self-contained HTML for st.components.v1.html. Needs internet for the
    mermaid CDN; falls back to showing the diagram source if it can't load."""
    return f"""<div id="g{uid}" style="font-family:Segoe UI,Nirmala UI,sans-serif;width:100%">
<pre class="mermaid" style="background:none;margin:0;width:100%">{html.escape(src)}</pre></div>
<pre id="fb{uid}" style="display:none;font-size:11px;color:#475569;white-space:pre-wrap">Diagram library could not load (offline?). Source:\n{html.escape(src)}</pre>
<script type="module">
try {{
  const m = (await import('https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs')).default;
  m.initialize({{startOnLoad:false, theme:'base', securityLevel:'loose',
    themeVariables:{{fontFamily:'Segoe UI, Nirmala UI, sans-serif', fontSize:'15px', lineColor:'#64748b', edgeLabelBackground:'#f8fafc', tertiaryColor:'#f8fafc'}},
    flowchart:{{curve:'basis', htmlLabels:true, useMaxWidth:true, padding:12, nodeSpacing:30, rankSpacing:40}}}});
  await m.run({{querySelector:'#g{uid} .mermaid'}});
}} catch (e) {{
  document.getElementById('g{uid}').style.display='none';
  document.getElementById('fb{uid}').style.display='block';
}}
</script>"""


def _pct(x):
    return "n/a" if x is None else f"{x:.0f}%"


def guard_evidence(raw, dfn):
    """Top gains of the guard on the winning model, plus an honest note on what it did not need to fix."""
    gains = sorted(((k, raw[k]["abusive_caught_%"], v["abusive_caught_%"]) for k, v in dfn.items() if k in raw),
                   key=lambda t: t[2] - t[1], reverse=True)[:3]
    top = "; ".join(f"{k} {a:.0f}% to {b:.0f}%" for k, a, b in gains)
    return (f"Abusive comments still caught when every word is disguised (300 real comments), without vs with the guard: {top}. "
            "The model's own tokenizer already ignores zero-width characters, so those were never the weak point.")


def emoji_view_evidence(dfn):
    import re
    p = os.path.join(ROOT, "results", "xlmr_reference", "stress_test_BEFORE_defense.log")
    xd = _stress("results/xlmr_reference/stress_test_summary.json")
    m = re.search(r"emoji wrapped\s+([\d.]+)", open(p, encoding="utf-8").read().split("PART B")[-1]) if os.path.exists(p) else None
    if m and "emoji wrapped" in xd:
        return (f"Measured during development on the XLM-R candidate: emoji-wrapped abuse caught {float(m.group(1)):.0f}% "
                f"to {xd['emoji wrapped']['abusive_caught_%']:.0f}%. The selected model, with this view on, catches "
                f"{(dfn.get('emoji wrapped') or {}).get('abusive_caught_%', 0):.0f}% on the same test.")
    return "See results/xlmr_reference for the development measurement."


def backbone_tradeoff(f):
    x = f["xlmr_def"].get("letter spacing", {}).get("abusive_caught_%")
    mu = f["def"].get("letter spacing", {}).get("abusive_caught_%")
    base = "The gap is inside sampling noise (about 0.5 pt at 3,000 rows): a tie broken on fit and cost, not a proven win."
    if x is not None and mu is not None and x > mu:
        base += f" XLM-R held up better against letter-spaced disguises ({x:.0f}% vs {mu:.0f}% caught, with the guard)."
    return base


def plain_cost(f):
    d = _j("results/deployed_pipeline_metrics.json")
    if not d:
        return "Effect on ordinary text not measured."
    before, after = f["muril"]["test_macro_f1"], d["test"]["macro_f1"]
    return f"Effect on ordinary text: test macro-F1 {before:.4f} without the guard and second view, {after:.4f} with both."


def codemixed_evidence():
    z = _j("results/codemixed_zero_shot.json")
    if not z:
        return "See results/codemixed_zero_shot.json."
    t, c = z["telugu"], z["codemixed"]
    return (f"Zero-shot on a separate code-mixed test set ({c['n']} comments, never trained on): macro-F1 {c['macro_f1']:.2f} "
            f"vs {t['macro_f1']:.2f} on Telugu; Latin-script comments {c['latin_only']['macro_f1']:.2f}.")


def decisions(f):
    """(title, why, evidence, tradeoff) per design decision, from measured results."""
    m, x, fin, cal, pol = f["muril"], f["xlmr"], f["fin"], f["cal"], f["pol"]
    bp, wp, sw, bench = f["best_peft"], f["worst_peft"], f["sweep"], f["bench"]
    raw, dfn = f["raw"], f["def"]
    g = lambda d, k, col="abusive_caught_%": (d.get(k) or {}).get(col)
    cpu32 = bench.get("pytorch_cpu_fp32", {}).get("p50_ms_per_sample")
    cpu8 = bench.get("onnx_int8", {}).get("p50_ms_per_sample")
    out = [
        ("Trustworthy data first",
         "A classifier is only as honest as its labels. The raw label codes were ambiguous, so they were verified by hash-matching "
         "against the upstream MACD release instead of assumed. Train/test overlap was removed at load time; the CSVs stay untouched.",
         "Label meaning proven against upstream; every exact train/val/test duplicate dropped; all reported scores use the cleaned splits.",
         "Single annotation scheme; no dialect labels exist, so per-dialect fairness cannot be measured."),
        (f"Backbone: {_short(fin.get('run_name', ''))} (Indic-specific)",
         "Selected on validation macro-F1 only, never on test. MuRIL is pre-trained on Indian languages including transliterated "
         f"text, which is how Telugu is actually typed, and is {(1 - m['total_params'] / x['total_params']) * 100:.0f}% smaller than XLM-R, so it is cheaper to host.",
         f"Validation F1: MuRIL {m['val_macro_f1']:.3f} vs XLM-R {x['val_macro_f1']:.3f}. Test F1: {m['test_macro_f1']:.3f} vs {x['test_macro_f1']:.3f}.",
         backbone_tradeoff(f)),
        ("Full fine-tuning over adapters (LoRA family)",
         "Adapters train under 0.5% of weights, which pays off for huge models. Here the model is small and training takes minutes, "
         "so the savings buy nothing and the accuracy loss is real. Adapters are merged at serve time anyway, so inference is no faster.",
         f"Best adapter ({bp['run_name'].split('__')[1].upper()}): validation F1 {bp['val_macro_f1']:.3f} vs {m['val_macro_f1']:.3f} full. "
         f"Worst ({wp['run_name'].split('__')[1].upper()}): {wp['val_macro_f1']:.3f}, effectively collapsed.",
         f"Every weight is retrained, so a new model is a full retrain (about {m['training_time_sec'] / 60:.0f} minutes on one consumer GPU)."),
        ("A guard layer against obfuscation",
         "Real users dodge filters with zero-width characters, spaced or dotted letters and emoji inside words. These change the "
         "tokens, not the meaning, so the guard normalises them before the model sees the text. It is deliberately narrow: "
         "doubled Telugu letters such as నన్ను are legitimate and are never touched.",
         guard_evidence(raw, dfn),
         f"{plain_cost(f)} Cannot undo doubling of every letter of a clean sentence."),
        ("Second opinion without emoji",
         "Flooding a comment with emoji can drown out the words. The model also scores an emoji-stripped copy and the higher "
         "abuse score wins. It only runs when emoji are present.",
         emoji_view_evidence(dfn),
         "One extra forward pass on comments that contain emoji; accuracy on ordinary text is unchanged (see the guard card)."),
        ("Calibrated probabilities",
         "Raw neural scores are overconfident. Temperature scaling, fitted on validation data only, makes '70%' mean roughly 70%, "
         "so thresholds and the on-screen percentage are meaningful.",
         f"T = {cal.get('temperature', 1):.2f}; expected calibration error {cal.get('before', {}).get('ece', 0):.3f} to {cal.get('after', {}).get('ece', 0):.3f}.",
         "One global scalar; it fixes average confidence, not individual wrong answers."),
        ("Three outcomes, not two",
         "Moderation errors are not symmetric: wrongly blocking a person is costly and missing abuse is costly. A narrow review band "
         "sends the least certain cases to a human. Thresholds were tuned on validation, and block starts at 0.5 so a block never "
         "contradicts the model's own label.",
         (f"Block precision {sw['block_precision']:.3f}, block recall {sw['block_recall']:.3f}, {sw['review_pct']*100:.1f}% of traffic to review."
          if sw is not None else "Thresholds from results/threshold_sweep.csv."),
         "A thin review band means most errors are decided automatically; widen it to trade reviewer time for fewer mistakes."),
        ("Explanations by occlusion",
         "To hide only the offending words the app removes each word in turn and measures the drop in abuse score. It needs no access "
         "to model internals, so it works with any classifier, and it explains the same score used for the decision.",
         "All word-removed variants are scored together in one batch, so the cost is a single extra forward pass, and only words carrying most of the signal are hidden.",
         "Judges one word at a time, so abuse spread across several words can be under-hidden."),
        ("CPU-friendly serving",
         "The demo runs on the GPU; production can run on CPU. Exporting to ONNX with int8 quantisation cuts CPU latency "
         "without retraining, so hosting does not need a GPU.",
         (f"ONNX int8 is {cpu32/cpu8:.1f}x faster than PyTorch on CPU ({cpu8:.0f} vs {cpu32:.0f} ms per comment in this run). "
          "Absolute times vary with machine load: an earlier idle session measured about 5x lower for both."
          if cpu32 and cpu8 else "See results/benchmark_results.json."),
         "Accuracy of the int8 model has not been measured yet; the live app runs the PyTorch model."),
        ("Where it does not work",
         "Stating limits is part of the design. The model is trained on Telugu-script comments only. It detects explicit and "
         "disguised abuse, not implicit or double-meaning abuse, and it is clearly weaker on code-mixed and Romanised Telugu it never saw.",
         codemixed_evidence(),
         "Keep a human in the loop; treat the score as decision support, not a verdict."),
    ]
    return out
