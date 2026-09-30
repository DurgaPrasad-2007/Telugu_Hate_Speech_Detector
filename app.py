"""
Telugu abusive-content detector demo.
Run:  streamlit run app.py     (MODEL_DIR env var overrides the model; default = selected winner)
"""
import html
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from text_mask import mask_text  # noqa: E402
from infer import setup_logging  # noqa: E402

log = setup_logging()

st.set_page_config(page_title="Telugu Abuse Detector", page_icon="🛡️", layout="centered")


def load_json(rel):
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


FINAL = load_json("results/final_summary.json")
MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(
    ROOT, "results", FINAL.get("run_name", "baseline__muril__full__telugu"), "model")
BASE_MODEL = os.environ.get("BASE_MODEL_FOR_ADAPTER")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 900px;}
.hero h1 {font-size: 2rem; margin: 0; letter-spacing: -0.02em;}
.hero p {color: #475569; margin: .25rem 0 0;}
.pill {display:inline-block; font-size:.75rem; padding:.15rem .6rem; border-radius:999px;
       background:#e0e7ff; color:#1e3a8a; margin-right:.4rem; font-weight:600;}
.card {background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:1.1rem 1.3rem;
       box-shadow:0 1px 2px rgba(15,23,42,.04); margin-top:1rem;}
.card h4 {margin:0 0 .5rem; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; color:#64748b;}
.verdict {display:flex; align-items:center; justify-content:space-between; gap:1rem;}
.verdict .v {font-size:1.9rem; font-weight:700;}
.verdict .p {color:#475569; font-size:.95rem; text-align:right;}
.allow {border-left:6px solid #16a34a;} .allow .v {color:#15803d;}
.review {border-left:6px solid #d97706;} .review .v {color:#b45309;}
.block {border-left:6px solid #dc2626;} .block .v {color:#b91c1c;}
.bar {height:8px; border-radius:99px; background:#e2e8f0; margin-top:.8rem; position:relative;}
.bar i {display:block; height:100%; border-radius:99px;}
.allow .bar i {background:#16a34a;} .review .bar i {background:#d97706;} .block .bar i {background:#dc2626;}
.text {font-size:1.25rem; line-height:2; word-break:break-word;}
.bad {filter:blur(7px); background:rgba(220,38,38,.18); border-radius:4px; padding:0 3px;
      cursor:pointer; transition:filter .2s; user-select:none;}
.bad:hover {filter:none; user-select:auto;}
.note {color:#64748b; font-size:.82rem; margin-top:.6rem;}
.flow {display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.5rem;}
.step {flex:1 1 120px; background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:.8rem .9rem;}
.step b {display:block; font-size:.72rem; color:#1d4ed8; letter-spacing:.05em; text-transform:uppercase;}
.step span {font-size:.85rem; color:#334155; display:block; margin-top:.25rem;}
.kpis {display:grid; grid-template-columns:repeat(4,1fr); gap:.6rem; margin-top:.5rem;}
.kpi {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:.8rem;}
.kpi .n {font-size:1.5rem; font-weight:700;} .kpi .l {color:#64748b; font-size:.78rem;}
.decs {display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-top:.5rem;}
.dec {background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:1rem 1.1rem;}
.dec .dt {font-weight:700; font-size:.98rem; margin-bottom:.4rem; display:flex; gap:.5rem; align-items:center;}
.dec .num {background:#1d4ed8; color:#fff; border-radius:999px; min-width:1.4rem; height:1.4rem; display:inline-flex;
           align-items:center; justify-content:center; font-size:.75rem;}
.dec .dl {font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:#1d4ed8; font-weight:700; margin-top:.5rem;}
.dec p {margin:.15rem 0 0; font-size:.84rem; color:#334155; line-height:1.45;}
@media (max-width:760px){.decs{grid-template-columns:1fr;}}
@media (max-width:640px){.kpis{grid-template-columns:repeat(2,1fr);}}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>🛡️ Telugu Abuse Detector</h1>'
    '<p>Screens Telugu and Telugu-English comments and blurs the offending words.</p>'
    '<div style="margin-top:.6rem"><span class="pill">MuRIL</span>'
    '<span class="pill">Telugu / code-mixed</span><span class="pill">Human-in-the-loop</span></div></div>',
    unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_classifier(model_dir, base_model):
    from infer import HateSpeechClassifier
    return HateSpeechClassifier(model_dir, base_model_for_adapter=base_model)


try:
    clf = load_classifier(MODEL_DIR, BASE_MODEL)
except Exception:
    st.error("The model could not be loaded. Check MODEL_DIR and restart.")
    st.stop()

VERDICT = {"allow": "Allowed", "review": "Needs review", "block": "Blocked"}


def render_text(words):
    parts = []
    for w, _, flagged in words:
        e = html.escape(w)
        parts.append(f'<span class="bad" title="Hover to reveal">{e}</span>' if flagged else e)
    return " ".join(parts)


tab_an, tab_sys = st.tabs(["Analyze", "System overview"])

# ------------------------------------------------------------------ Analyze
with tab_an:
    with st.form("check", clear_on_submit=True, border=False):
        text = st.text_area("Comment", height=110, label_visibility="collapsed",
                            placeholder="Type or paste a Telugu comment...")
        go = st.form_submit_button("Analyze", type="primary", width="stretch")
    if go and text.strip():
        log.info("=" * 78)
        log.info("[app] Analyze clicked")
        r = clf.predict([text])[0]
        words = clf.explain(text)
        st.session_state["res"] = {"r": r, "words": words}
        log.info("[app] render: verdict card=%s, %d word(s) blurred in text card",
                 r["decision"].upper(), sum(1 for w in words if w[2]))
    elif go:
        st.warning("Please enter some text.")

    res = st.session_state.get("res")
    if res:
        r, words = res["r"], res["words"]
        d = r["decision"]
        n_flag = sum(1 for w in words if w[2])
        st.markdown(f"""
<div class="card verdict-card {d}">
  <div class="verdict"><div><h4>Verdict</h4><div class="v">{VERDICT[d]}</div></div>
  <div class="p">Abuse probability<br><b style="font-size:1.4rem;color:#0f172a">{r['probability']*100:.1f}%</b></div></div>
  <div class="bar"><i style="width:{r['probability']*100:.1f}%"></i></div>
</div>
<div class="card"><h4>Analyzed text · {n_flag} word{'s' if n_flag != 1 else ''} hidden</h4>
  <div class="text">{render_text(words)}</div>
  <div class="note">Words driving the score are blurred; hover over one to reveal it.
  Detected by removing each word and measuring the drop in abuse probability.</div></div>
""", unsafe_allow_html=True)
        st.caption("Trained on Telugu and Telugu-English comments only; other languages are out of scope.")

    with st.expander("Score a CSV file"):
        up = st.file_uploader("CSV with a `text` column", type="csv")
        if up is not None:
            try:
                df = pd.read_csv(up)
                if "text" not in df.columns:
                    st.error("The CSV needs a `text` column.")
                else:
                    log.info("=" * 78)
                    log.info("[app] CSV upload: %d rows", len(df))
                    results = clf.predict(df["text"].tolist())
                    log.info("[app] CSV verdicts: %s", pd.Series([r["decision"] for r in results]).value_counts().to_dict())
                    out = pd.DataFrame({
                        "text": [mask_text(str(t)) if r["decision"] != "allow" else t
                                 for t, r in zip(df["text"], results)],
                        "abuse_probability": [round(r.get("probability") or 0, 3) for r in results],
                        "verdict": [VERDICT[r["decision"]] for r in results],
                    })
                    st.dataframe(out, width="stretch", hide_index=True)
                    st.download_button("Download results", out.to_csv(index=False),
                                       "predictions.csv", "text/csv")
            except Exception:
                st.error("That file couldn't be processed. Make sure it's a valid CSV.")

# ---------------------------------------------------------- System overview
with tab_sys:
    import streamlit.components.v1 as components
    from architecture import load_facts, mermaid_offline, mermaid_online, mermaid_component, decisions
    F = load_facts()
    runs = F["runs"]

    st.markdown("##### How it was built")
    components.html(mermaid_component(mermaid_offline(F), 150, "a"), height=160)
    st.markdown("##### What happens to one comment")
    components.html(mermaid_component(mermaid_online(F), 300, "b"), height=320)

    st.markdown("##### Test-set performance")
    st.markdown(f"""
<div class="kpis">
 <div class="kpi"><div class="n">{FINAL.get('test_macro_f1',0):.3f}</div><div class="l">Macro-F1</div></div>
 <div class="kpi"><div class="n">{FINAL.get('precision',0):.3f}</div><div class="l">Precision</div></div>
 <div class="kpi"><div class="n">{FINAL.get('recall',0):.3f}</div><div class="l">Recall</div></div>
 <div class="kpi"><div class="n">{FINAL.get('ece') or 0:.3f}</div><div class="l">Calibration error (ECE)</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown("##### Model comparison (validation macro-F1, the selection metric)")
    short = {"xlm-roberta-base": "XLM-R", "muril-base-cased": "MuRIL"}
    names = {"full": "full fine-tune", "lora": "LoRA", "qlora": "QLoRA",
             "adalora": "AdaLoRA", "dora": "DoRA", "ia3": "IA3"}
    runs["run"] = [f"{short.get(m.split('/')[-1], m)} · {names.get(t, t)}"
                   for m, t in zip(runs["model"], runs["method"])]
    chart = runs.sort_values("val_macro_f1")[["run", "val_macro_f1"]].set_index("run")
    st.bar_chart(chart, horizontal=True, color="#1d4ed8", height=280)

    st.markdown("##### Why each design choice")
    cards = "".join(
        f'<div class="dec"><div class="dt"><span class="num">{i}</span>{html.escape(t)}</div>'
        f'<div class="dl">Why</div><p>{html.escape(why)}</p>'
        f'<div class="dl">Evidence</div><p>{html.escape(ev)}</p>'
        f'<div class="dl">Trade-off accepted</div><p>{html.escape(tr)}</p></div>'
        for i, (t, why, ev, tr) in enumerate(decisions(F), 1))
    st.markdown(f'<div class="decs">{cards}</div>', unsafe_allow_html=True)

    if F["raw"] and F["def"]:
        st.markdown("##### Attack test: 300 real abusive + 300 clean comments, every word disguised")
        rows = [{"Disguise": k, "Caught, no guard": f"{F['raw'][k]['abusive_caught_%']:.0f}%",
                 "Caught, with guard": f"{v['abusive_caught_%']:.0f}%",
                 "Clean wrongly blocked": f"{v['clean_wrongly_blocked_%']:.0f}%"}
                for k, v in F["def"].items() if k in F["raw"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
