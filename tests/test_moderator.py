import csv, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import moderator as m

# occlusion: score = fraction of words equal to 'bad'
fn = lambda ts: np.array([t.split().count("bad") / max(len(t.split()), 1) if t != "." else 0 for t in ts])
pairs = m.occlusion("good bad fine", fn)
assert [w for w, _ in pairs] == ["good", "bad", "fine"] and max(pairs, key=lambda x: x[1])[0] == "bad"
h = m.highlight_html(pairs)
assert h.count("<span") == 1 and "bad</span>" in h
assert "&lt;" in m.highlight_html([("<b>", 0.5)])
n, ch = m.normalization("వె.ధ.వ")
assert n == "వెధవ" and ch and m.normalization("నన్ను చచ్చి")[1] is False
p = os.path.join(tempfile.mkdtemp(), "c.csv")
m.save_correction("a, \"b\"\nc", 0, p); m.save_correction("x", 1, p)
rows = list(csv.reader(open(p, encoding="utf-8", newline="")))
assert rows[0] == ["text", "label", "timestamp"] and rows[1][:2] == ["a, \"b\"\nc", "0"] and rows[2][1] == "1"
assert m.verdict(0.9, 0.2, 0.8) == "BLOCK" and m.verdict(0.1, 0.2, 0.8) == "ALLOW" and m.verdict(0.5, 0.2, 0.8) == "REVIEW"
if os.path.isfile(os.path.join(m.ROOT, "results", "clean_muril-base-cased_lp8_lr5e-5", "preds.csv")):
    pol = m.build_policy(os.path.join(m.ROOT, "results", "clean_muril-base-cased_lp8_lr5e-5", "preds.csv"))
    los = [pol["thr"][b][0] for b in m.BETAS]
    assert los[0] <= los[1] <= los[2] and all(pol["thr"][b][0] < pol["thr"][b][1] for b in m.BETAS), pol["thr"]
    pipe = m.load_tfidf_pipe()
    pr = m.p_tfidf(pipe, pol["platt"], ["ఆరాధ్య teri puku దే రే సుఖీభవ", "హాయ్ గుడ్ మార్నింగ్ సార్"])
    assert pr[0] > pr[1], pr
    print("thr", pol["thr"], pr)
print("test_moderator ok")
