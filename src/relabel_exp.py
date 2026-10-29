"""Relabel experiment: apply FIRM (borderline==0) model-authored labels to clean train rows (match by text);
val/test untouched. Retrains deployed TF-IDF recipe, reports clean-test macro-F1 vs baseline + paired bootstrap,
and the sheet's held-out (val/test) rows. usage: python src/relabel_exp.py"""
import os, sys, shutil
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC, OUT = os.path.join(ROOT, "data", "clean"), os.path.join(ROOT, "data", "clean_relabel")
os.makedirs(OUT, exist_ok=True)
for f in ("telugu_val.csv", "telugu_test.csv"): shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f))
sheet = pd.read_csv(os.path.join(SRC, "label_audit_gold_review_labeled.csv"), encoding="utf-8")
firm = sheet[sheet.borderline == 0].copy(); firm["text"] = firm.text.astype(str).str.strip()
tr = pd.read_csv(os.path.join(SRC, "telugu_train.csv"), encoding="utf-8"); tr["text"] = tr.text.astype(str)
key = tr.text.str.strip(); m = dict(zip(firm.text, firm.human_label))
hit = key.isin(m); chg = hit & (key.map(m) != tr.label)
print(f"sheet {len(sheet)}, firm {len(firm)}; firm rows in train {hit.sum()} (rows incl dup texts), labels changed {chg.sum()}")
tr.loc[hit, "label"] = key[hit].map(m).astype(int)
tr.to_csv(os.path.join(OUT, "telugu_train.csv"), index=False, encoding="utf-8")
heldout = {s: firm.text.isin(pd.read_csv(os.path.join(SRC, f"telugu_{s}.csv"), encoding="utf-8").text.astype(str).str.strip()).sum() for s in ("val", "test")}
print("firm sheet rows found in val/test:", heldout, "| all sheet rows in train:", sheet.text.astype(str).str.strip().isin(key).sum())

os.environ["THS_DATA_DIR"] = SRC
import negation_aug
from tfidf_classifier import guard
from conformal_policy import split_val
from significance import macro_f1_counts
negation_aug.fit_tfidf(OUT, os.path.join(ROOT, "results", "relabel_tfidf", "relabel.joblib"))
import data_utils; data_utils.DATA_DIR = SRC
te = data_utils.load_split("telugu", "test", "normalized"); y = te.label.values
pb = joblib.load(os.path.join(ROOT, "models", "tfidf", "tfidf_best.joblib")).decision_function([guard(t)[0] for t in te.text]) > 0
pc = joblib.load(os.path.join(ROOT, "results", "relabel_tfidf", "relabel.joblib")).decision_function([guard(t)[0] for t in te.text]) > 0
pb, pc = pb.astype(int), pc.astype(int)
idx = np.random.default_rng(0).integers(0, len(y), size=(10000, len(y)), dtype=np.int32)
boot = lambda p: np.concatenate([macro_f1_counts(y[idx[s:s+500]], p[idx[s:s+500]]) for s in range(0, 10000, 500)])
d = boot(pc) - boot(pb); lo, hi = np.percentile(d, [2.5, 97.5])
fb, fc = f1_score(y, pb, average="macro"), f1_score(y, pc, average="macro")
print(f"test n={len(y)} baseline {fb:.4f} relabel {fc:.4f} delta {fc-fb:+.4f} CI [{lo:+.4f},{hi:+.4f}]")
