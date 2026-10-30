"""Distill a MuRIL teacher into a 4-layer truncated MuRIL student, export INT8 ONNX, measure CPU batch-1 latency + clean-test F1.
usage: python src/distill_sweep.py <teacher_run_dir> [n_layers=4]   (teacher run = results/... with LoRA adapter or full model)
Needs a free GPU for distill.py. Writes results/sweep/distill/result.json."""
import os, sys, json, time, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
teacher_run, L = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
W = os.path.join(ROOT, "results", "sweep", "distill"); os.makedirs(W, exist_ok=True)
os.environ["THS_DATA_DIR"] = os.path.join(ROOT, "data", "clean")
env = {**os.environ}
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score
tdir, sdir, odir = (os.path.join(W, x) for x in ("teacher", "student_init", "student"))
mdir = os.path.join(ROOT, teacher_run, "model")
if not os.path.exists(tdir):  # merged teacher
    if os.path.exists(os.path.join(mdir, "adapter_config.json")):
        from peft import AutoPeftModelForSequenceClassification
        m = AutoPeftModelForSequenceClassification.from_pretrained(mdir).merge_and_unload()
    else: m = AutoModelForSequenceClassification.from_pretrained(mdir)
    m.save_pretrained(tdir); AutoTokenizer.from_pretrained(mdir).save_pretrained(tdir)
if not os.path.exists(sdir):  # student = pretrained MuRIL embeddings + first L layers
    base = AutoModelForSequenceClassification.from_pretrained("google/muril-base-cased", num_labels=2)
    base.bert.encoder.layer = base.bert.encoder.layer[:L]; base.config.num_hidden_layers = L
    base.save_pretrained(sdir); AutoTokenizer.from_pretrained("google/muril-base-cased").save_pretrained(sdir)
if not os.path.exists(odir):
    subprocess.run([PY, "src/distill.py", "--teacher", tdir, "--student", sdir, "--out", odir, "--epochs", "5", "--bs", "32",
                    "--lr", "5e-5", "--max_length", "128", "--alpha", "0.5", "--beta", "0.5"], cwd=ROOT, check=True, env=env)
ox = os.path.join(W, "onnx")
if not os.path.exists(ox + "_int8"):
    subprocess.run([PY, "src/export_onnx.py", "--model_dir", odir, "--out_dir", ox, "--quantize"], cwd=ROOT, check=True, env=env)

sys.path.insert(0, HERE)
from data_utils import load_split
test = load_split("telugu", "test", "normalized"); y = test.label.values
from optimum.onnxruntime import ORTModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(odir)
res = {}
def lat_f1(name, predict):
    t = []; ps = []
    for i, x in enumerate(test.text.tolist()):
        s = time.perf_counter(); ps.append(predict(x)); t.append(time.perf_counter() - s)
    res[name] = dict(macro_f1=f1_score(y, (np.array(ps) >= .5).astype(int), average="macro"), ms_mean=1e3 * np.mean(t), ms_p95=1e3 * np.percentile(t, 95))
    print(name, res[name], flush=True)
torch.set_num_threads(4)
def mk_ort(path):
    m = ORTModelForSequenceClassification.from_pretrained(path, file_name="model_quantized.onnx" if path.endswith("_int8") else None)
    def f(x):
        e = tok(x, truncation=True, max_length=128, return_tensors="pt"); return torch.softmax(torch.tensor(m(**e).logits), -1)[0, 1].item()
    return f
def mk_pt(path):
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    def f(x):
        e = tok(x, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad(): return torch.softmax(m(**e).logits, -1)[0, 1].item()
    return f
lat_f1("student_int8_onnx", mk_ort(ox + "_int8"))
lat_f1("student_fp32_torch", mk_pt(odir))
lat_f1("teacher_fp32_torch", mk_pt(tdir))
json.dump(res, open(os.path.join(W, "result.json"), "w"), indent=1)
