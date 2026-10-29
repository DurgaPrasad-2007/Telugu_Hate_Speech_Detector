"""CPU-only: write preds.csv (split,text,label,p_abusive; label 1=abusive internal) for a trained run dir.
usage: THS_DATA_DIR=data/clean python src/dump_probs.py results/<run> [max_length]"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU is reserved for training
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

run, ml = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 128
mdir = os.path.join(run, "model")
tok = AutoTokenizer.from_pretrained(mdir)
if os.path.exists(os.path.join(mdir, "adapter_config.json")):
    from peft import AutoPeftModelForSequenceClassification
    model = AutoPeftModelForSequenceClassification.from_pretrained(mdir).merge_and_unload()
else:
    model = AutoModelForSequenceClassification.from_pretrained(mdir)
model.eval()
torch.set_num_threads(8)
out = []
for sp in ("val", "test"):
    df = load_split("telugu", sp, "normalized")
    ps = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            enc = tok(df.text[i:i + 32].tolist(), truncation=True, padding=True, max_length=ml, return_tensors="pt")
            ps.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
    out.append(pd.DataFrame({"split": sp, "text": df.text, "label": df.label, "p_abusive": np.concatenate(ps)}))
    print(sp, len(df), flush=True)
pd.concat(out).to_csv(os.path.join(run, "preds.csv"), index=False)
