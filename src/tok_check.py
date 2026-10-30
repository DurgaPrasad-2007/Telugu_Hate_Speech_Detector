"""Tokens/word + UNK rate on 200 train comments for several tokenizers. usage: python src/tok_check.py"""
import os, sys, json
import pandas as pd
from transformers import AutoTokenizer
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = pd.read_csv(os.path.join(ROOT, "data", "clean", "telugu_train.csv")).sample(200, random_state=0)
words = sum(len(t.split()) for t in df.text); out = {}
for m in ["google/muril-base-cased", "FacebookAI/xlm-roberta-base", "jhu-clsp/mmBERT-base", "ai4bharat/IndicBERTv2-MLM-only", "sarvamai/sarvam-1"]:
    try:
        tk = AutoTokenizer.from_pretrained(m)
        ids = [tk(t, add_special_tokens=False)["input_ids"] for t in df.text]
        unk = sum(i == tk.unk_token_id for x in ids for i in x) if tk.unk_token_id is not None else None
        out[m] = dict(tok_per_word=sum(map(len, ids)) / words, unk=unk, n_tok=sum(map(len, ids)), cls=type(tk).__name__)
    except Exception as e:
        out[m] = dict(error=repr(e)[:200])
    print(m, out[m], flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "sweep", "tok_check.json"), "w"), indent=1)
