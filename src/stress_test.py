"""
Brutal obfuscation stress test, run against the REAL model.

Part A: hand-written insults morphed in ways people use to dodge filters.
Part B: real test-set comments the model already handles correctly, morphed
        at every word; reports how often the verdict survives.
Usage: python src/stress_test.py [--n 300] [--model_dir ...]
"""
import argparse
import json
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split  # noqa: E402
from infer import HateSpeechClassifier, setup_logging  # noqa: E402
from text_mask import mask_text  # noqa: E402

ZWJ, ZWNJ = "‍", "‌"
EMO = ["😂", "🤣", "🔥", "😡", "🤬", "💩", "👎", "🙏"]
LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})
# Latin look-alikes for Telugu-script-adjacent obfuscation (Latin digits/letters mixed in)
R = random.Random(7)


def w_zwj(w):        return ZWJ.join(w)
def w_zwnj(w):       return ZWNJ.join(w)
def w_spaced(w):     return " ".join(w)
def w_dotted(w):     return ".".join(w)
def w_dashed(w):     return "-".join(w)
def w_emoji_in(w):   return R.choice(EMO).join(w)
def w_emoji_wrap(w): return f"{R.choice(EMO)}{w}{R.choice(EMO)}"
def w_repeat(w):     return "".join(c * R.choice([2, 3, 4]) for c in w)
def w_leet(w):       return w.lower().translate(LEET)
def w_star(w):       return w[0] + "*" * (len(w) - 2) + w[-1] if len(w) > 2 else w
def w_upper(w):      return w.upper()
def w_combo(w):      return w_emoji_in(w_zwj(w_repeat(w)))
def w_trailing(w):   return w + "".join(R.choice(EMO) for _ in range(3))


WORD_MORPHS = {
    "zero-width joiner": w_zwj, "zero-width non-joiner": w_zwnj,
    "letter spacing": w_spaced, "dots between letters": w_dotted,
    "dashes between letters": w_dashed, "emoji inside word": w_emoji_in,
    "emoji wrapped": w_emoji_wrap, "char repetition": w_repeat,
    "leetspeak digits": w_leet, "star-censored": w_star,
    "UPPERCASE": w_upper, "trailing emoji spam": w_trailing,
    "COMBO (repeat+ZWJ+emoji)": w_combo,
}


def morph_text(text, fn):
    return " ".join(fn(w) for w in text.split())


# (label for the demo, text) -- moderate insults, Telugu script and Romanized
HAND = [
    "నువ్వు చాలా చెత్తవాడివి, నీతో మాట్లాడటం కూడా వృథా.",
    "పనికిమాలిన వెధవ, నోరు మూసుకో",
    "వీడొక దరిద్రుడు, ఎదవ",
    "nuvvu chala vedhava, ninnu chusthe asahyam",
    "ఒరేయ్ వెధవ 😡 నీ మొహం చూపించకు",
]


def verdict_stats(clf, texts):
    rs = clf.predict(texts)
    return [r["decision"] for r in rs], [r["probability"] for r in rs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="results/baseline__muril__full__telugu/model")
    ap.add_argument("--no_defend", action="store_true", help="disable src/canon.py to measure the raw model")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="results/stress_test")
    args = ap.parse_args()

    setup_logging("WARNING")  # keep the table readable; demo uses the app for live logs
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    clf = HateSpeechClassifier(args.model_dir, defend=not args.no_defend)

    print("\n=== PART A: hand-written insults, key words morphed ===")
    hand_rows = []
    for base in HAND:
        d0, p0 = verdict_stats(clf, [base])
        print(f"\nORIGINAL  {mask_text(base)}  -> {d0[0].upper()} ({p0[0]:.2f})")
        for name, fn in WORD_MORPHS.items():
            t = morph_text(base, fn)
            d, p = verdict_stats(clf, [t])
            hand_rows.append({"base": base, "morph": name, "decision": d[0], "p": p[0], "orig": d0[0]})
            print(f"  {name:28s} {d[0].upper():7s} P={p[0]:.2f}   {mask_text(t)[:60]}")
    pd.DataFrame(hand_rows).to_csv(args.out + "_hand.csv", index=False, encoding="utf-8")

    print("\n=== PART B: real test comments, EVERY word morphed ===")
    df = load_split("telugu", "test", "normalized").sample(frac=1, random_state=1)
    ab = df[df.label == 1].head(args.n * 3)
    ok = df[df.label == 0].head(args.n * 3)
    dec_a, _ = verdict_stats(clf, ab["text"].tolist())
    dec_o, _ = verdict_stats(clf, ok["text"].tolist())
    ab = ab[[d == "block" for d in dec_a]].head(args.n)      # abusive & originally blocked
    ok = ok[[d == "allow" for d in dec_o]].head(args.n)      # clean & originally allowed
    print(f"baseline set: {len(ab)} abusive (blocked) + {len(ok)} clean (allowed)")

    rows = []
    for name, fn in WORD_MORPHS.items():
        da, _ = verdict_stats(clf, [morph_text(t, fn) for t in ab["text"]])
        do, _ = verdict_stats(clf, [morph_text(t, fn) for t in ok["text"]])
        caught = sum(d in ("block", "review") for d in da) / len(da)
        blocked = sum(d == "block" for d in da) / len(da)
        fp = sum(d == "block" for d in do) / len(do)
        rows.append({"morph": name, "abusive_caught_%": round(caught * 100, 1),
                     "abusive_blocked_%": round(blocked * 100, 1),
                     "clean_wrongly_blocked_%": round(fp * 100, 1)})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(args.out + "_data.csv", index=False)
    json.dump({"n_abusive": len(ab), "n_clean": len(ok), "results": rows},
              open(args.out + "_summary.json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
