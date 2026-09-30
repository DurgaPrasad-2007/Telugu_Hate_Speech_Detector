"""
Robustness suite: apply cheap adversarial-style text transforms and measure
how much the model's prediction flips / F1 degrades relative to the
original text on the same validation sample.

Usage:
  python src/robustness.py --model_dir results/<run>/model \
      --base_model_for_adapter <backbone> --domain telugu --n 200
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split  # noqa: E402
from infer import HateSpeechClassifier  # noqa: E402

EMOJIS = ["😂", "🙏", "🔥", "👍", "😡"]


def t_char_repeat(text):
    return re.sub(r"([a-zA-Zఅ-హ])", lambda m: m.group(1) * random.choice([1, 1, 2, 3]), text)


def t_spacing(text):
    return " ".join(list(text.replace(" ", ""))) if len(text) < 40 else re.sub(r"\s+", "  ", text)


def t_emoji_insert(text):
    words = text.split()
    if not words:
        return text
    i = random.randrange(len(words))
    words.insert(i, random.choice(EMOJIS))
    return " ".join(words)


def t_punctuation(text):
    return re.sub(r"(\w)(\s)", lambda m: m.group(1) + random.choice([".", "!", ","]) + m.group(2), text, count=3)


def t_capitalization(text):
    return text.upper() if random.random() < 0.5 else text.lower()


def t_number_subst(text):
    return text.replace("a", "4").replace("e", "3").replace("i", "1").replace("o", "0")

def t_mixed_case_random(text):
    return "".join(c.upper() if random.random() < 0.5 else c for c in text)


TRANSFORMS = {
    "char_repeat": t_char_repeat,
    "extra_spacing": t_spacing,
    "emoji_insert": t_emoji_insert,
    "punctuation_insert": t_punctuation,
    "capitalization": t_capitalization,
    "number_substitution": t_number_subst,
    "random_case": t_mixed_case_random,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--base_model_for_adapter", default=None)
    p.add_argument("--domain", default="telugu")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/robustness_results.csv")
    args = p.parse_args()
    random.seed(args.seed)

    clf = HateSpeechClassifier(args.model_dir, base_model_for_adapter=args.base_model_for_adapter)
    df = load_split(args.domain, "val", preprocess="normalized").sample(
        n=min(args.n, 999999), random_state=args.seed).reset_index(drop=True)

    orig_preds = clf.predict(df["text"].tolist())
    orig_labels = [r["label_id"] for r in orig_preds]

    rows = []
    summary = {}
    for name, fn in TRANSFORMS.items():
        transformed = [fn(t) for t in df["text"].tolist()]
        preds = clf.predict(transformed)
        flips = sum(1 for a, b in zip(orig_labels, [r["label_id"] for r in preds]) if a != b)
        acc_vs_true = sum(1 for r, y in zip(preds, df["label"].tolist()) if r["label_id"] == y) / len(df)
        summary[name] = {"flip_rate": flips / len(df), "accuracy_vs_true_label": acc_vs_true}
        for i in range(len(df)):
            rows.append({
                "transform": name, "original_text_len": len(df["text"].iloc[i]),
                "true_label": int(df["label"].iloc[i]),
                "orig_pred": orig_labels[i], "transformed_pred": preds[i]["label_id"],
                "flipped": orig_labels[i] != preds[i]["label_id"],
            })

    baseline_acc = sum(1 for r, y in zip(orig_preds, df["label"].tolist()) if r["label_id"] == y) / len(df)
    summary["_baseline_accuracy"] = baseline_acc

    import pandas as pd
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    with open(args.out.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
