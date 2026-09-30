"""Dataset forensics: distribution, script mix, duplicates, leakage. Real
numbers only, computed from data/ via data_utils (post label-remap)."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split  # noqa: E402

TELUGU_RE = re.compile(r"[ఀ-౿]")
LATIN_RE = re.compile(r"[A-Za-z]")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
URL_RE = re.compile(r"https?://|www\.")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def script_mix(text):
    has_te, has_en = bool(TELUGU_RE.search(text)), bool(LATIN_RE.search(text))
    if has_te and has_en:
        return "telugu+english"
    if has_te:
        return "telugu_script"
    if has_en:
        return "latin_only"
    return "other"


def audit_split(domain, split):
    df = load_split(domain, split, preprocess="raw")
    lengths = df["text"].str.len()
    tokens = df["text"].str.split().str.len()
    scripts = df["text"].apply(script_mix).value_counts().to_dict()
    return {
        "n": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "char_len_mean": round(float(lengths.mean()), 1),
        "char_len_p95": int(lengths.quantile(0.95)),
        "token_len_mean": round(float(tokens.mean()), 1),
        "empty_after_strip": int((df["text"].str.strip() == "").sum()),
        "exact_dup_texts": int(df["text"].duplicated().sum()),
        "script_mix": scripts,
        "emoji_rows": int(df["text"].apply(lambda t: bool(EMOJI_RE.search(t))).sum()),
        "url_rows": int(df["text"].apply(lambda t: bool(URL_RE.search(t))).sum()),
    }


def leakage(domain):
    splits = {s: set(load_split(domain, s, preprocess="raw")["text"]) for s in ["train", "val", "test"]}
    return {
        "train_val_overlap": len(splits["train"] & splits["val"]),
        "train_test_overlap": len(splits["train"] & splits["test"]),
        "val_test_overlap": len(splits["val"] & splits["test"]),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {}
    for domain in ["telugu", "codemixed"]:
        report[domain] = {s: audit_split(domain, s) for s in ["train", "val", "test"]}
        report[domain]["leakage"] = leakage(domain)

    with open(os.path.join(OUT_DIR, "data_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = ["# Data Quality Report\n", "Computed on internal label convention (1=Abusive) after RAW_LABEL_MAP.\n"]
    for domain, d in report.items():
        lines.append(f"\n## {domain}\n")
        lines.append("| split | n | abusive | non_abusive | char_len_mean | dup_texts | emoji | url |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in ["train", "val", "test"]:
            v = d[s]
            lines.append(f"| {s} | {v['n']} | {v['label_counts'].get(1,0)} | {v['label_counts'].get(0,0)} | "
                         f"{v['char_len_mean']} | {v['exact_dup_texts']} | {v['emoji_rows']} | {v['url_rows']} |")
        lines.append(f"\nScript mix (train): {d['train']['script_mix']}\n")
        lines.append(f"Leakage (exact text overlap): {d['leakage']}\n")
    with open(os.path.join(OUT_DIR, "data_quality_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps(report, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
