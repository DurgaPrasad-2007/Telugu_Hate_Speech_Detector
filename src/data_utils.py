"""
Data loading and preprocessing utilities for Telugu / Code-mixed
abusive-content detection.

Datasets expected in `data/`:
  telugu_train.csv, telugu_val.csv, telugu_test.csv        (columns: label, text)
  codemixed_train.csv, codemixed_validation.csv, codemixed_test.csv  (columns: text, label)

RAW label convention in these CSVs: 0 = Abusive, 1 = Non-Abusive.
Verified by lineage: telugu_{train,val,test}.csv are byte-identical to
ShareChatAI/MACD `dataset_80_10_10/telugu_*.csv` (NeurIPS 2022), whose
README states "labels 0 and 1 are used for abusive and non-abusive
comments, respectively." This is NOT HOLD-Telugu (DravidianLangTech 2024)
data -- that dataset has ~4-8K string-labelled ("hate"/"non-hate") rows
with entirely different text. See docs/RESEARCH_REVIEW.md for the full
cross-check (manual sample + hash diff + upstream README).

Everything past `_standardize()` in this module uses the REMAPPED
convention instead, matching the rest of the codebase/README: RAW_LABEL_MAP
below flips 0<->1 on load so downstream code, metrics and docs can keep
saying 1 = Abusive. The source CSVs on disk are never modified.
"""
RAW_LABEL_MAP = {0: 1, 1: 0}  # raw(0=abusive,1=non-abusive) -> internal(1=abusive,0=non-abusive)
import os
import re
import unicodedata
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

REQUIRED_COLS = ["text", "label"]

# --- Cleaning -----------------------------------------------------------
# We deliberately do NOT strip stopwords / "unimportant" tokens: abusive
# language detection relies on short function words and particles that a
# generic NLP pipeline would discard.

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")  # aaaa -> aaa (cap runaway repeats, keep some emphasis)


def normalize_text(text: str, level: str = "normalized") -> str:
    """
    level:
      'raw'        - no changes besides str() cast
      'normalized' - unicode NFC normalization, URL/mention masking, whitespace collapse
      'clean'      - normalized + collapse excessive character repetition (noise cleaning)
    """
    text = str(text)
    if level == "raw":
        return text

    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    if level == "clean":
        text = _REPEAT_CHAR_RE.sub(r"\1\1\1", text)

    return text


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}; found {df.columns.tolist()}")
    df = df[REQUIRED_COLS].copy()
    df = df.dropna(subset=REQUIRED_COLS)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int).map(RAW_LABEL_MAP)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return df


def load_split(domain: str, split: str, preprocess: str = "normalized") -> pd.DataFrame:
    """
    domain: 'telugu' | 'codemixed' | 'combined'
    split:  'train' | 'val' | 'test'
    preprocess: 'raw' | 'normalized' | 'clean'
    """
    if domain == "combined":
        te = load_split("telugu", split, preprocess)
        cm = load_split("codemixed", split, preprocess)
        te["domain"] = "telugu"
        cm["domain"] = "codemixed"
        return pd.concat([te, cm], ignore_index=True)

    fname_map = {
        ("telugu", "train"): "telugu_train.csv",
        ("telugu", "val"): "telugu_val.csv",
        ("telugu", "test"): "telugu_test.csv",
        ("codemixed", "train"): "codemixed_train.csv",
        ("codemixed", "val"): "codemixed_validation.csv",
        ("codemixed", "test"): "codemixed_test.csv",
    }
    key = (domain, split)
    if key not in fname_map:
        raise ValueError(f"Unknown domain/split combination: {key}")

    path = os.path.join(DATA_DIR, fname_map[key])
    df = pd.read_csv(path)
    df = _standardize(df)
    df["text"] = df["text"].apply(lambda t: normalize_text(t, preprocess))
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    # Leakage guard: drop val/test rows whose exact text also appears in
    # train (docs/RESEARCH_REVIEW.md sec 4 -- source CSVs are never
    # modified; this filter is applied here on every load instead).
    if split != "train":
        train_path = os.path.join(DATA_DIR, fname_map[(domain, "train")])
        train_df = _standardize(pd.read_csv(train_path))
        train_texts = set(train_df["text"].str.strip())
        before = len(df)
        df = df[~df["text"].str.strip().isin(train_texts)].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"[data_utils] dropped {dropped} train-leaked rows from {domain}/{split}")
    return df


def load_dataset_dict(domain: str = "telugu", preprocess: str = "normalized",
                       text_col: str = "text", label_col: str = "label") -> "DatasetDict":
    """Returns a HF DatasetDict with train/validation/test splits."""
    from datasets import Dataset, DatasetDict  # lazy: inference/serving never needs it
    train_df = load_split(domain, "train", preprocess)
    val_df = load_split(domain, "val", preprocess)
    test_df = load_split(domain, "test", preprocess)

    cols = [text_col if c == "text" else c for c in ["text", "label"]]
    keep = ["text", "label"]

    return DatasetDict({
        "train": Dataset.from_pandas(train_df[keep], preserve_index=False),
        "validation": Dataset.from_pandas(val_df[keep], preserve_index=False),
        "test": Dataset.from_pandas(test_df[keep], preserve_index=False),
    })


def dataset_stats(domain: str = "telugu"):
    stats = {}
    for split in ["train", "val", "test"]:
        df = load_split(domain, split)
        stats[split] = {
            "total": len(df),
            "abusive": int((df["label"] == 1).sum()),
            "non_abusive": int((df["label"] == 0).sum()),
        }
    return stats


if __name__ == "__main__":
    for domain in ["telugu", "codemixed", "combined"]:
        print(f"\n=== {domain} ===")
        for split, s in dataset_stats(domain if domain != "combined" else "telugu").items():
            pass
        try:
            print(dataset_stats(domain))
        except Exception as e:
            print("skipped combined stats:", e)
