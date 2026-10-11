"""
Exploratory lexical analysis of the Telugu TRAIN split (abusive vs non-abusive).

Exploratory frequency evidence only. This is NOT a model explanation.

Outputs (repo-relative):
  docs/figures/lexical_top_terms.png   raw top-25 frequency per class (two panels)
  docs/figures/lexical_log_odds.png    top-25 log-odds per class (two panels)
  docs/LEXICAL_ANALYSIS.md             tables + short commentary

Run:  .venv\\Scripts\\python -I src\\lexical_analysis.py
"""
import math
import os
import re
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(ROOT, "docs", "figures")
DOC_PATH = os.path.join(ROOT, "docs", "LEXICAL_ANALYSIS.md")

TOP_K = 25
MIN_COUNT = 20
FONT_NAME = "Nirmala UI"  # Windows Telugu font; fallbacks tried in order below
FONT_FALLBACKS = ["Gautami", "Vani"]

# Placeholder tokens present in the corpus (e.g. "<number>" x1411) are removed
# before tokenising so they do not show up as terms.
_PLACEHOLDER_RE = re.compile(r"<\s*[A-Za-z_]+\s*>")
# Telugu letters/marks/virama: U+0C00-U+0C65 (excludes Telugu digits U+0C66-0C6F
# and Telugu numeric/fraction signs U+0C70-0C7F).
_TOKEN_CHARS_RE = re.compile(r"[^ఀ-౥A-Za-z\s]")
_JOINER_RE = re.compile(r"[‌‍]")  # ZWNJ/ZWJ: delete, do not split words
_LATIN_DIGIT_RE = re.compile(r"[0-9]")


def tokenize(text: str) -> list[str]:
    """Strip placeholders, emoji, punctuation, digits and URL leftovers; keep Telugu + Latin."""
    text = _PLACEHOLDER_RE.sub(" ", text)
    text = _JOINER_RE.sub("", text)
    text = _TOKEN_CHARS_RE.sub(" ", text)
    text = _LATIN_DIGIT_RE.sub(" ", text)
    text = text.lower()  # no-op for Telugu; lowercases Latin only
    return [t for t in text.split() if t]


def _self_check():
    assert tokenize("ఇది <number> 👌 test, 123 ok!") == ["ఇది", "test", "ok"]
    assert tokenize("a‌b c") == ["ab", "c"]


def pick_font():
    for name in [FONT_NAME] + FONT_FALLBACKS:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            if path:
                return name, path
        except ValueError:
            continue
    raise RuntimeError("No Telugu font found")


def log_odds_table(c_abu, c_non, min_count):
    """Add-1 smoothed log-odds ratio of P(term|abusive) vs P(term|non-abusive)."""
    vocab = set(c_abu) | set(c_non)
    v = len(vocab)
    n_abu = sum(c_abu.values())
    n_non = sum(c_non.values())
    rows = []
    for term in vocab:
        a, b = c_abu.get(term, 0), c_non.get(term, 0)
        if a + b < min_count:
            continue
        lo = math.log((a + 1) / (n_abu + v)) - math.log((b + 1) / (n_non + v))
        rows.append((term, a, b, lo))
    return rows


def bar_panels(title_prefix, left, right, path, xlabel, font_name, left_label, right_label, fmt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    for ax, (items, label, color) in zip(axes, [(left, left_label, "#b03a2e"), (right, right_label, "#2e6da4")]):
        terms = [t for t, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        ax.barh(range(len(terms)), vals, color=color)
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms, fontfamily=font_name, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(xlabel)
        for i, v in enumerate(vals):
            ax.text(v, i, " " + fmt(v), va="center", fontsize=8)
        ax.margins(x=0.15)
    fig.suptitle(title_prefix, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    _self_check()
    os.makedirs(FIG_DIR, exist_ok=True)

    font_name, font_path = pick_font()
    plt.rcParams["font.family"] = font_name
    print(f"font: {font_name} -> {font_path}")

    df = load_split("telugu", "train", "normalized")  # label: 1 = Abusive, 0 = Non-abusive
    c_abu, c_non = Counter(), Counter()
    for text, label in zip(df["text"], df["label"]):
        (c_abu if label == 1 else c_non).update(tokenize(text))
    n_abu_docs = int((df["label"] == 1).sum())
    n_non_docs = int((df["label"] == 0).sum())
    print(f"train docs: abusive={n_abu_docs} non-abusive={n_non_docs}")
    print(f"tokens: abusive={sum(c_abu.values())} non-abusive={sum(c_non.values())}")

    # View 1: raw top-K frequency per class
    top_abu = c_abu.most_common(TOP_K)
    top_non = c_non.most_common(TOP_K)
    bar_panels(
        "Telugu train split: top-25 terms by raw frequency (exploratory)",
        top_abu, top_non,
        os.path.join(FIG_DIR, "lexical_top_terms.png"),
        "count", font_name,
        f"Abusive (label 1), n={n_abu_docs} docs",
        f"Non-abusive (label 0), n={n_non_docs} docs",
        lambda v: f"{int(v)}",
    )

    # View 2: log-odds, top-K each way
    rows = log_odds_table(c_abu, c_non, MIN_COUNT)
    rows_sorted = sorted(rows, key=lambda r: r[3])
    toward_abu = sorted(rows, key=lambda r: -r[3])[:TOP_K]
    toward_non = rows_sorted[:TOP_K]  # most negative = most non-abusive
    bar_panels(
        f"Telugu train split: top-25 log-odds terms (add-1 smoothed, min count {MIN_COUNT})",
        [(t, lo) for t, a, b, lo in toward_abu],
        [(t, -lo) for t, a, b, lo in toward_non],
        os.path.join(FIG_DIR, "lexical_log_odds.png"),
        "log-odds ratio (natural log)", font_name,
        "Over-represented in abusive",
        "Over-represented in non-abusive (sign flipped)",
        lambda v: f"{v:.2f}",
    )

    # Markdown
    lines = []
    lines.append("# Lexical Analysis: Telugu Train Split\n")
    lines.append(
        "Exploratory word-frequency analysis, reproducing the lexical view of the "
        "Indonesian hate-speech paper for Telugu. It is exploratory lexical evidence, "
        "not a model explanation. Source: `load_split('telugu', 'train', 'normalized')` only. "
        "Labels: 1 = Abusive, 0 = Non-abusive.\n"
    )
    lines.append("## Method\n")
    lines.append(
        f"- Documents: {n_abu_docs} abusive, {n_non_docs} non-abusive (after dedup in `data_utils`).\n"
        f"- Tokens: {sum(c_abu.values()):,} abusive, {sum(c_non.values()):,} non-abusive.\n"
        "- Tokenisation: `<number>` placeholders removed; emoji, punctuation, digits and "
        "ZWJ/ZWNJ joiners removed; whitespace split; Telugu script (U+0C00-U+0C65) and Latin kept; "
        "Latin lowercased.\n"
        "- No stopword removal, no stemming.\n"
        f"- Log-odds: per term `ln((a+1)/(N_abu+V)) - ln((b+1)/(N_non+V))`, where a and b are the "
        f"term's counts per class, V is the vocabulary size, and terms need total count >= {MIN_COUNT}.\n"
        f"- Font for Telugu glyphs in figures: {font_name} ({font_path}).\n"
        "- Word clouds were not produced because the `wordcloud` package is not installed; "
        "horizontal bar charts are used instead.\n"
    )
    lines.append("## Figures\n")
    lines.append("![Top-25 raw frequency](figures/lexical_top_terms.png)\n")
    lines.append("![Top-25 log-odds](figures/lexical_log_odds.png)\n")

    lines.append("## Top 25 terms by raw frequency\n")
    lines.append("### Abusive (label 1)\n")
    lines.append(md_table(
        ["rank", "term", "count (abusive)", "count (non-abusive)"],
        [(i + 1, t, c_abu[t], c_non.get(t, 0)) for i, (t, _) in enumerate(top_abu)],
    ) + "\n")
    lines.append("### Non-abusive (label 0)\n")
    lines.append(md_table(
        ["rank", "term", "count (non-abusive)", "count (abusive)"],
        [(i + 1, t, c_non[t], c_abu.get(t, 0)) for i, (t, _) in enumerate(top_non)],
    ) + "\n")

    lines.append("## Top 25 terms by log-odds\n")
    lines.append(
        f"Restricted to terms with total count >= {MIN_COUNT}. Log-odds is signed: positive = "
        "over-represented in abusive, negative = over-represented in non-abusive. The figure shows "
        "the non-abusive side as absolute values.\n"
    )
    lines.append("### Over-represented in abusive\n")
    lines.append(md_table(
        ["rank", "term", "log-odds", "count (abusive)", "count (non-abusive)"],
        [(i + 1, t, f"{lo:.3f}", a, b) for i, (t, a, b, lo) in enumerate(toward_abu)],
    ) + "\n")
    lines.append("### Over-represented in non-abusive\n")
    lines.append(md_table(
        ["rank", "term", "log-odds", "count (abusive)", "count (non-abusive)"],
        [(i + 1, t, f"{lo:.3f}", a, b) for i, (t, a, b, lo) in enumerate(toward_non)],
    ) + "\n")

    lines.append("## Observations\n")
    lines.append(
        "Raw frequency is led by a small set of high-frequency tokens shared across both classes, "
        "including possessive pronouns, the particle meaning 'also', and a generic noun meaning 'problems', "
        "so the raw view separates the classes only weakly. The log-odds view surfaces more contrastive "
        "terms: the abusive side is enriched for profane terms and for terms tied to astrology, spells, "
        "marriage and family matters, and court cases, while the non-abusive side is enriched mostly for "
        "greetings and well-wishes (for example 'good morning', 'best wishes' and 'thank you'). Because Telugu is agglutinative and no stemming was applied, inflected forms of "
        "one stem are counted separately, so these rankings understate the true size of each theme. "
        "This is exploratory frequency evidence only: it describes which surface strings co-occur with each "
        "label in the training split and does not show how the model uses them.\n"
    )
    lines.append("## Caveats\n")
    lines.append(
        "- Agglutination: surface forms dominate; no stemming or morphological analysis.\n"
        "- Emoji and joiners were removed, so emoji-only signal is not represented.\n"
        "- Spelling variants are separate terms (for example two spellings of one profane term both "
        "appear in the abusive log-odds list), so per-term counts are split across variants.\n"
        "- Log-odds with add-1 smoothing is unstable for rare terms; the min-count threshold limits this.\n"
        "- Train split only; val and test were not inspected.\n"
    )

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
