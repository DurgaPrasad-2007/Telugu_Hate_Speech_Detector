"""
Display-only masking for abusive text shown during manual review, error
analysis exports, and console/log output. Never applied to model input —
only to what a human reads while testing/inspecting results, per Rule 9
(don't expose raw abusive content unnecessarily) while keeping the text
readable enough to judge if the model/label is right.
"""
import re

_WORD_RE = re.compile(r"\S+")


def mask_word(tok: str) -> str:
    if len(tok) <= 2:
        return "*" * len(tok)
    return tok[0] + "*" * (len(tok) - 2) + tok[-1]


def mask_text(text: str) -> str:
    """Keep first/last char of each token, star out the middle. Shape and
    length stay recognizable; the exact slur isn't rendered."""
    if not isinstance(text, str):
        return text
    return _WORD_RE.sub(lambda m: mask_word(m.group()), text)


def mask_if_abusive(text: str, label: int) -> str:
    """label uses the internal convention: 1 = Abusive."""
    return mask_text(text) if label == 1 else text


if __name__ == "__main__":
    assert mask_word("పూకు") == "ప**ు"
    assert mask_word("ok") == "**"
    assert mask_text("idi manchidi") == "i*i m******i"
    print("text_mask self-check OK")
