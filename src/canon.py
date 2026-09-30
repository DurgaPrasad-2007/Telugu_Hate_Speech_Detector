"""
Anti-obfuscation canonicalization, applied before the model sees the text.
Deliberately conservative: only patterns that are attacks, not ordinary
Telugu (doubled letters like నన్ను, emoji after words, "word.word" all stay).

Each rule returns (text, changed) so the live log can show what it fixed.
"""
import re

_EMO = "\U0001F000-\U0001FAFF☀-➿⬀-⯿️"
_U = r"[ఀ-౿A-Za-z]{1,2}"            # one letter / letter+mark
_SEP = rf"[.\-_·•|~^{_EMO}]{{1,2}}"  # separator an attacker hides in

_ZW = re.compile("[​‌‍⁠﻿]")
_SEP_RUN = re.compile(rf"{_U}(?:{_SEP}{_U}){{2,}}")           # v.e.d.h.a / వె😂ధ😂వ
_SPACE_RUN = re.compile(rf"(?<!\S){_U}(?: {_U}){{3,}}(?!\S)")  # v e d h a
_TE_REPEAT = re.compile(r"([ఀ-౿])\1{2,}")           # వవవవ -> వ
_LATIN_REPEAT = re.compile(r"([A-Za-z])\1{2,}")               # chaaaala -> chaala

RULES = [
    ("zero-width characters", lambda t: _ZW.sub("", t)),
    ("separators inside a word", lambda t: _SEP_RUN.sub(lambda m: re.sub(_SEP, "", m.group()), t)),
    ("spaced-out letters", lambda t: _SPACE_RUN.sub(lambda m: m.group().replace(" ", ""), t)),
    ("stretched Telugu letters", lambda t: _TE_REPEAT.sub(r"\1", t)),
    ("stretched Latin letters", lambda t: _LATIN_REPEAT.sub(r"\1\1", t)),
]


def display_tokens(text: str):
    """Whitespace tokens, except spaced-out letter runs stay one token so the
    UI can hide 'v e d h a' as a single word."""
    out, pos = [], 0
    for m in _SPACE_RUN.finditer(text):
        out += text[pos:m.start()].split()
        out.append(m.group())
        pos = m.end()
    return out + text[pos:].split()


def canonicalize(text: str):
    """Returns (clean_text, [names of rules that changed something])."""
    fired = []
    for name, fn in RULES:
        new = fn(text)
        if new != text:
            fired.append(name)
            text = new
    return text, fired


if __name__ == "__main__":
    ok = lambda s: canonicalize(s)[0]
    assert ok("వె‍ధ‍వ") == "వెధవ"
    assert ok("వె😂ధ😂వ") == "వెధవ"
    assert ok("వెధవవవవవ") == "వెధవ"
    assert ok("v e d h a v a") == "vedhava"
    assert ok("v.e.d.h.a") == "vedha"
    assert ok("chaaaaala") == "chaaala".replace("aaa", "aa")
    # ordinary text must be untouched
    for s in ["నన్ను చచ్చి", "బాగుంది😂నువ్వు", "బాగుంది.నువ్వు", "good video 👍", "a b c", "వాళ్లలో ఇంకా"]:
        assert canonicalize(s)[0] == s, s
    print("canon self-check OK")
