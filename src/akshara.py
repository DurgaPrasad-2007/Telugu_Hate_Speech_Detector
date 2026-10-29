"""Telugu akshara segmenter + conservative anti-obfuscation normalizer (U+0C00-0C7F).
Narrower than canon.py: single-akshara units only, higher run minimums, star runs
only between lone letters (so 'వె**వ' and ordinary 'a b c' / 'a.b' stay untouched)."""
import re

_C = "క-హౘ-ౚ"
_AKSHARA = re.compile(
    rf"(?:[{_C}]్)*[{_C}][ా-ౌఀ-ఄౕౖ]*్?"  # (C virama)* C matras [virama]
    r"|[అ-ఔౠౡ][ఀ-ఄ]*|[^\W\d_]|\d+|\S", re.U)
_ZW = re.compile("[​‌‍⁠﻿­]")
_U = rf"(?:(?:[{_C}]్)*[{_C}][ా-ౌఀ-ఄౕౖ]*్?|[అ-ఔౠౡ][ఀ-ఄ]*|[A-Za-z])"
_SEP = r"[.\-_*]"
_SEP_RUN = re.compile(rf"(?<![\w]){_U}(?:{_SEP}{_U}){{2,}}(?![\w])")        # d*e*n*g*u, వె.ధ.వ
_SP_TE = re.compile(rf"(?<!\S)(?:{_U} ){{4,}}{_U}(?!\S)")                     # >=5 lone units
_SP_LA = re.compile(r"(?<!\S)(?:[A-Za-z] ){3,}[A-Za-z](?!\S)")               # >=4 lone letters


def segment(text):
    """Orthographic syllables (aksharas); non-Telugu chars come out one by one."""
    return _AKSHARA.findall(text)


def normalize(text):
    """Strip zero-width chars; rejoin separator/space-split single-letter runs."""
    t = _ZW.sub("", text)
    t = _SEP_RUN.sub(lambda m: re.sub(_SEP, "", m.group()), t)
    return _SP_LA.sub(lambda m: m.group().replace(" ", ""),
                      _SP_TE.sub(lambda m: m.group().replace(" ", ""), t))


def ngrams(text, n_max=3):
    """Akshara n-grams per whitespace word (analyzer for TfidfVectorizer)."""
    out = []
    for w in text.split():
        s = segment(w)
        out += ["".join(s[i:i + n]) for n in range(1, n_max + 1) for i in range(len(s) - n + 1)]
    return out
