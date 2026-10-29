"""Rule-based Romanized-Telugu -> Telugu script / phonetic key. No deps."""
import re

_CONS = {"ksh": "క్ష", "chh": "ఛ", "kh": "ఖ", "gh": "ఘ", "ch": "చ", "jh": "ఝ", "th": "థ", "dh": "ధ",
         "ph": "ఫ", "bh": "భ", "sh": "శ", "k": "క", "g": "గ", "j": "జ", "t": "త", "d": "ద", "n": "న",
         "p": "ప", "b": "బ", "m": "మ", "y": "య", "r": "ర", "l": "ల", "v": "వ", "s": "స", "h": "హ",
         "f": "ఫ", "z": "జ", "c": "క", "q": "క", "x": "క్స"}
_DENT = {"th": "త", "dh": "ద"}  # romanized th/dh usually dental in Telugu chat
_VOW = {"aa": ("ఆ", "ా"), "ee": ("ఈ", "ీ"), "ii": ("ఈ", "ీ"), "oo": ("ఊ", "ూ"), "uu": ("ఊ", "ూ"),
        "ai": ("ఐ", "ై"), "au": ("ఔ", "ౌ"), "a": ("అ", ""), "i": ("ఇ", "ి"), "u": ("ఉ", "ు"),
        "e": ("ఎ", "ె"), "o": ("ఒ", "ొ")}
_CK = sorted(_CONS, key=len, reverse=True)
_VK = sorted(_VOW, key=len, reverse=True)


def _pre(w):
    w = re.sub(r"[^a-z]", "", w.lower().replace("w", "v"))
    return re.sub(r"o", "a", w)  # vaddu/voddu/woddu -> vaddu (o~a spelling noise)


def to_telugu(word):
    w, out, i, prev_c = _pre(word), "", 0, False
    while i < len(w):
        for k in _VK:
            if w.startswith(k, i) and (prev_c or True):
                ind, mat = _VOW[k]
                if prev_c and k != "a":
                    out += mat
                elif not prev_c:
                    out += ind
                prev_c, i = False, i + len(k)
                break
        else:
            for k in _CK:
                if w.startswith(k, i):
                    if prev_c:
                        out += "్"
                    out += _DENT.get(k, _CONS[k])
                    prev_c, i = True, i + len(k)
                    break
            else:
                i += 1
    return out + ("్" if False else "")


def phonetic_key(word):
    """Spelling-variant-insensitive key: vaddu/voddu/woddu -> 'vadu'."""
    w = _pre(word)
    w = re.sub(r"([bcdfghjklmnpqrstvxyz])h", r"\1", w)   # aspirates fold
    w = re.sub(r"[ei]+", "i", w)
    w = re.sub(r"a+", "a", w)
    w = re.sub(r"u+", "u", w)
    return re.sub(r"(.)\1+", r"\1", w)


def translit_text(text):
    """Replace each pure-Latin word with its Telugu-script form."""
    return re.sub(r"[A-Za-z]+", lambda m: to_telugu(m.group()) or m.group(), text)
