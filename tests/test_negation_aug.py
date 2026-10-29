import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import negation_aug as n

lex = [(l.strip(), 1, 0) for l in open(n.LEX, encoding="utf-8")]
slurs = n.pick_slurs(lex)
assert len(slurs) == n.N_SLURS and "వెధవ" in slurs and not set(slurs) & n.DENY
rows = n.generate(slurs)
assert len({t for t, _ in rows}) == len(rows)
neg, pos = [t for t, y in rows if y == 0], [t for t, y in rows if y == 1]
assert len(neg) == len(pos), (len(neg), len(pos))  # contrast, not 'negation => safe'

# holdout disjointness: no shared text, no shared word 4-gram after masking the slurs
hold = pd.read_csv(n.HATECHECK, encoding="utf-8").text.tolist()
mask = lambda t: " ".join("X" if w in slurs or w in {"వెధవ", "లంజ", "సన్నాసి"} else w for w in re.sub(r"[,.]", "", t).split())
grams = lambda t: {tuple(w[i:i + 4]) for w in [mask(t).split()] for i in range(len(w) - 3)}
hg = set().union(*(grams(t) for t in hold))
assert not {t for t, _ in rows} & set(hold)
shared = [t for t, _ in rows if grams(t) & hg]
assert not shared, shared[:3]

# built files: val/test identical to clean, raw labels (0=abusive), aug rows present, no val/test leakage
if os.path.isdir(n.OUT):
    for f in ("telugu_val.csv", "telugu_test.csv"):
        assert open(os.path.join(n.OUT, f), "rb").read() == open(os.path.join(n.SRC, f), "rb").read()
    a = pd.read_csv(os.path.join(n.OUT, "telugu_train.csv"))
    d = dict(zip(a.text, a.label))
    assert all(d[t] == (0 if y == 1 else 1) for t, y in rows)
print("negation_aug tests OK:", len(rows), "rows,", len(slurs), "slurs")
