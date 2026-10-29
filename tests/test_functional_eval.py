import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import functional_eval as fe

df = fe.load_cases()
assert 150 <= len(df) <= 220 and df.text.is_unique
assert (df.confidence == "high").sum() >= 120
assert {"explicit_slur", "threat", "negated_hate", "obfuscation_star", "identity_neutral_caste"} <= set(df.functionality)
assert all(0x0C00 <= ord(c) <= 0x0C7F or not c.isalpha() or c in "‌" for t in df.text for c in t), "non-Telugu letters"
t = pd.DataFrame({"text": list("abcd"), "expected": ["abusive", "abusive", "non_abusive", "non_abusive"],
                  "functionality": ["x", "x", "identity_neutral_g", "identity_neutral_g"],
                  "confidence": ["high", "low", "high", "high"], "p": [0.9, 0.1, 0.7, 0.2],
                  "v": ["BLOCK", "ALLOW", "REVIEW", "ALLOW"]})
t = fe.add_pass(t, "p")
r = fe.pass_rates(t, "pass_p")
assert r.loc["x", "n"] == 1 and r.loc["x", "rate"] == 1.0 and r.loc["identity_neutral_g", "rate"] == 0.5
b = fe.identity_bias(t, "p", "v").loc["identity_neutral_g"]
assert b.fp == 0.5 and b.review == 0.5 and b.n == 2
print("test_functional_eval ok")
