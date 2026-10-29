"""Apply model-authored (Claude) labels to the label-audit review sheet. NOT a native-speaker annotation.
Convention: CSV labels, 0 = abusive, 1 = non-abusive. Rubric: 0 = slur/insult/threat/dehumanizing or explicit
sexual harassment aimed at a person; 1 = compliments, neutral, criticism without abuse. 'borderline' marks
judgement calls (sexualised compliments, mild insults, condemnation of criminals) a native speaker should recheck."""
import sys, pandas as pd
ABUSIVE = {2,3,5,11,12,15,19,21,28,29,30,31,36,37,38,39,40,44,54,65,67,71,75,77,79,84,86,91,98,
102,105,110,117,125,128,136,138,141,148,152,153,157,158,159,160,167,174,183,184,188,199,
203,208,212,222,223,225,226,231,235,238,242,243,244,245,247,249,250,251,255,256,257,271,274,276,288,291,295,297}
BORDER = {0,1,8,9,10,13,14,15,16,22,23,24,25,27,31,32,38,42,43,45,46,48,53,56,57,59,60,61,67,74,78,80,89,93,95,
100,103,104,108,110,111,114,117,123,128,129,131,132,133,135,137,142,143,147,148,150,153,155,157,158,160,165,
167,171,172,175,176,182,184,186,190,193,196,198,199,200,204,205,206,209,212,213,214,216,217,219,221,225,226,
228,232,233,235,236,239,242,243,245,248,249,250,251,255,256,257,259,262,263,265,266,267,274,276,281,285,286,
288,291,296,297,298}
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = pd.read_csv("data/clean/label_audit_gold_review.csv")
    d["human_label"] = [0 if i in ABUSIVE else 1 for i in range(len(d))]
    d["borderline"] = [int(i in BORDER) for i in range(len(d))]
    d["labeler"] = "claude-model (not native speaker)"
    d.to_csv("data/clean/label_audit_gold_review_labeled.csv", index=False, encoding="utf-8")
    flip = d.human_label != d.given_label_csv
    firm = d.borderline == 0
    print("rows", len(d), "| disagree with given label:", flip.sum(), f"({flip.mean():.1%})")
    print("firm (non-borderline) rows:", firm.sum(), "| disagree among firm:", (flip & firm).sum(), f"({(flip & firm).sum()/firm.sum():.1%})")
    print("given 1 -> human 0:", ((d.given_label_csv == 1) & (d.human_label == 0)).sum(),
          "| given 0 -> human 1:", ((d.given_label_csv == 0) & (d.human_label == 1)).sum())
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 300)):
        s = d.iloc[lo:hi]; print(f"ranks {lo}-{hi}: disagree {(s.human_label != s.given_label_csv).mean():.0%}")
