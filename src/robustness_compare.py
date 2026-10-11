"""Robustness of the selected model vs the TF-IDF baseline under the same text perturbations.

Same 600 test comments, same perturbations (sentence-level from robustness.py, per-word from
stress_test.py). Metric: accuracy against the TRUE label (classes are ~balanced) and the share of
predictions that flip relative to the unperturbed text.
Writes results/study/robustness_compare.{json,md}.
"""
import json, os, random, sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import robustness as RB            # noqa: E402
import stress_test as ST           # noqa: E402
from data_utils import load_split  # noqa: E402
from infer import HateSpeechClassifier  # noqa: E402

N = 600
df = load_split("telugu", "test", "normalized").sample(N, random_state=0).reset_index(drop=True)
texts, y = df["text"].tolist(), df["label"].to_numpy()

tfidf = joblib.load(os.path.join(ROOT, "results/study/tfidf/tfidf_best.joblib"))


def p_tfidf(xs):
    return np.asarray(tfidf.predict(xs)).astype(int)


from canon import canonicalize  # noqa: E402


def p_tfidf_guard(xs):
    return np.asarray(tfidf.predict([canonicalize(t)[0] for t in xs])).astype(int)


def mk(model_dir, defend):
    clf = HateSpeechClassifier(model_dir, defend=defend)

    def f(xs):
        out = []
        for i in range(0, len(xs), 64):
            out += [int(r["probs"]["Abusive"] >= 0.5) for r in clf.predict(xs[i:i + 64])]
        return np.array(out)
    return f


MODELS = {
    "TF-IDF + LinearSVC": p_tfidf,
    "XLM-R LoRA (selected), raw": mk(os.path.join(ROOT, "models/best"), False),
    "XLM-R LoRA (selected), with input guard": mk(os.path.join(ROOT, "models/best"), True),
    "XLM-R full FT, raw": mk(os.path.join(ROOT, "results/study/xlm-roberta-base_full_lr2e-5_e5_s2024/model"), False),
    "TF-IDF + input guard": p_tfidf_guard,
    "XLM-R full FT, with input guard": mk(os.path.join(ROOT, "results/study/xlm-roberta-base_full_lr2e-5_e5_s2024/model"), True),
}

ATTACKS = {}
for name, fn in RB.TRANSFORMS.items():
    ATTACKS["sentence: " + name] = (lambda xs, fn=fn: [fn(t) for t in xs])
for name, fn in ST.WORD_MORPHS.items():
    ATTACKS["every word: " + name] = (lambda xs, fn=fn: [ST.morph_text(t, fn) for t in xs])

res = {m: {} for m in MODELS}
clean_pred = {}
for m, pf in MODELS.items():
    clean_pred[m] = pf(texts)
    res[m]["clean"] = {"acc": float((clean_pred[m] == y).mean()), "flip": 0.0}
for a, af in ATTACKS.items():
    random.seed(0)
    ST.R.seed(0)
    xs = af(texts)
    for m, pf in MODELS.items():
        p = pf(xs)
        res[m][a] = {"acc": float((p == y).mean()), "flip": float((p != clean_pred[m]).mean())}
        print(f"{a:38s} {m:42s} acc={res[m][a]['acc']:.3f} flip={res[m][a]['flip']:.3f}", flush=True)

json.dump({"n": N, "results": res}, open(os.path.join(ROOT, "results/study/robustness_compare.json"), "w"), indent=1)
names = list(MODELS)
lines = ["| Perturbation | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for a in ["clean"] + list(ATTACKS):
    lines.append(f"| {a} | " + " | ".join(
        f"{res[m][a]['acc']:.3f}" + ("" if a == "clean" else f" ({res[m][a]['flip']:.0%} flip)") for m in names) + " |")
mean = {m: np.mean([res[m][a]["acc"] for a in ATTACKS]) for m in names}
lines.append("| **mean over attacks** | " + " | ".join(f"{mean[m]:.3f}" for m in names) + " |")
open(os.path.join(ROOT, "results/study/robustness_compare.md"), "w", encoding="utf-8").write(
    f"Accuracy on {N} test comments (true labels); flip = predictions changed vs the clean text.\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
