"""
Lightweight second model: TF-IDF + LinearSVC, behind the same input guard,
decision rule and result format as HateSpeechClassifier (src/infer.py).

LinearSVC has no probabilities, so a Platt scaler (1-D logistic regression on
decision_function) is fitted on VALIDATION only. Thresholds low/high are fitted
on validation with finalize.threshold_sweep, using the rule of src/set_policy.py.
Both are stored in models/tfidf/calibration.json.

Run: .venv\\Scripts\\python -I src\\tfidf_classifier.py   (fit + report)
"""
import json
import logging
import os
import shutil
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from canon import canonicalize, display_tokens  # noqa: E402
from data_utils import load_split, normalize_text  # noqa: E402
from infer import LABELS, decide, sanitize_text  # noqa: E402  (same decision rule and labels)
from text_mask import mask_text  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_MODEL = os.path.join(ROOT, "models", "tfidf", "tfidf_best.joblib")
DEFAULT_CALIB = os.path.join(ROOT, "models", "tfidf", "calibration.json")
STUDY_MODEL = os.path.join(ROOT, "results", "study", "tfidf", "tfidf_best.joblib")
MIN_BLOCK_PRECISION = 0.9

log = logging.getLogger("abuse")


def guard(text):
    """Same input guard as HateSpeechClassifier.predict: canonicalize, then normalize."""
    clean, fired = canonicalize(text)
    return normalize_text(clean, "normalized"), fired


class TfidfGuardClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL, calib_path: str = DEFAULT_CALIB):
        self.pipe = joblib.load(model_path)
        with open(calib_path, encoding="utf-8") as f:
            c = json.load(f)
        self.platt_a = float(c["platt"]["coef"])
        self.platt_b = float(c["platt"]["intercept"])
        self.thresholds = {"low": float(c["thresholds"]["low"]), "high": float(c["thresholds"]["high"])}
        self.device = "cpu"

    def _p_abusive(self, guarded_texts):
        d = self.pipe.decision_function(guarded_texts)
        return 1.0 / (1.0 + np.exp(-(self.platt_a * d + self.platt_b)))

    def explain(self, text: str, max_words: int = 80, min_drop: float = 0.03):
        """Occlusion attribution, identical to HateSpeechClassifier.explain.
        Returns [(word, drop, flagged)]."""
        words = display_tokens(text)
        if not words or len(words) > max_words:
            log.info("[explain] skipped (%d words; limit %d)", len(words), max_words)
            return [(w, 0.0, False) for w in words]
        base = self.predict([text], _quiet=True)[0]
        if base.get("decision") == "allow":
            log.info("[explain] verdict is allow -> nothing to hide")
            return [(w, 0.0, False) for w in words]
        variants = [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))]
        drops = [base["probability"] - r["probability"] for r in self.predict(variants, _quiet=True)]
        top = max(drops)
        out = [(w, d, d >= min_drop and d >= 0.35 * top) for w, d in zip(words, drops)]
        log.info("[explain] occlusion (tfidf): %d variants; flag if drop >= %.2f",
                 len(variants), max(min_drop, 0.35 * top))
        for w, d, f in out:
            log.info("[explain]   %-22s drop=%+.3f %s", mask_text(w), d, "<- HIDDEN" if f else "")
        return out

    def predict(self, texts, batch_size: int = 32, preprocess: str = "normalized",
                return_probs: bool = True, _quiet: bool = False):
        """Same contract as HateSpeechClassifier.predict: bad items get an 'error' entry."""
        if isinstance(texts, str) or texts is None:
            texts = [texts]
        results = [None] * len(texts)
        valid_idx, valid = [], []
        for i, t in enumerate(texts):
            clean = sanitize_text(t)
            if clean.strip() == "":
                results[i] = {"text": t, "error": "empty_or_invalid_input",
                              "label": None, "decision": "allow", "review_required": False}
                continue
            norm, fired = guard(clean)
            if fired and not _quiet:
                log.info("[input %d] anti-obfuscation fixed: %s", i, ", ".join(fired))
            valid_idx.append(i)
            valid.append(norm)

        for start in range(0, len(valid), batch_size):
            ps = self._p_abusive(valid[start:start + batch_size])
            for j, p in enumerate(ps):
                orig_i = valid_idx[start + j]
                p_ab = float(p)
                pred = int(p_ab > 0.5)  # argmax of the two calibrated probabilities
                decision = decide(p_ab, self.thresholds)
                entry = {"text": texts[orig_i], "label": LABELS[pred], "label_id": pred,
                         "probability": p_ab, "decision": decision,
                         "review_required": decision == "review"}
                if return_probs:
                    entry["confidence"] = p_ab if pred == 1 else 1.0 - p_ab
                    entry["probs"] = {LABELS[0]: 1.0 - p_ab, LABELS[1]: p_ab}
                results[orig_i] = entry
        return results


def block_stats(probs, labels, thr):
    """Block precision/recall and review share for the given thresholds (validation or test)."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    blk = probs >= thr["high"]
    rev = ~blk & (probs >= thr["low"])
    tp = int((blk & (labels == 1)).sum())
    fp = int((blk & (labels == 0)).sum())
    fn = int((~blk & (labels == 1)).sum())
    return {"block_precision": tp / (tp + fp) if tp + fp else 0.0,
            "block_recall": tp / (tp + fn) if tp + fn else 0.0,
            "review_pct": float(rev.mean())}


def fit_and_save(model_path: str = DEFAULT_MODEL, calib_path: str = DEFAULT_CALIB):
    from finalize import threshold_sweep  # imports infer (torch) only at fit time

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.copyfile(STUDY_MODEL, model_path)
        log.info("copied %s -> %s", STUDY_MODEL, model_path)
    pipe = joblib.load(model_path)

    va = load_split("telugu", "val", "raw")  # raw text, then the same guard as inference
    y = va["label"].values
    d = pipe.decision_function([guard(t)[0] for t in va["text"]]).reshape(-1, 1)
    platt = LogisticRegression().fit(d, y)
    probs = platt.predict_proba(d)[:, 1]

    sweep = threshold_sweep(probs, y)
    ok = sweep[sweep["block_precision"] >= MIN_BLOCK_PRECISION]
    best = (ok if len(ok) else sweep.sort_values("block_precision", ascending=False)).sort_values(
        ["block_recall", "review_pct"], ascending=[False, True]).iloc[0]
    thr = {"low": float(best["low"]), "high": float(best["high"])}

    calib = {
        "model": os.path.relpath(model_path, ROOT),
        "fit_on": "telugu validation split only (raw text, same guard as inference)",
        "n_val": int(len(va)),
        "platt": {"coef": float(platt.coef_[0, 0]), "intercept": float(platt.intercept_[0])},
        "thresholds": thr,
        "min_block_precision": MIN_BLOCK_PRECISION,
        "validation": block_stats(probs, y, thr),
    }
    os.makedirs(os.path.dirname(calib_path), exist_ok=True)
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    return calib


def _evaluate(clf, split):
    df = load_split("telugu", split, "raw")
    res = clf.predict(df["text"].tolist(), _quiet=True)
    y = df["label"].values
    pred = np.array([r["label_id"] for r in res])
    probs = np.array([r["probability"] for r in res])
    m = block_stats(probs, y, clf.thresholds)
    m["macro_f1_argmax"] = float(f1_score(y, pred, average="macro"))
    m["n"] = len(df)
    return m


if __name__ == "__main__":
    calib = fit_and_save()
    clf = TfidfGuardClassifier()
    val, test = _evaluate(clf, "val"), _evaluate(clf, "test")
    print(f"thresholds: low={clf.thresholds['low']:.3f} high={clf.thresholds['high']:.3f}")
    print(f"platt: a={calib['platt']['coef']:.4f} b={calib['platt']['intercept']:.4f}")
    for name, m in [("VAL", val), ("TEST", test)]:
        print(f"{name} (n={m['n']}): guarded macro-F1 (argmax@0.5)={m['macro_f1_argmax']:.4f} | "
              f"block precision={m['block_precision']:.4f} recall={m['block_recall']:.4f} "
              f"review%={100 * m['review_pct']:.2f}")
