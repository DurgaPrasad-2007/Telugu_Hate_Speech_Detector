"""
Fast inference wrapper for the trained abusive-content classifier.

Designed for short, informal, code-mixed text as seen in Google/Instagram
comments — not for the full "any-format multimodal hate speech" problem
(see README for that scope caveat).

Usage:
  python src/infer.py --model_dir results/peft__qlora__telugu/model --text "..."

Or import:
  from infer import HateSpeechClassifier
  clf = HateSpeechClassifier("results/peft__qlora__telugu/model")
  clf.predict(["some text", "another text"])
"""
import argparse
import json
import sys
import os
import time
import logging
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import normalize_text
from text_mask import mask_text
from canon import canonicalize, display_tokens

LABELS = {0: "Non-Abusive", 1: "Abusive"}

log = logging.getLogger("abuse")
_EMOJI = re.compile("[🀀-🫿☀-➿⬀-⯿️]+")


def setup_logging(level=None):
    """Live pipeline trace on stdout. INFO logs show masked text only; set
    LOG_RAW=1 (DEBUG) to also print raw input. Safe to call repeatedly."""
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    if not log.handlers:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(logging.DEBUG if os.environ.get("LOG_RAW") == "1" else level)
    return log

# Moderation decision thresholds on P(abusive). Placeholders until
# calibration/threshold-sweep (docs/RESEARCH_REVIEW.md phase 5) picks real
# values from validation data; override via configs/policy.json if present.
_POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "policy.json")
DEFAULT_THRESHOLDS = {"low": 0.35, "high": 0.65}


def load_thresholds():
    if os.path.exists(_POLICY_PATH):
        with open(_POLICY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_THRESHOLDS


_CALIB_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "calibration.json")


def load_temperature() -> float:
    """Temperature fit on validation only (src/calibration.py); 1.0 if not run."""
    if os.path.exists(_CALIB_PATH):
        with open(_CALIB_PATH, encoding="utf-8") as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0


def decide(p_abusive: float, thresholds: dict) -> str:
    if p_abusive >= thresholds["high"]:
        return "block"
    if p_abusive >= thresholds["low"]:
        return "review"
    return "allow"


def sanitize_text(t) -> str:
    """None/non-str/whitespace-only/etc all become '' rather than crashing
    the batch; caller still gets a structured result for that item."""
    if t is None:
        return ""
    if not isinstance(t, str):
        t = str(t)
    return t


class HateSpeechClassifier:
    def __init__(self, model_dir: str, device: str = None, max_length: int = 256,
                 base_model_for_adapter: str = None, defend: bool = True):
        """
        model_dir: path to a saved full model, OR a saved PEFT adapter directory.
        base_model_for_adapter: required only if model_dir holds a PEFT adapter
            rather than a merged full model (adapter_config.json present).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.defend = defend  # anti-obfuscation canonicalization (src/canon.py)

        is_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
        is_onnx = any(f.endswith(".onnx") for f in os.listdir(model_dir)) if os.path.isdir(model_dir) else False

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.is_onnx = is_onnx

        if is_onnx:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            onnx_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
            onnx_file = "model_quantized.onnx" if "model_quantized.onnx" in onnx_files else onnx_files[0]
            self.model = ORTModelForSequenceClassification.from_pretrained(model_dir, file_name=onnx_file)
        elif is_adapter:
            from peft import PeftModel, PeftConfig
            if base_model_for_adapter is None:
                base_model_for_adapter = PeftConfig.from_pretrained(model_dir).base_model_name_or_path
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_for_adapter, num_labels=2)
            self.model = PeftModel.from_pretrained(base_model, model_dir)
            # merge adapter weights into the base model for fastest inference
            self.model = self.model.merge_and_unload()
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()

        self.thresholds = load_thresholds()
        self.temperature = load_temperature()

    def explain(self, text: str, max_words: int = 80, min_drop: float = 0.03):
        """Occlusion attribution: drop each whitespace word in turn and see how
        far P(abusive) falls. Returns [(word, drop, flagged)]; flagged words are
        the ones carrying most of the signal, and none are flagged when the
        text isn't blocked/reviewed."""
        words = display_tokens(text)
        if not words or len(words) > max_words:
            log.info("[explain] skipped (%d words; limit %d)", len(words), max_words)
            return [(w, 0.0, False) for w in words]
        base = self.predict([text], _quiet=True)[0]
        if base.get("decision") == "allow":
            log.info("[explain] verdict is allow -> nothing to hide")
            return [(w, 0.0, False) for w in words]
        variants = [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))]
        t0 = time.time()
        drops = [base["probability"] - r["probability"]
                 for r in self.predict(variants, _quiet=True)]
        top = max(drops)
        out = [(w, d, d >= min_drop and d >= 0.35 * top) for w, d in zip(words, drops)]
        log.info("[explain] occlusion: %d variants scored in %.0f ms; flag if drop >= %.2f",
                 len(variants), (time.time() - t0) * 1000, max(min_drop, 0.35 * top))
        for w, d, f in out:
            log.info("[explain]   %-22s drop=%+.3f %s", mask_text(w), d, "<- HIDDEN" if f else "")
        return out

    @torch.no_grad()
    def predict(self, texts, batch_size: int = 32, preprocess: str = "normalized",
                return_probs: bool = True, _quiet: bool = False):
        """Never raises on a bad individual item (None, non-str, empty,
        whitespace-only, punctuation/emoji-only, very long) -- that item
        gets an 'error' entry and the rest of the batch still runs."""
        if isinstance(texts, str) or texts is None:
            texts = [texts]

        verbose = not _quiet and len(texts) <= 5
        t_start = time.time()
        if not _quiet:
            log.info("[predict] batch of %d text(s), device=%s, T=%.3f, policy=allow<%.2f<=review<%.2f<=block",
                     len(texts), self.device, self.temperature,
                     self.thresholds["low"], self.thresholds["high"])
        results = [None] * len(texts)
        valid_idx, valid_texts = [], []
        for i, t in enumerate(texts):
            clean = sanitize_text(t)
            if clean.strip() == "":
                results[i] = {"text": t, "error": "empty_or_invalid_input",
                              "label": None, "decision": "allow", "review_required": False}
                if verbose:
                    log.info("[input %d] empty/invalid -> rejected, decision=allow", i)
            else:
                if self.defend:
                    clean_c, fired = canonicalize(clean)
                    if fired and not _quiet:
                        log.info("[input %d] anti-obfuscation fixed: %s", i, ", ".join(fired))
                        log.info("[input %d]   before: %s", i, mask_text(clean)[:80])
                        log.info("[input %d]   after : %s", i, mask_text(clean_c)[:80])
                        log.debug("[input %d]   RAW after: %s", i, clean_c)
                    clean = clean_c
                norm = normalize_text(clean, preprocess)
                valid_idx.append(i)
                valid_texts.append(norm)
                if verbose:
                    log.info("[input %d] %s | chars=%d words=%d", i, mask_text(clean),
                             len(clean), len(clean.split()))
                    log.debug("[input %d] RAW: %s", i, clean)
                    if norm != clean:
                        log.info("[input %d] normalization changed the text (NFC/URL/space cleanup)", i)

        for start in range(0, len(valid_texts), batch_size):
            idxs = valid_idx[start:start + batch_size]
            batch = valid_texts[start:start + batch_size]
            enc = self.tokenizer(batch, truncation=True, padding=True,
                                  max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits / self.temperature, dim=-1)
            # second view without emoji: emoji-flooded text can drown out the words,
            # so an abusive score on either view counts.
            for j, orig_i in enumerate(idxs):
                stripped = _EMOJI.sub("", batch[j]).strip()
                if stripped and stripped != batch[j]:
                    e2 = self.tokenizer([stripped], truncation=True, max_length=self.max_length,
                                        return_tensors="pt").to(self.device)
                    p2 = float(torch.softmax(self.model(**e2).logits / self.temperature, dim=-1)[0, 1])
                    if verbose:
                        log.info("[input %d] emoji-stripped 2nd view: P(abusive)=%.3f (using max of views)",
                                 orig_i, p2)
                    if p2 > float(probs[j, 1]):
                        probs[j, 1], probs[j, 0] = p2, 1 - p2
            preds = torch.argmax(probs, dim=-1)
            if verbose:
                for j, orig_i in enumerate(idxs):
                    n_tok = int(enc["attention_mask"][j].sum())
                    raw_p = float(torch.softmax(logits[j], dim=-1)[1])
                    log.info("[input %d] tokenized -> %d subword tokens (max %d) | logits=[%.2f, %.2f] "
                             "| P(abusive) raw=%.3f -> calibrated=%.3f",
                             orig_i, n_tok, self.max_length, float(logits[j, 0]), float(logits[j, 1]),
                             raw_p, float(probs[j, 1]))

            for j, orig_i in enumerate(idxs):
                p_abusive = float(probs[j, 1])
                decision = decide(p_abusive, self.thresholds)
                entry = {
                    "text": texts[orig_i],
                    "label": LABELS[int(preds[j])],
                    "label_id": int(preds[j]),
                    "probability": p_abusive,
                    "decision": decision,
                    "review_required": decision == "review",
                }
                if return_probs:
                    entry["confidence"] = float(probs[j, preds[j]])
                    entry["probs"] = {LABELS[k]: float(probs[j, k]) for k in range(2)}
                results[orig_i] = entry
                if verbose:
                    log.info("[input %d] => %s | P=%.1f%% | decision=%s", orig_i,
                             entry["label"], p_abusive * 100, decision.upper())
        if not _quiet:
            log.info("[predict] done in %.0f ms", (time.time() - t_start) * 1000)
        return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text", nargs="+", required=True)
    p.add_argument("--base_model_for_adapter", default=None)
    args = p.parse_args()

    clf = HateSpeechClassifier(args.model_dir, base_model_for_adapter=args.base_model_for_adapter)

    start = time.time()
    results = clf.predict(args.text)
    elapsed = time.time() - start

    for r in results:
        print(f"[{r['label']}] ({r['confidence']:.3f}) {r['text']}")
    print(f"\n{len(args.text)} texts in {elapsed*1000:.1f} ms "
          f"({elapsed*1000/len(args.text):.2f} ms/text)")


if __name__ == "__main__":
    main()
