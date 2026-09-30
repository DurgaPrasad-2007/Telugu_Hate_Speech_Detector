"""
Confidence calibration: ECE, Brier score, reliability diagram, and
temperature scaling fit on VALIDATION data only (never test — Rule 3).

Usage:
  python src/calibration.py --model_dir results/<run>/model \
      --base_model_for_adapter <backbone> --domain telugu
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_split  # noqa: E402
from infer import HateSpeechClassifier  # noqa: E402


def expected_calibration_error(probs, labels, n_bins=10):
    """probs: P(abusive) per example. labels: 0/1 ground truth."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # confidence of the predicted class, not raw P(abusive)
        conf = np.where(preds == 1, probs, 1 - probs)
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "acc": None, "conf": None})
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(acc - avg_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                      "acc": float(acc), "conf": float(avg_conf)})
    return float(ece), bins


def brier_score(probs, labels):
    probs, labels = np.asarray(probs), np.asarray(labels)
    return float(np.mean((probs - labels) ** 2))


def fit_temperature(logits, labels, lr=0.01, iters=200):
    """Guo et al. 2017 temperature scaling: single scalar T minimizing NLL
    on validation logits. Applied as logits/T before softmax at serve time."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)  # optimize log(T) for positivity
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=iters)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        T = torch.exp(log_t)
        loss = nll(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--base_model_for_adapter", default=None)
    p.add_argument("--domain", default="telugu")
    p.add_argument("--out", default="results/calibration.json")
    args = p.parse_args()

    clf = HateSpeechClassifier(args.model_dir, base_model_for_adapter=args.base_model_for_adapter)
    val_df = load_split(args.domain, "val", preprocess="normalized")

    # raw logits needed for temperature fit; predict() only returns probs,
    # so recompute logits directly here.
    all_logits, labels = [], []
    with torch.no_grad():
        for i in range(0, len(val_df), 32):
            batch = val_df["text"].iloc[i:i + 32].tolist()
            enc = clf.tokenizer(batch, truncation=True, padding=True,
                                 max_length=clf.max_length, return_tensors="pt").to(clf.device)
            logits = clf.model(**enc).logits.cpu().numpy()
            all_logits.append(logits)
        labels = val_df["label"].tolist()
    logits = np.concatenate(all_logits, axis=0)
    probs_abusive_before = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    ece_before, bins_before = expected_calibration_error(probs_abusive_before, labels)
    brier_before = brier_score(probs_abusive_before, labels)

    T = fit_temperature(logits, labels)
    probs_abusive_after = torch.softmax(torch.tensor(logits) / T, dim=-1)[:, 1].numpy()
    ece_after, bins_after = expected_calibration_error(probs_abusive_after, labels)
    brier_after = brier_score(probs_abusive_after, labels)

    report = {
        "domain": args.domain, "n_val": len(val_df), "temperature": T,
        "before": {"ece": ece_before, "brier": brier_before, "reliability_bins": bins_before},
        "after": {"ece": ece_after, "brier": brier_after, "reliability_bins": bins_after},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"T={T:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}  Brier {brier_before:.4f} -> {brier_after:.4f}")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
