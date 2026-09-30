"""Shared evaluation metrics and parameter/efficiency accounting."""
import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1,
        "weighted_f1": f1_w,
    }


def parameter_statistics(model) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params if total_params else 0.0
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": trainable_pct,
    }


class GPUMemoryTracker:
    """Context manager to record peak CUDA memory usage during a block."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            self.peak_memory_mb = 0.0
        self.elapsed_sec = time.time() - self.start


def full_classification_report(y_true, y_pred, target_names=("Non-Abusive", "Abusive")) -> str:
    return classification_report(y_true, y_pred, target_names=list(target_names), digits=4)


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def measure_inference_latency(model, tokenizer, texts, device="cuda" if torch.cuda.is_available() else "cpu",
                               max_length=256, batch_size=32, n_warmup=2):
    """Returns (avg_ms_per_sample, throughput_samples_per_sec)."""
    model.eval()
    model.to(device)

    def _batches(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    with torch.no_grad():
        # warmup
        for _ in range(n_warmup):
            batch = texts[:batch_size] if len(texts) >= batch_size else texts
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
            model(**enc)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        n_processed = 0
        for batch in _batches(texts, batch_size):
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
            model(**enc)
            n_processed += len(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-9)

    avg_ms = (elapsed / n_processed) * 1000
    throughput = n_processed / elapsed
    return avg_ms, throughput
