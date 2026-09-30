"""
Unified training script for Telugu / code-mixed abusive-content detection.

Supports:
  - Full fine-tuning of any AutoModelForSequenceClassification-compatible checkpoint
  - PEFT methods: LoRA, QLoRA, AdaLoRA, DoRA, IA3
  - Any domain: telugu | codemixed | combined
  - Config-driven so every experiment in the paper's comparison table is one CLI call

Usage examples:
  python src/train.py --model xlm-roberta-base --method full --domain telugu
  python src/train.py --model xlm-roberta-base --method lora --domain telugu --lora_r 16
  python src/train.py --model xlm-roberta-base --method qlora --domain telugu
  python src/train.py --model xlm-roberta-base --method adalora --domain telugu
  python src/train.py --model xlm-roberta-base --method dora --domain telugu
  python src/train.py --model xlm-roberta-base --method ia3 --domain telugu

Each run writes:
  results/<run_name>/metrics.json         (test metrics, params, timing, memory)
  results/<run_name>/confusion_matrix.png
  results/<run_name>/classification_report.txt
  results/<run_name>/model/                (saved adapter or full model)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_dataset_dict
from metrics_utils import (
    compute_metrics,
    parameter_statistics,
    GPUMemoryTracker,
    full_classification_report,
    get_confusion_matrix,
    measure_inference_latency,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    set_seed,
)


PEFT_METHODS = {"lora", "qlora", "adalora", "dora", "ia3"}


def build_peft_config(method: str, args):
    from peft import LoraConfig, AdaLoraConfig, IA3Config, TaskType

    if method in {"lora", "qlora", "dora"}:
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(","),
            bias="none",
            use_dora=(method == "dora"),
        )
    elif method == "adalora":
        return AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(","),
            init_r=args.lora_r,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            deltaT=args.adalora_deltaT,
            beta1=0.85,
            beta2=0.85,
            orth_reg_weight=0.5,
            total_step=args.adalora_total_step,
        )
    elif method == "ia3":
        # IA3 scales q/v and the feedforward output projection.
        # feedforward_modules must be a subset of target_modules (PEFT requirement).
        feedforward_modules = [m for m in args.ia3_feedforward_modules.split(",") if m]
        target_modules = list(dict.fromkeys(args.target_modules.split(",") + feedforward_modules))
        return IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
        )
    else:
        raise ValueError(f"Unknown PEFT method: {method}")


def load_model(method: str, model_name: str, num_labels: int = 2):
    if method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels,
            quantization_config=bnb_config, device_map="auto",
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF checkpoint, e.g. xlm-roberta-base")
    p.add_argument("--method", required=True,
                    choices=["full", "lora", "qlora", "adalora", "dora", "ia3"])
    p.add_argument("--domain", default="telugu", choices=["telugu", "codemixed", "combined"])
    p.add_argument("--eval_domain", default=None,
                    help="If set, evaluate on this domain's test set instead of --domain (for cross-domain transfer experiments)")
    p.add_argument("--preprocess", default="normalized", choices=["raw", "normalized", "clean"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=None, help="Defaults: 2e-5 full-FT, 1e-4 for PEFT")
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default=None,
                    help="Comma-separated module names; if unset, resolved from configs/targets.json by --model (docs/RESEARCH_REVIEW.md sec 2)")
    p.add_argument("--ia3_feedforward_modules", default="dense")
    p.add_argument("--adalora_tinit", type=int, default=100)
    p.add_argument("--adalora_tfinal", type=int, default=500)
    p.add_argument("--adalora_deltaT", type=int, default=10)
    p.add_argument("--adalora_total_step", type=int, default=1000)
    p.add_argument("--output_root", default="results")
    p.add_argument("--run_name", default=None)
    p.add_argument("--fp16", action="store_true", default=None)
    args = p.parse_args()

    set_seed(args.seed)

    if args.target_modules is None:
        import json
        targets_path = os.path.join(os.path.dirname(__file__), "..", "configs", "targets.json")
        with open(targets_path, encoding="utf-8") as f:
            targets_cfg = json.load(f)
        if args.model in targets_cfg:
            args.target_modules = ",".join(targets_cfg[args.model])
        else:
            print(f"[train] WARNING: {args.model} not in configs/targets.json, falling back to 'query,value' -- verify via named_modules() before trusting these results")
            args.target_modules = "query,value"

    if args.lr is None:
        args.lr = 2e-5 if args.method == "full" else 1e-4

    run_name = args.run_name or f"{args.model.replace('/', '_')}__{args.method}__{args.domain}__seed{args.seed}"
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Run: {run_name} ===")
    print(f"Model: {args.model} | Method: {args.method} | Domain: {args.domain} | Seed: {args.seed}")

    # ---- Data ----
    ds = load_dataset_dict(domain=args.domain, preprocess=args.preprocess)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=args.max_length)

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])

    eval_test_tok = ds_tok["test"]
    if args.eval_domain and args.eval_domain != args.domain:
        eval_ds = load_dataset_dict(domain=args.eval_domain, preprocess=args.preprocess)
        eval_test_tok = eval_ds["test"].map(tok_fn, batched=True, remove_columns=["text"])

    # ---- Model ----
    model = load_model(args.method, args.model)

    if args.method in PEFT_METHODS:
        from peft import get_peft_model
        peft_config = build_peft_config(args.method, args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    param_stats = parameter_statistics(model)
    print("Parameter stats:", param_stats)

    use_fp16 = torch.cuda.is_available() if args.fp16 is None else args.fp16
    # QLoRA already loads in 4-bit; fp16 flag on TrainingArguments is unnecessary/conflicting there.
    if args.method == "qlora":
        use_fp16 = False

    # transformers 5.17.0 dropped TrainingArguments.warmup_ratio (verified via
    # inspect.signature) -- compute warmup_steps (10% of total steps) instead.
    steps_per_epoch = max(1, len(ds_tok["train"]) // (args.train_bs * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=use_fp16,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    with GPUMemoryTracker() as mem:
        trainer.train()
    training_time_sec = mem.elapsed_sec
    peak_memory_mb = mem.peak_memory_mb

    # ---- Test evaluation ----
    test_results = trainer.evaluate(eval_test_tok)
    pred_output = trainer.predict(eval_test_tok)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids

    report_str = full_classification_report(labels, preds)
    cm = get_confusion_matrix(labels, preds)

    # ---- Inference latency (on a sample of test texts) ----
    raw_test_texts = load_dataset_dict(domain=args.eval_domain or args.domain,
                                        preprocess=args.preprocess)["test"]["text"]
    sample_texts = raw_test_texts[: min(200, len(raw_test_texts))]
    try:
        avg_ms, throughput = measure_inference_latency(model, tokenizer, sample_texts, max_length=args.max_length)
    except Exception as e:
        print("Latency measurement skipped:", e)
        avg_ms, throughput = None, None

    # ---- Save artifacts ----
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Abusive", "Abusive"],
                yticklabels=["Non-Abusive", "Abusive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{run_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)

    summary = {
        "run_name": run_name,
        "model": args.model,
        "method": args.method,
        "domain": args.domain,
        "eval_domain": args.eval_domain or args.domain,
        "seed": args.seed,
        "test_accuracy": test_results.get("eval_accuracy"),
        "test_precision": test_results.get("eval_precision"),
        "test_recall": test_results.get("eval_recall"),
        "test_macro_f1": test_results.get("eval_macro_f1"),
        "test_weighted_f1": test_results.get("eval_weighted_f1"),
        # Rule 3: model/config selection must use validation, not test.
        # trainer.state.best_metric is the validation macro-F1 of the
        # checkpoint actually loaded (load_best_model_at_end=True).
        "val_macro_f1": trainer.state.best_metric,
        "total_params": param_stats["total_params"],
        "trainable_params": param_stats["trainable_params"],
        "trainable_pct": param_stats["trainable_pct"],
        "training_time_sec": training_time_sec,
        "peak_gpu_memory_mb": peak_memory_mb,
        "avg_inference_latency_ms": avg_ms,
        "throughput_samples_per_sec": throughput,
        "hyperparams": {
            "lr": args.lr, "epochs": args.epochs, "train_bs": args.train_bs,
            "max_length": args.max_length, "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha, "target_modules": args.target_modules,
        },
        "confusion_matrix": cm.tolist(),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save model/adapter
    model_dir = os.path.join(out_dir, "model")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
