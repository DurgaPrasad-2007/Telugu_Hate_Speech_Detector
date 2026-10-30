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
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    set_seed,
)


PEFT_METHODS = {"lora", "qlora", "adalora", "dora", "ia3"}


class AdaLoraAllocCallback(TrainerCallback):
    """AdaLoRA only reallocates rank when update_and_allocate(step) is called once per optimizer step.
    The orthogonal regularisation is already added inside AdaLoraModel.forward (do not duplicate it)."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "base_model") and hasattr(model.base_model, "update_and_allocate"):
            model.base_model.update_and_allocate(state.global_step)


def _targets(t):
    # peft needs the plain string 'all-linear' (a one-element list is treated as a module name)
    return t if t == "all-linear" else t.split(",")


class MultiKernelCNNHead(torch.nn.Module):
    """Parallel 1D convs (k=3,5,7) over all encoder states -> global max-pool -> linear.
    Drop-in for RobertaClassificationHead (which receives the full sequence output).
    ponytail: no attention-mask in pooling (pad states can win the max); add mask if padding-heavy."""

    def __init__(self, hidden, num_labels, channels=128, kernels=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv1d(hidden, channels, k, padding=k // 2) for k in kernels)
        self.drop = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(channels * len(kernels), num_labels)

    def forward(self, features, **kwargs):
        x = features.transpose(1, 2)
        pooled = torch.cat([torch.relu(c(x)).max(dim=2).values for c in self.convs], dim=1)
        return self.out(self.drop(pooled))


def build_peft_config(method: str, args, total_steps: int):
    from peft import LoraConfig, AdaLoraConfig, IA3Config, TaskType

    if method in {"lora", "qlora", "dora"}:
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=_targets(args.target_modules),
            bias="none",
            use_dora=(method == "dora"),
        )
    elif method == "adalora":
        return AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=_targets(args.target_modules),
            init_r=args.lora_r * 3 // 2,
            target_r=args.lora_r,
            tinit=int(0.1 * total_steps),
            tfinal=int(0.2 * total_steps),
            deltaT=10,
            beta1=0.85,
            beta2=0.85,
            orth_reg_weight=0.5,
            total_step=total_steps,
        )
    elif method == "ia3":
        # IA3 scales q/v and the feedforward output projection.
        # feedforward_modules must be a subset of target_modules (PEFT requirement).
        feedforward_modules = [m for m in args.ia3_feedforward_modules.split(",") if m]
        if args.target_modules == "all-linear":
            raise SystemExit("--target-modules all-linear is not supported with ia3")
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
    p.add_argument("--target_modules", "--target-modules", dest="target_modules", default=None,
                    help="Comma-separated module names, or 'all-linear'; if unset, resolved from configs/targets.json by --model (docs/RESEARCH_REVIEW.md sec 2)")
    p.add_argument("--ia3_feedforward_modules", default="dense")
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--early_stop_patience", type=int, default=0, help="0 = off")
    p.add_argument("--output_root", default="results")
    p.add_argument("--run_name", default=None)
    p.add_argument("--loraplus-ratio", dest="loraplus_ratio", type=float, default=None,
                    help="Opt-in LoRA+ (Hayou 2024): lr of B = ratio * lr of A (paper uses 16). lora/qlora/dora only")
    p.add_argument("--cnn-head", dest="cnn_head", action="store_true",
                    help="Opt-in multi-kernel (3,5,7) 1D-CNN pooling head; RoBERTa/XLM-R-style classifier only")
    p.add_argument("--fp16", action="store_true", default=None)
    p.add_argument("--grad_ckpt", action="store_true", help="Opt-in gradient checkpointing (decoder LLMs / QLoRA)")
    p.add_argument("--dump_preds", action="store_true",
                    help="Opt-in: write preds.csv (split,text,label,p_abusive) for val+test; label 1=abusive (internal convention)")
    args = p.parse_args()

    set_seed(args.seed)

    if args.target_modules is None:
        targets_path = os.path.join(os.path.dirname(__file__), "..", "configs", "targets.json")
        with open(targets_path, encoding="utf-8") as f:
            targets_cfg = json.load(f)
        if args.model in targets_cfg:
            args.target_modules = ",".join(targets_cfg[args.model])
        elif "bert" in args.model.lower():
            print(f"[train] WARNING: {args.model} not in configs/targets.json, falling back to 'query,value' -- verify via named_modules() before trusting these results")
            args.target_modules = "query,value"
        else:
            raise SystemExit(f"{args.model} not in configs/targets.json and not BERT-style; pass --target_modules explicitly")

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

    # transformers 5.17.0 dropped TrainingArguments.warmup_ratio (verified via
    # inspect.signature) -- compute warmup_steps (10% of total steps) instead.
    steps_per_epoch = max(1, len(ds_tok["train"]) // (args.train_bs * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    # ---- Model ----
    model = load_model(args.method, args.model)
    if tokenizer.pad_token is None:  # decoder LLMs (Llama-style) have no pad token; seq-cls pools at last non-pad token
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.loraplus_ratio is not None and args.method not in {"lora", "qlora", "dora"}:
        raise SystemExit("--loraplus-ratio only applies to lora/qlora/dora")
    if args.cnn_head:
        if type(model.classifier).__name__ != "RobertaClassificationHead":
            raise SystemExit("--cnn-head supports RoBERTa/XLM-R classifiers only")
        model.classifier = MultiKernelCNNHead(model.config.hidden_size, model.config.num_labels)

    if args.method in PEFT_METHODS:
        from peft import get_peft_model
        peft_config = build_peft_config(args.method, args, total_steps)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    param_stats = parameter_statistics(model)
    print("Parameter stats:", param_stats)

    use_fp16 = torch.cuda.is_available() if args.fp16 is None else args.fp16
    # QLoRA already loads in 4-bit; fp16 flag on TrainingArguments is unnecessary/conflicting there.
    if args.method == "qlora":
        use_fp16 = False

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
        lr_scheduler_type=args.lr_scheduler,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=use_fp16,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
        seed=args.seed,
    )

    callbacks = []
    if args.method == "adalora":
        callbacks.append(AdaLoraAllocCallback())
    if args.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    optimizers = (None, None)
    if args.loraplus_ratio is not None:
        from peft.optimizers import create_loraplus_optimizer
        opt = create_loraplus_optimizer(model=model, optimizer_cls=torch.optim.AdamW, lr=args.lr,
                                        loraplus_lr_ratio=args.loraplus_ratio, weight_decay=0.01)
        optimizers = (opt, None)  # Trainer builds the scheduler around the given optimizer

    trainer = Trainer(
        model=model,
        optimizers=optimizers,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    with GPUMemoryTracker() as mem:
        trainer.train()
    training_time_sec = mem.elapsed_sec
    peak_memory_mb = mem.peak_memory_mb

    with open(os.path.join(out_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    eval_logs = [e for e in trainer.state.log_history if "eval_macro_f1" in e]
    best_epoch = next((e.get("epoch") for e in eval_logs
                       if e["eval_macro_f1"] == trainer.state.best_metric), None)

    # ---- Test evaluation ----
    test_results = trainer.evaluate(eval_test_tok)
    pred_output = trainer.predict(eval_test_tok)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids

    if args.dump_preds:  # text-keyed so rows align by content, not position (dedup/leak filters change n)
        import pandas as pd
        from scipy.special import softmax
        ev_texts = (load_dataset_dict(domain=args.eval_domain, preprocess=args.preprocess)["test"]["text"]
                    if args.eval_domain and args.eval_domain != args.domain else ds["test"]["text"])
        vo = trainer.predict(ds_tok["validation"])
        pd.concat([
            pd.DataFrame({"split": "val", "text": ds["validation"]["text"], "label": vo.label_ids,
                          "p_abusive": softmax(vo.predictions, axis=1)[:, 1]}),
            pd.DataFrame({"split": "test", "text": ev_texts, "label": labels,
                          "p_abusive": softmax(pred_output.predictions, axis=1)[:, 1]}),
        ]).to_csv(os.path.join(out_dir, "preds.csv"), index=False)

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
            "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules, "lr_scheduler": args.lr_scheduler,
            "early_stop_patience": args.early_stop_patience,
            "use_dora": args.method == "dora",
            "loraplus_ratio": args.loraplus_ratio, "cnn_head": args.cnn_head, "best_epoch": best_epoch,
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
