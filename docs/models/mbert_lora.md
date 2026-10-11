# mbert_lora

| Field | Value |
|---|---|
| Experiment | E6 |
| Status | complete - 3 seed(s) [42, 123, 2024] |
| n seeds | 3 |
| Backbone | `bert-base-multilingual-cased` (bert-base-multilingual-cased (mBERT)) |
| Method | LoRA |
| Config | lr=0.0002, epochs=10, r=8, alpha=64 |

## 2. Goal

LoRA freezes W and learns a low-rank update W + BA on the query and value projections (B in R^{d_out x r}, A in R^{r x d_in}); the effective update is scaled by alpha/r.

Question answered: how much macro-F1 does a LoRA adapter with r=8, alpha=64 on query/value keep relative to full fine-tuning?

## 3. Architecture end-to-end

```text
Telugu text (raw)
  |  normalize_text('normalized'): NFC -> mask URLs & @mentions -> collapse spaces  [src/data_utils.py]
  v
Tokenizer, truncation to max_length=96  [src/train.py]
  |
  v
Embeddings: word + position (119,547 x 768)      [frozen]
  |
  v
Encoder layer x12:
    self-attn query (q):  frozen W + TRAINABLE LoRA A,B  (targets: query,value)
    self-attn key   (k):  frozen
    self-attn value (v):  frozen W + TRAINABLE LoRA A,B
    self-attn out   (o):  frozen
    FFN 768->3072->768:    frozen
    LayerNorm:            frozen
  |
  v
[CLS] -> pooler dense+tanh (pooler: frozen)
  |
  v
head: classifier Linear(768->2)  [TRAINABLE (modules_to_save)]
  |
  v
2 logits -> softmax -> argmax -> Non-abusive (0) / Abusive (1)
```

Adapter formula (q,v with d_in = d_out = hidden = 768):
- LoRA: `layers x targets x r x (d_in + d_out)`
- DoRA: LoRA + `layers x targets x d_out` (magnitude vector)
- AdaLoRA: `layers x targets x init_r x (d_in + d_out + 1)`, init_r = r*3//2 (train.py) or adapter_config.json init_r
- Head (modules_to_save): RoBERTa-type `h*h+h + h*2+2`; BERT-type `h*2+2`; num_labels=2

| Architecture | Value | Source |
|---|---|---|
| layers | 12 | AutoConfig.num_hidden_layers |
| hidden size | 768 | AutoConfig.hidden_size |
| attention heads | 12 | AutoConfig.num_attention_heads |
| FFN intermediate | 3072 | AutoConfig.intermediate_size |
| vocab size | 119,547 | AutoConfig.vocab_size |
| max positions | 512 | AutoConfig.max_position_embeddings |

Parameter counts across seeds identical: OK.

| Parameters | Value | Source |
|---|---|---|
| total params (metrics.json total_params) | 178,151,428 | metrics.json |
| trainable params | 296,450 | metrics.json |
| trainable % | 0.1664 | metrics.json |
| frozen = total - trainable | 177,854,978 | derived |
| adapter params (formula) | 294,912 | formula above |
| head params (formula) | 1,538 | formula above |
| adapter + head vs trainable | OK | formula check |

## 4. Training setup

| Setting | Value | Source |
|---|---|---|
| learning rate | 0.0002 | metrics.json hyperparams.lr |
| epochs (configured) | 10 | CSV / metrics.json |
| train batch size | 16 | metrics.json hyperparams.train_bs |
| eval batch size | 64 | training_args.bin |
| max_length | 96 | metrics.json hyperparams.max_length |
| optimizer | adamw_torch | training_args.bin (HF Trainer default; train.py does not set it) |
| weight decay | 0.01 | training_args.bin (train.py sets 0.01) |
| LR schedule | cosine | training_args.bin / metrics.json |
| warmup steps | 1498 | training_args.bin (train.py: 10% of total steps, see below) |
| fp16 | True | training_args.bin (train.py: fp16 = CUDA available unless --fp16) |
| early stopping patience | 3 | metrics.json hyperparams |
| grad clip (max_grad_norm) | 1.0 | training_args.bin |
| LoRA r / alpha / dropout | 8 / 64 / 0.05 | metrics.json hyperparams |
| target modules | query,value | metrics.json hyperparams.target_modules |
| preprocessing | not recorded in metrics.json | train.py default --preprocess normalized |

Warmup and schedule code: `src/train.py` line ~208 sets warmup_steps = max(1, int(0.1 * total_steps)); line 237 sets weight_decay=0.01; line 239 uses lr_scheduler_type=args.lr_scheduler (default cosine).

## 5. What happened

| seed | best epoch | epochs run / configured | early stop | val macro-F1 | test macro-F1 | train time (s) | peak GPU (MB) |
|---|---|---|---|---|---|---|---|
| 42 | 5 | 8 / 10 | yes | 0.8989 | 0.8962 | 715.3 | 1365.7 |
| 123 | 6 | 9 / 10 | yes | 0.9009 | 0.8931 | 728.3 | 1365.7 |
| 2024 | 9 | 10 / 10 | no | 0.8999 | 0.8945 | 597.7 | 1364.6 |

Early stopping (epochs run < configured) triggered in 2 of 3 seeds. Epochs run is the last epoch with a Trainer eval entry in log_history.json.

![eval macro-F1 and training loss per epoch](figures/mbert_lora_curves.png)

Left: validation macro-F1 per epoch (Trainer eval entries, all seeds). Right: training loss from log_history.json (logged every 50 steps), x-axis in epochs.

## 6. Results

| Metric (test set, mean ± sample std over seeds) | Value |
|---|---|
| accuracy | 0.8947 ± 0.0015 |
| abusive precision | 0.9173 ± 0.0155 |
| abusive recall | 0.8772 ± 0.0148 |
| abusive F1 | 0.8966 ± 0.0009 |
| non-abusive F1 | 0.8926 ± 0.0032 |
| macro-F1 (from CM) | 0.8946 ± 0.0015 |
| weighted-F1 (from CM) | 0.8947 ± 0.0015 |

Consistency: CM-derived macro-F1 vs metrics.json test_macro_f1 per seed: OK, OK, OK.

| seed | test n | TN | FP (pred Abusive, true not) | FN (pred not, true Abusive) | TP | FP/FN |
|---|---|---|---|---|---|---|
| 42 | 2997 | 1343 | 93 | 218 | 1343 | 0.427 |
| 123 | 2997 | 1298 | 138 | 182 | 1379 | 0.758 |
| 2024 | 2997 | 1295 | 141 | 175 | 1386 | 0.806 |

Error counts (mean over seeds): FP = 124.0, FN = 191.7, FP/FN = 0.647. Class 1 = Abusive, confusion matrix layout [[TN,FP],[FN,TP]].

Confusion matrix, seed 123 (highest validation macro-F1 = 0.9009):

![confusion matrix](figures/mbert_lora_cm.png)

## 7. Compared with baseline

No same-backbone full fine-tune is recorded in results/study for bert-base-multilingual-cased (mBERT), so no baseline difference is computed.

## 8. Interpretation

- Test macro-F1 mean 0.8946 over 3 seed(s) (std 0.0015).
- Seed with the highest validation macro-F1: 123 (val 0.9009, test 0.8931).
- Best epochs by seed: 5, 6, 9.
- Hypothesis (not tested): the adapter capacity limits how closely LoRA-type runs track full fine-tuning on this data; the diffs in section 7 are the only evidence here.

## 9. Spec alignment

| parameter | ours | paper Table 2 | professor start | match? |
|---|---|---|---|---|
| LoRA rank r | 8 | 4 | 16 | paper: no; prof: no |
| LoRA alpha | 64 | 32 | 32 | paper: no; prof: no |
| LoRA dropout | 0.05 | 0.1 | 0.1 | paper: no; prof: no |
| train batch size | 16 | 8 | 16 | paper: no; prof: yes |
| max_length | 96 | 128 | 128 | paper: no; prof: no |
| epochs (configured) | 10 | 5 | 5 | paper: no; prof: no |
| learning rate | 0.0002 | not specified | 0.0002 | paper: n/a (not in spec); prof: yes |
| optimizer | adamw_torch | AdamW | not specified | paper: yes; prof: n/a (not in spec) |
| fp16 | True | True | not specified | paper: yes; prof: n/a (not in spec) |
| target modules | query,value | query,value | not specified | paper: yes; prof: n/a (not in spec) |
| weight decay | 0.01 | not specified | 0.01 | paper: n/a (not in spec); prof: yes |

Paper and professor values come from the project spec, not from run artifacts. Paper lr is not specified in the spec.

## 10. Limitations and artifact paths

- Seeds: 3 (42, 123, 2024). Std is over these seeds only; n=1 means no std.
- Single dataset split from train.py / data_utils.py; test size is the sum of the CM (see section 6).
- 'Peak GPU MB' and training time are from one GPU environment; hardware is not recorded in the artifacts.
- Preprocessing level is not recorded in metrics.json; the train.py default is 'normalized'.
- Parameter counts: trainable_params is the count at initialisation for AdaLoRA (final pruned ranks are not recorded).
- Regenerate with `.venv\Scripts\python -I src/model_reports.py`.

Artifacts:

- `results/study/bert-base-multilingual-cased_lora_r8_a64_lr2e-4_e10_s42/` (metrics.json, log_history.json, model/)
- `results/study/bert-base-multilingual-cased_lora_r8_a64_lr2e-4_e10_s123/` (metrics.json, log_history.json, model/)
- `results/study/bert-base-multilingual-cased_lora_r8_a64_lr2e-4_e10_s2024/` (metrics.json, log_history.json, model/)
- `results/study/study_results.csv`
- `docs/models/figures/mbert_lora_curves.png`, `docs/models/figures/mbert_lora_cm.png`

