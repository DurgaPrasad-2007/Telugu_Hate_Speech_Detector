# muril_lora

| Field | Value |
|---|---|
| Experiment | extra |
| Status | complete - 3 seed(s) [42, 123, 2024] |
| n seeds | 3 |
| Backbone | `google/muril-base-cased` (MuRIL base cased) |
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
Embeddings: word + position (197,285 x 768)      [frozen]
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
| vocab size | 197,285 | AutoConfig.vocab_size |
| max positions | 512 | AutoConfig.max_position_embeddings |

Parameter counts across seeds identical: OK.

| Parameters | Value | Source |
|---|---|---|
| total params (metrics.json total_params) | 237,854,212 | metrics.json |
| trainable params | 296,450 | metrics.json |
| trainable % | 0.1246 | metrics.json |
| frozen = total - trainable | 237,557,762 | derived |
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
| 42 | 8 | 10 / 10 | no | 0.9048 | 0.9078 | 486.8 | 1592.9 |
| 123 | 8 | 10 / 10 | no | 0.9019 | 0.9058 | 891.4 | 1591.7 |
| 2024 | 7 | 10 / 10 | no | 0.9010 | 0.9018 | 777.3 | 1591.7 |

Early stopping (epochs run < configured) triggered in 0 of 3 seeds. Epochs run is the last epoch with a Trainer eval entry in log_history.json.

![eval macro-F1 and training loss per epoch](figures/muril_lora_curves.png)

Left: validation macro-F1 per epoch (Trainer eval entries, all seeds). Right: training loss from log_history.json (logged every 50 steps), x-axis in epochs.

## 6. Results

| Metric (test set, mean ± sample std over seeds) | Value |
|---|---|
| accuracy | 0.9052 ± 0.0031 |
| abusive precision | 0.9185 ± 0.0012 |
| abusive recall | 0.8977 ± 0.0078 |
| abusive F1 | 0.9080 ± 0.0034 |
| non-abusive F1 | 0.9023 ± 0.0026 |
| macro-F1 (from CM) | 0.9052 ± 0.0030 |
| weighted-F1 (from CM) | 0.9053 ± 0.0030 |

Consistency: CM-derived macro-F1 vs metrics.json test_macro_f1 per seed: OK, OK, OK.

| seed | test n | TN | FP (pred Abusive, true not) | FN (pred not, true Abusive) | TP | FP/FN |
|---|---|---|---|---|---|---|
| 42 | 2997 | 1309 | 127 | 149 | 1412 | 0.852 |
| 123 | 2997 | 1311 | 125 | 157 | 1404 | 0.796 |
| 2024 | 2997 | 1315 | 121 | 173 | 1388 | 0.699 |

Error counts (mean over seeds): FP = 124.3, FN = 159.7, FP/FN = 0.779. Class 1 = Abusive, confusion matrix layout [[TN,FP],[FN,TP]].

Confusion matrix, seed 42 (highest validation macro-F1 = 0.9048):

![confusion matrix](figures/muril_lora_cm.png)

## 7. Compared with baseline

| metric | LoRA (this) | full FT (same backbone) | diff (this - full) | |diff| vs max(std) |
|---|---|---|---|---|
| test macro-F1 | 0.9052 ± 0.0030 | 0.9049 ± 0.0028 | +0.0002 | within seed noise |
| val macro-F1 | 0.9026 ± 0.0020 | 0.9035 ± 0.0023 | -0.0009 | within seed noise |

| ratio | value |
|---|---|
| trainable params: this / full | 0.0012 (296,450 / 237,557,762) |
| peak GPU memory: this / full | 0.347 (1592 MB / 4586 MB) |
| training time: this / full (mean) | 0.575 (718 s / 1250 s) |

Std-based noise rule: 'within seed noise' if |mean diff| < max(std_this, std_full). Sample std (n-1).

## 8. Interpretation

- Test macro-F1 mean 0.9052 over 3 seed(s) (std 0.0030).
- Seed with the highest validation macro-F1: 42 (val 0.9048, test 0.9078).
- Best epochs by seed: 8, 8, 7.
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

- `results/study/lora_r8_a64_lr2e-4_e10_s42/` (metrics.json, log_history.json, model/)
- `results/study/lora_r8_a64_lr2e-4_e10_s123/` (metrics.json, log_history.json, model/)
- `results/study/lora_r8_a64_lr2e-4_e10_s2024/` (metrics.json, log_history.json, model/)
- `results/study/study_results.csv`
- `docs/models/figures/muril_lora_curves.png`, `docs/models/figures/muril_lora_cm.png`

