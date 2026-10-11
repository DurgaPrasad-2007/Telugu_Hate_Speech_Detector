# xlmr_dora

| Field | Value |
|---|---|
| Experiment | E5 |
| Status | complete - 3 seed(s) [42, 123, 2024] |
| n seeds | 3 |
| Backbone | `FacebookAI/xlm-roberta-base` (xlm-roberta-base) |
| Method | DoRA |
| Config | lr=0.0002, epochs=10, r=8, alpha=64 |

## 2. Goal

DoRA decomposes each adapted weight into magnitude and direction, W' = m * (W + BA) / ||W + BA||, training a magnitude vector m plus the LoRA direction BA (use_dora=True in PEFT).

Question answered: does DoRA (r=8, alpha=64) differ from plain LoRA and full fine-tuning?

## 3. Architecture end-to-end

```text
Telugu text (raw)
  |  normalize_text('normalized'): NFC -> mask URLs & @mentions -> collapse spaces  [src/data_utils.py]
  v
Tokenizer, truncation to max_length=96  [src/train.py]
  |
  v
Embeddings: word + position (250,002 x 768)      [frozen]
  |
  v
Encoder layer x12:
    self-attn query (q):  frozen W + TRAINABLE LoRA A,B + magnitude m  (targets: query,value)
    self-attn key   (k):  frozen
    self-attn value (v):  frozen W + TRAINABLE LoRA A,B + magnitude m
    self-attn out   (o):  frozen
    FFN 768->3072->768:    frozen
    LayerNorm:            frozen
  |
  v
<s> token (no pooler)
  |
  v
head: dense(768->768) + out_proj(768->2)  [TRAINABLE (modules_to_save)]
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
| vocab size | 250,002 | AutoConfig.vocab_size |
| max positions | 514 | AutoConfig.max_position_embeddings |

Parameter counts across seeds identical: OK.

| Parameters | Value | Source |
|---|---|---|
| total params (metrics.json total_params) | 278,950,660 | metrics.json |
| trainable params | 905,474 | metrics.json |
| trainable % | 0.3246 | metrics.json |
| frozen = total - trainable | 278,045,186 | derived |
| adapter params (formula) | 313,344 | formula above |
| head params (formula) | 592,130 | formula above |
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
| 42 | 7 | 10 / 10 | no | 0.9038 | 0.9055 | 1261.3 | 1892.6 |
| 123 | 8 | 10 / 10 | no | 0.9026 | 0.9064 | 947.9 | 1892.6 |
| 2024 | 6 | 9 / 10 | yes | 0.9017 | 0.9046 | 1166.1 | 1893.5 |

Early stopping (epochs run < configured) triggered in 1 of 3 seeds. Epochs run is the last epoch with a Trainer eval entry in log_history.json.

![eval macro-F1 and training loss per epoch](figures/xlmr_dora_curves.png)

Left: validation macro-F1 per epoch (Trainer eval entries, all seeds). Right: training loss from log_history.json (logged every 50 steps), x-axis in epochs.

## 6. Results

| Metric (test set, mean ± sample std over seeds) | Value |
|---|---|
| accuracy | 0.9056 ± 0.0010 |
| abusive precision | 0.9224 ± 0.0088 |
| abusive recall | 0.8941 ± 0.0122 |
| abusive F1 | 0.9079 ± 0.0020 |
| non-abusive F1 | 0.9031 ± 0.0002 |
| macro-F1 (from CM) | 0.9055 ± 0.0009 |
| weighted-F1 (from CM) | 0.9056 ± 0.0010 |

Consistency: CM-derived macro-F1 vs metrics.json test_macro_f1 per seed: OK, OK, OK.

| seed | test n | TN | FP (pred Abusive, true not) | FN (pred not, true Abusive) | TP | FP/FN |
|---|---|---|---|---|---|---|
| 42 | 2997 | 1317 | 119 | 164 | 1397 | 0.726 |
| 123 | 2997 | 1303 | 133 | 147 | 1414 | 0.905 |
| 2024 | 2997 | 1335 | 101 | 185 | 1376 | 0.546 |

Error counts (mean over seeds): FP = 117.7, FN = 165.3, FP/FN = 0.712. Class 1 = Abusive, confusion matrix layout [[TN,FP],[FN,TP]].

Confusion matrix, seed 42 (highest validation macro-F1 = 0.9038):

![confusion matrix](figures/xlmr_dora_cm.png)

## 7. Compared with baseline

| metric | DoRA (this) | full FT (same backbone) | diff (this - full) | |diff| vs max(std) |
|---|---|---|---|---|
| test macro-F1 | 0.9055 ± 0.0009 | 0.9057 ± 0.0033 | -0.0002 | within seed noise |
| val macro-F1 | 0.9027 ± 0.0011 | 0.9040 ± 0.0011 | -0.0013 | outside seed noise |

| ratio | value |
|---|---|
| trainable params: this / full | 0.0033 (905,474 / 278,045,186) |
| peak GPU memory: this / full | 0.354 (1893 MB / 5353 MB) |
| training time: this / full (mean) | 0.719 (1125 s / 1564 s) |

Std-based noise rule: 'within seed noise' if |mean diff| < max(std_this, std_full). Sample std (n-1).

## 8. Interpretation

- Test macro-F1 mean 0.9055 over 3 seed(s) (std 0.0009).
- Seed with the highest validation macro-F1: 42 (val 0.9038, test 0.9055).
- Best epochs by seed: 7, 8, 6.
- Hypothesis (not tested): the adapter capacity limits how closely LoRA-type runs track full fine-tuning on this data; the diffs in section 7 are the only evidence here.
- Hypothesis (not tested): the magnitude term adds little over LoRA at this rank; compare with section 7 and the LoRA group.

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

- `results/study/xlm-roberta-base_dora_r8_a64_lr2e-4_e10_s42/` (metrics.json, log_history.json, model/)
- `results/study/xlm-roberta-base_dora_r8_a64_lr2e-4_e10_s123/` (metrics.json, log_history.json, model/)
- `results/study/xlm-roberta-base_dora_r8_a64_lr2e-4_e10_s2024/` (metrics.json, log_history.json, model/)
- `results/study/study_results.csv`
- `docs/models/figures/xlmr_dora_curves.png`, `docs/models/figures/xlmr_dora_cm.png`

