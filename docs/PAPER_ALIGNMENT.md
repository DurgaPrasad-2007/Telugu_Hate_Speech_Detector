# Alignment with the research paper and the supervisor's plan

Sources compared:

- **Paper**: Kaunang, Hakim, Thenata, *Indonesian Hate Speech Detection Across Diverse Domains
  Using Parameter-Efficient Fine-Tuning with IndoBERT and LoRA*, Jurnal Minfo Polgan 15(2), 2026.
- **Supervisor plan**: the E1-E7 framework from the supervisor's ChatGPT discussion
  (full FT, LoRA, rank tuning, AdaLoRA, DoRA, mBERT, IndicBERT; 3-5 seeds; Macro-F1 primary).
- **This work**: the same idea on **pure Telugu** (MACD Telugu files only), not Indonesian.

Status key: **Yes** = matches; **Adapted** = changed on purpose for Telugu/single-domain;
**Deviation** = differs from the spec and the difference may matter; **Not done**.

## 1. Specification matrix

| Item | Paper | Supervisor | This work (main study) | Status |
|---|---|---|---|---|
| Task | Binary hate/abusive vs normal | Binary sequence classification | Binary, abusive vs non-abusive | Yes |
| Language / data | Indonesian, 3 domains (Instagram, Twitter, news) | MACD Telugu CSVs | Telugu MACD train/val/test only; code-mixed files not used | Adapted |
| Backbone | IndoBERT-base-p1 | XLM-R, mBERT, IndicBERT | XLM-R (primary), MuRIL, mBERT, IndicBERTv2 | Yes |
| IndicBERT | - | `indic-bert` | Original is gated (HTTP 403); `ai4bharat/IndicBERTv2-MLM-only` used | Deviation (substitute) |
| Adaptation methods | LoRA | Full FT, LoRA, AdaLoRA, DoRA | Full FT, LoRA, DoRA, AdaLoRA (3 seeds each on XLM-R and MuRIL) | Yes |
| LoRA target modules | query, value | query, value | query, value | Yes |
| Classifier head trainable | implied | `modules_to_save=["classifier"]` | Head is trained and saved (`adapter_config.json`: `modules_to_save=[classifier, score]`) | Yes |
| Rank / alpha search | r in {4,8,16,32} x alpha in {16,32,64} | r in {4,8,16,32} | Full 12-cell grid, but on **MuRIL, one seed**; best (r=8, a=64) reused on XLM-R | Deviation (grid not repeated on XLM-R) |
| Paper's final r / alpha | r=4, alpha=32 | start r=16, alpha=32 | Main study r=8, alpha=64 (validation-selected); r=4/a=32 and r=16/a=32 replicated separately (section 3) | Adapted |
| Learning rate | search {1e-4, 2e-4, 5e-4} | 2e-4 | 1e-4, 2e-4, 5e-4 tested on MuRIL (one seed); 2e-4 used everywhere | Deviation (not searched per backbone) |
| Epochs | search {3,5,10,15,20} + early stop (patience 3, val F1) | 5 | Up to 10, early stop patience 3 on val macro-F1, best checkpoint restored | Deviation (no epoch grid, cap 10) |
| Batch size | 8 | 16 | 16 | Deviation (paper-exact runs use 8) |
| Max length | 128 | 128 | 96 | Deviation (paper-exact runs use 128) |
| LoRA dropout | 0.1 | 0.1 | 0.05 (library default carried from the earlier code) | Deviation (paper-exact runs use 0.1) |
| Optimizer / precision | AdamW, FP16 | AdamW, wd 0.01 | AdamW, wd 0.01, FP16, cosine schedule, 10% warmup | Yes |
| Preprocessing | lowercase, strip emoji/punctuation/digits/hashtags, slang lexicon | keep punctuation, emoji, hashtags; Unicode normalisation only | NFC, URL/mention masking, whitespace collapse; punctuation/emoji kept | Adapted (follows supervisor; Telugu has no case; no Telugu slang lexicon exists in the repo) |
| Data augmentation | back-translation for scarce Instagram data | - | None. Single domain, 23,980 train rows, ~52/48 class split | Adapted (not applicable) |
| Domain balancing | stratified downsampling of Twitter/news | - | None (single domain) | Adapted (not applicable) |
| Splits | stratified, 20% test, 10% of rest val | train/val/test CSVs | Provided MACD 80/10/10 CSVs; exact-text train-to-val/test leakage dropped at load time (2+3 rows) | Adapted |
| Seeds | single run reported | 3-5 seeds, mean +/- std | 3 seeds (42, 123, 2024) for the main comparisons; grid and lr search single-seed | Yes (partly single-seed) |
| Selection criterion | F1 | validation Macro-F1 | validation macro-F1; config chosen on 3-seed mean with a stated tie rule (`results/study/selection.md`) | Yes |
| Metrics | Accuracy, P, R, F1, confusion matrix | + abusive F1, non-abusive F1, macro-F1, weighted-F1 | All of these per run, plus calibration (ECE, Brier) | Yes |
| Efficiency metrics | trainable fraction (1-2%), training loss | + GPU memory, time, adapter size | Trainable params and %, peak GPU memory, train time, adapter size | Yes |
| Trainable fraction | ~1-2% | reported | 0.11-0.37% (LoRA family) vs 100% (full) | Yes (lower: dropout/rank differ) |
| Lexical analysis | word clouds, top-100 terms per class | - | Top-term bar charts + log-odds (`docs/LEXICAL_ANALYSIS.md`); no word-cloud package installed | Yes (bar charts instead of clouds) |
| Evaluation setting | multi-domain hold-out (not leave-one-domain-out) | - | Single-domain hold-out; leave-one-domain-out not applicable | Adapted |
| Hardware | NVIDIA T4 | Colab-class | RTX 4060 Ti 8 GB | Adapted |

## 2. Where this work goes beyond the paper

- Compares four adaptation methods and four backbones, with full fine-tuning as a baseline, instead of one LoRA model.
- Three seeds with mean and standard deviation; the paper reports single runs.
- Selection uses validation only and a stated rule, then reports the test set once per run.
- Adds calibration (temperature scaling), allow/review/block thresholds, obfuscation robustness tests, and an INT8 export audit that found the earlier committed INT8 model unusable.

## 3. Paper-exact and supervisor-start replication (XLM-R, 3 seeds)

The main study deviates from the paper on dropout, max length and batch size. To check
whether that matters, two configurations were trained with the **exact** published settings:

| Config | r | alpha | dropout | batch | max len | lr | epochs | patience |
|---|---|---|---|---|---|---|---|---|
| paper (Table 2) | 4 | 32 | 0.1 | 8 | 128 | 2e-4 (paper tuned over {1e-4,2e-4,5e-4}; final value not stated) | 5 | 3 |
| supervisor start | 16 | 32 | 0.1 | 16 | 128 | 2e-4 | 5 | 3 |

### Results (XLM-R base, LoRA on query/value, 3 seeds: 42, 123, 2024; mean ± sample std)

| Configuration | n | Val macro-F1 | Test macro-F1 | Trainable params | Train time (s) | Peak GPU (MB) |
|---|---|---|---|---|---|---|
| Paper config (r4, a32, do 0.1, bs 8, len 128, 5 ep) | 3 | 0.8957 ± 0.0014 | 0.8985 ± 0.0018 | 739,586 | 706 | 1589 |
| Supervisor start (r16, a32, do 0.1, bs 16, len 128, 5 ep) | 3 | 0.8933 ± 0.0002 | 0.8953 ± 0.0033 | 1,181,954 | 454 | 1915 |
| Main study (r8, a64, do 0.05, bs 16, len 96, <=10 ep) | 3 | 0.9035 ± 0.0043 | 0.9035 ± 0.0033 | 887,042 | 774 | 1753 |
| Full fine-tune (reference) | 3 | 0.9040 ± 0.0011 | 0.9057 ± 0.0033 | 278,045,186 | 1564 | 5353 |

Best epoch (1-indexed): paper config 4 of 5 in all three seeds; supervisor-start config 5 of 5
in two of three seeds; main-study config 5-6 of 10.

What this shows:

- The paper's exact settings reach **0.8985 ± 0.0018** test macro-F1 and the supervisor's
  starting settings **0.8953 ± 0.0033**. Both are below the main-study LoRA (0.9035 ± 0.0033)
  and full fine-tuning (0.9057 ± 0.0033) by roughly 0.5-1 point, i.e. about 1.5-3 seed standard
  deviations. The gap is small but consistent in validation too (0.8957 and 0.8933 vs 0.9035).
- The methodology holds up: even with the paper's exact recipe, LoRA reaches about 99% of full
  fine-tuning macro-F1 while training 0.27% (r=4) to 0.43% (r=16) of the parameters.
- The main study's settings (higher alpha, up to 10 epochs, dropout 0.05) were not worse than the
  paper's, so the deviations listed in section 1 did not hurt; they probably helped slightly.
- **What this does not show**: which change is responsible. The paper config and the main
  study differ in rank, alpha, dropout, batch size, max length and epoch budget at once. In
  two of three seeds the supervisor-start runs peaked on their last allowed epoch, so the
  5-epoch cap likely cost some score (hypothesis, not isolated). Ranking `r=4` above `r=16` in
  the paper-style runs also contradicts nothing here: seed spread is 0.002-0.003 and the
  gap between them is 0.003.
- Test macro-F1 across all three LoRA settings stays within 0.9 +/- 0.005, so the practical
  conclusion (adapter-based fine-tuning is sufficient for this task) does not depend on the
  setting chosen.


## 4. What is still different from the paper, and why it is acceptable or not

- **Preprocessing** is deliberately lighter than the paper's. The supervisor's plan asks to keep punctuation and emoji, and the repo's robustness tests show the model uses them. The paper's regex cleaning was not replicated.
- **Learning rate and epoch search** were not repeated per backbone. Results for DoRA, AdaLoRA and the other backbones therefore use LoRA-tuned settings and may understate them.
- **IndicBERT v1** was not evaluated (gated). IndicBERTv2 results are a substitute, not a replication.
- **Cross-domain**: the paper itself notes its evaluation is not leave-one-domain-out. This work evaluates one domain only, so it makes no cross-domain claim.
- **Dataset difference**: the paper's data are Indonesian and partly news text; MACD Telugu is short social-media comments. Absolute scores are not comparable (paper F1 0.81-0.83; this work macro-F1 about 0.90). The comparison that is meaningful is the method-level conclusion: low-rank adapters on query/value match or nearly match full fine-tuning at a small fraction of trainable parameters.
