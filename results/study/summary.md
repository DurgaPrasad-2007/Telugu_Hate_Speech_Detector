# Study summary

Source: `results/study/study_results.csv`. Model selection used **validation macro-F1 only**; test macro-F1 is shown for reporting and was not used to choose any configuration.

## (a) LoRA rank x alpha (MuRIL, lr 2e-4, seed 42)

**Validation macro-F1**

| rank \ alpha | a16 | a32 | a64 |
|---|---|---|---|
| r4 | 0.8896 | 0.8978 | 0.9002 |
| r8 | 0.8915 | 0.8978 | 0.9048 |
| r16 | 0.8982 | 0.8957 | 0.9006 |
| r32 | 0.8944 | 0.8973 | 0.9019 |

**Test macro-F1 (reporting only)**

| rank \ alpha | a16 | a32 | a64 |
|---|---|---|---|
| r4 | 0.8951 | 0.9014 | 0.8988 |
| r8 | 0.8991 | 0.8961 | 0.9078 |
| r16 | 0.8994 | 0.9011 | 0.9025 |
| r32 | 0.8984 | 0.9025 | 0.9055 |

## (b) Learning rate (MuRIL, LoRA r8 a64, seed 42)

| LR | Epochs | Val F1 | Test F1 (reporting) | Train time (s) |
|---|---|---|---|---|
| 0.0001 | 10 | 0.8922 | 0.8981 | 646 |
| 0.0002 | 10 | 0.9048 | 0.9078 | 487 |
| 0.0005 | 10 | 0.8438 | 0.8422 | 193 |

## (c) Method comparison (MuRIL, r8 a64, seed 42)

| Method | LR | Epochs | Val F1 | Test F1 (reporting) | Trainable params | Trainable % | Train time (s) | Peak GPU (MB) |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0002 | 10 | 0.9048 | 0.9078 | 296,450 | 0.125 | 487 | 1593 |
| dora | 0.0002 | 10 | 0.8983 | 0.8972 | 314,882 | 0.132 | 1036 | 1731 |
| adalora | 0.0002 | 10 | 0.8926 | 0.8949 | 444,194 | 0.187 | 1502 | 1600 |
| full | 2e-05 | 5 | 0.9018 | 0.9078 | 237,557,762 | 100.000 | 1446 | 4589 |

## (d) Backbone (r8 a64, lr 2e-4, seed 42)

| Method | Backbone | Val F1 | Test F1 (reporting) | Trainable params | Train time (s) |
|---|---|---|---|---|---|
| lora | MuRIL | 0.9048 | 0.9078 | 296,450 | 487 |
| lora | xlm-roberta-base | 0.8988 | 0.8999 | 887,042 | 719 |
| lora | bert-base-multilingual-cased | 0.8989 | 0.8962 | 296,450 | 715 |
| dora | MuRIL | 0.8983 | 0.8972 | 314,882 | 1036 |
| dora | xlm-roberta-base | 0.9038 | 0.9055 | 905,474 | 1261 |
| dora | bert-base-multilingual-cased | 0.8987 | 0.8966 | 314,882 | 511 |

## (e) Multi-seed (MuRIL; mean ± sample std, ddof=1)

| Method | LR | n | Seeds present | Val F1 mean ± std | Test F1 mean ± std (reporting) |
|---|---|---|---|---|---|
| lora | 0.0002 | 3 | 42,123,2024 | 0.9026 ± 0.0020 | 0.9052 ± 0.0030 |
| dora | 0.0002 | 3 | 42,123,2024 | 0.8984 ± 0.0013 | 0.9005 ± 0.0030 |
| full | 2e-05 | 3 | 42,123,2024 | 0.9035 ± 0.0023 | 0.9049 ± 0.0028 |

Seeds requested: 42, 123, 2024. `n` is the number of seeds found in the CSV.

## (f) Professor's E1-E7 framework

Metrics are recomputed from each run's `metrics.json` confusion matrix (class 1 = Abusive); val macro-F1 comes from `study_results.csv`. Cells are mean ± sample std (ddof=1); std shows `-` when n < 2. Experiments with no completed run show `not run`.

### (f1) Experiments E1-E7 (LoRA family: r8 a64 lr 2e-4 e10; full: lr 2e-5 e5)

| Exp | Setup | n | Seeds | Test acc | Abusive P | Abusive R | Abusive F1 | Non-abusive F1 | Macro-F1 | Weighted-F1 | Val macro-F1 | Trainable params | Trainable % | Train time (s) | Peak GPU (MB) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | FacebookAI/xlm-roberta-base full | 3 | 42,123,2024 | 0.9058 ± 0.0032 | 0.9138 ± 0.0100 | 0.9045 ± 0.0051 | 0.9091 ± 0.0024 | 0.9022 ± 0.0042 | 0.9057 ± 0.0033 | 0.9058 ± 0.0032 | 0.9040 ± 0.0011 | 278,045,186 | 100.000 | 1564.1 | 5353 | 1060.68 |
| E2 | xlm-roberta-base LoRA | 3 | 42,123,2024 | 0.9036 ± 0.0034 | 0.9289 ± 0.0135 | 0.8828 ± 0.0218 | 0.9050 ± 0.0052 | 0.9020 ± 0.0015 | 0.9035 ± 0.0033 | 0.9036 ± 0.0034 | 0.9035 ± 0.0043 | 887,042 | 0.318 | 774.4 | 1753 | 3.39 |
| E4 | xlm-roberta-base AdaLoRA | 3 | 42,123,2024 | 0.8892 ± 0.0013 | 0.9035 ± 0.0058 | 0.8815 ± 0.0090 | 0.8923 ± 0.0020 | 0.8859 ± 0.0010 | 0.8891 ± 0.0013 | 0.8893 ± 0.0013 | 0.8855 ± 0.0015 | 1,034,786 | 0.371 | 1617.7 | 1760 | 3.96 |
| E5 | xlm-roberta-base DoRA | 3 | 42,123,2024 | 0.9056 ± 0.0010 | 0.9224 ± 0.0088 | 0.8941 ± 0.0122 | 0.9079 ± 0.0020 | 0.9031 ± 0.0002 | 0.9055 ± 0.0009 | 0.9056 ± 0.0010 | 0.9027 ± 0.0011 | 905,474 | 0.325 | 1125.1 | 1893 | 3.46 |
| E6 | bert-base-multilingual-cased LoRA | 3 | 42,123,2024 | 0.8947 ± 0.0015 | 0.9173 ± 0.0155 | 0.8772 ± 0.0148 | 0.8966 ± 0.0009 | 0.8926 ± 0.0032 | 0.8946 ± 0.0015 | 0.8947 ± 0.0015 | 0.8999 ± 0.0010 | 296,450 | 0.166 | 680.4 | 1365 | 1.14 |
| E7 | IndicBERTv2-MLM-only LoRA | 3 | 42,123,2024 | 0.9018 ± 0.0039 | 0.9171 ± 0.0033 | 0.8922 ± 0.0101 | 0.9044 ± 0.0044 | 0.8990 ± 0.0034 | 0.9017 ± 0.0039 | 0.9018 ± 0.0039 | 0.8985 ± 0.0014 | 296,450 | 0.107 | 672.7 | 1747 | 1.14 |

### (f2) MuRIL block (same definitions)

| Exp | Setup | n | Seeds | Test acc | Abusive P | Abusive R | Abusive F1 | Non-abusive F1 | Macro-F1 | Weighted-F1 | Val macro-F1 | Trainable params | Trainable % | Train time (s) | Peak GPU (MB) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MuRIL full | MuRIL full | 3 | 42,123,2024 | 0.9050 ± 0.0028 | 0.9188 ± 0.0098 | 0.8971 ± 0.0109 | 0.9077 ± 0.0029 | 0.9021 ± 0.0032 | 0.9049 ± 0.0028 | 0.9050 ± 0.0028 | 0.9035 ± 0.0023 | 237,557,762 | 100.000 | 1250.1 | 4586 | 906.23 |
| MuRIL LoRA | MuRIL LoRA | 3 | 42,123,2024 | 0.9052 ± 0.0031 | 0.9185 ± 0.0012 | 0.8977 ± 0.0078 | 0.9080 ± 0.0034 | 0.9023 ± 0.0026 | 0.9052 ± 0.0030 | 0.9053 ± 0.0030 | 0.9026 ± 0.0020 | 296,450 | 0.125 | 718.5 | 1592 | 1.14 |
| MuRIL AdaLoRA | MuRIL AdaLoRA | 3 | 42,123,2024 | 0.8946 ± 0.0003 | 0.9138 ± 0.0039 | 0.8806 ± 0.0043 | 0.8969 ± 0.0005 | 0.8921 ± 0.0007 | 0.8945 ± 0.0003 | 0.8946 ± 0.0003 | 0.8892 ± 0.0030 | 444,194 | 0.187 | 1281.9 | 1599 | 1.70 |
| MuRIL DoRA | MuRIL DoRA | 3 | 42,123,2024 | 0.9006 ± 0.0031 | 0.9127 ± 0.0056 | 0.8947 ± 0.0130 | 0.9036 ± 0.0040 | 0.8974 ± 0.0021 | 0.9005 ± 0.0030 | 0.9006 ± 0.0030 | 0.8984 ± 0.0013 | 314,882 | 0.132 | 1077.7 | 1731 | 1.21 |

### (f3) E3 rank study (MuRIL LoRA, lr 2e-4, e10, seed 42; best alpha per rank by val)

| Rank | Best alpha (by val) | Val macro-F1 | Test macro-F1 | Trainable params | Trainable % | Train time (s) |
|---|---|---|---|---|---|---|
| r4 | a64 | 0.9002 | 0.8988 | 148,994 | 0.063 | 911.2 |
| r8 | a64 | 0.9048 | 0.9078 | 296,450 | 0.125 | 486.8 |
| r16 | a64 | 0.9006 | 0.9025 | 591,362 | 0.248 | 896.7 |
| r32 | a64 | 0.9019 | 0.9055 | 1,181,186 | 0.495 | 763.0 |

### (f4) Per-run confusion matrices, seed 42 (rows = actual [non-abusive, abusive]; cols = predicted; `[[TN, FP], [FN, TP]]`)

- **E1** FacebookAI/xlm-roberta-base full (seed 42, `xlm-roberta-base_full_lr2e-5_e5_s42`): `[[1301, 135], [149, 1412]]`
- **E2** xlm-roberta-base LoRA (seed 42, `xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s42`): `[[1356, 80], [220, 1341]]`
- **E4** xlm-roberta-base AdaLoRA (seed 42, `xlm-roberta-base_adalora_r8_a64_lr2e-4_e10_s42`): `[[1279, 157], [175, 1386]]`
- **E5** xlm-roberta-base DoRA (seed 42, `xlm-roberta-base_dora_r8_a64_lr2e-4_e10_s42`): `[[1317, 119], [164, 1397]]`
- **E6** bert-base-multilingual-cased LoRA (seed 42, `bert-base-multilingual-cased_lora_r8_a64_lr2e-4_e10_s42`): `[[1343, 93], [218, 1343]]`
- **E7** IndicBERTv2-MLM-only LoRA (seed 42, `IndicBERTv2-MLM-only_lora_r8_a64_lr2e-4_e10_s42`): `[[1303, 133], [151, 1410]]`
- **MuRIL full** MuRIL full (seed 42, `full_lr2e-5_e5_s42`): `[[1308, 128], [148, 1413]]`
- **MuRIL LoRA** MuRIL LoRA (seed 42, `lora_r8_a64_lr2e-4_e10_s42`): `[[1309, 127], [149, 1412]]`
- **MuRIL AdaLoRA** MuRIL AdaLoRA (seed 42, `adalora_r8_a64_lr2e-4_e10_s42`): `[[1313, 123], [192, 1369]]`
- **MuRIL DoRA** MuRIL DoRA (seed 42, `dora_r8_a64_lr2e-4_e10_s42`): `[[1315, 121], [187, 1374]]`

- **E3 r4 a64** MuRIL LoRA (seed 42, `lora_r4_a64_lr2e-4_e10_s42`): `[[1301, 135], [168, 1393]]`
- **E3 r8 a64** MuRIL LoRA (seed 42, `lora_r8_a64_lr2e-4_e10_s42`): `[[1309, 127], [149, 1412]]`
- **E3 r16 a64** MuRIL LoRA (seed 42, `lora_r16_a64_lr2e-4_e10_s42`): `[[1311, 125], [167, 1394]]`
- **E3 r32 a64** MuRIL LoRA (seed 42, `lora_r32_a64_lr2e-4_e10_s42`): `[[1313, 123], [160, 1401]]`
