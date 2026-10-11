# PEFT configuration selection

## Rule (validation macro-F1 only)

- 1. Group runs by config key (model, method, lr, epochs, and lora_r+lora_alpha only when method != full).
- 2. Keep configs with >= 3 seeds.
- 3. For each config: mean and sample SD (ddof=1) of val and test macro-F1; mean trainable params, peak GPU MB, train time.
- 4. top = config with highest mean val macro-F1.
- 5. Tie set = all configs with mean val >= top.mean_val - top.std_val.
- 6. Winner = tie-set member with the FEWEST trainable params (ties: higher mean val).
- 7. Winner run = winner config's seed with the highest val macro-F1. Selection uses validation only; test is never used.

- Top config: **xlm-roberta-base / full / lr2e-05 / e5 / full** (val 0.9040 ± 0.0011)
- Tie threshold: 0.9030

## Winner

- Config: **xlm-roberta-base / lora / lr0.0002 / e10 / r8a64**
- Winner run: `xlm-roberta-base_lora_r8_a64_lr2e-4_e10_s2024`
- Run val macro-F1: 0.9073; run test macro-F1: 0.9042
- Config val: 0.9035 ± 0.0043; test: 0.9035 ± 0.0033 (n=3)
- Trainable params: 887,042

## Ranking (all configs with >= 3 seeds, by mean val macro-F1)

| Rank | Config | n | Val F1 mean ± SD | Test F1 mean ± SD | Trainable params | Peak GPU MB | Train time s | In tie set |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | xlm-roberta-base / full / lr2e-05 / e5 / full | 3 | 0.9040 ± 0.0011 | 0.9057 ± 0.0033 | 278,045,186 | 5353 | 1564 | yes |
| 2 | xlm-roberta-base / lora / lr0.0002 / e10 / r8a64 | 3 | 0.9035 ± 0.0043 | 0.9035 ± 0.0033 | 887,042 | 1753 | 774 | yes |
| 3 | muril-base-cased / full / lr2e-05 / e5 / full | 3 | 0.9035 ± 0.0023 | 0.9049 ± 0.0028 | 237,557,762 | 4586 | 1250 | yes |
| 4 | xlm-roberta-base / dora / lr0.0002 / e10 / r8a64 | 3 | 0.9027 ± 0.0011 | 0.9055 ± 0.0009 | 905,474 | 1893 | 1125 |  |
| 5 | muril-base-cased / lora / lr0.0002 / e10 / r8a64 | 3 | 0.9026 ± 0.0020 | 0.9052 ± 0.0030 | 296,450 | 1592 | 718 |  |
| 6 | bert-base-multilingual-cased / lora / lr0.0002 / e10 / r8a64 | 3 | 0.8999 ± 0.0010 | 0.8946 ± 0.0015 | 296,450 | 1365 | 680 |  |
| 7 | IndicBERTv2-MLM-only / lora / lr0.0002 / e10 / r8a64 | 3 | 0.8985 ± 0.0014 | 0.9017 ± 0.0039 | 296,450 | 1747 | 673 |  |
| 8 | muril-base-cased / dora / lr0.0002 / e10 / r8a64 | 3 | 0.8984 ± 0.0013 | 0.9005 ± 0.0030 | 314,882 | 1731 | 1078 |  |
| 9 | xlm-roberta-base / lora / lr0.0002 / e5 / r4a32 | 3 | 0.8957 ± 0.0014 | 0.8985 ± 0.0018 | 739,586 | 1589 | 706 |  |
| 10 | xlm-roberta-base / lora / lr0.0002 / e5 / r16a32 | 3 | 0.8933 ± 0.0002 | 0.8953 ± 0.0033 | 1,181,954 | 1915 | 454 |  |
| 11 | muril-base-cased / adalora / lr0.0002 / e10 / r8a64 | 3 | 0.8892 ± 0.0030 | 0.8945 ± 0.0003 | 444,194 | 1599 | 1282 |  |
| 12 | xlm-roberta-base / adalora / lr0.0002 / e10 / r8a64 | 3 | 0.8855 ± 0.0015 | 0.8891 ± 0.0013 | 1,034,786 | 1760 | 1618 |  |

## Excluded (< 3 seeds)

- muril-base-cased / lora / lr0.0002 / e10 / r4a16: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r4a32: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r4a64: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r8a16: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r8a32: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r16a16: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r16a32: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r16a64: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r32a16: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r32a32: 1 seed(s)
- muril-base-cased / lora / lr0.0002 / e10 / r32a64: 1 seed(s)
- muril-base-cased / lora / lr0.0001 / e10 / r8a64: 1 seed(s)
- muril-base-cased / lora / lr0.0005 / e10 / r8a64: 1 seed(s)
- bert-base-multilingual-cased / dora / lr0.0002 / e10 / r8a64: 1 seed(s)
