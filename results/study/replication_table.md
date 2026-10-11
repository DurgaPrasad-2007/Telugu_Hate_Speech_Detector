| Configuration | n | Val macro-F1 | Test macro-F1 | Trainable params | Train time (s) | Peak GPU (MB) |
|---|---|---|---|---|---|---|
| Paper config (r4, a32, do 0.1, bs 8, len 128, 5 ep) | 3 | 0.8957 ± 0.0014 | 0.8985 ± 0.0018 | 739,586 | 706 | 1589 |
| Supervisor start (r16, a32, do 0.1, bs 16, len 128, 5 ep) | 3 | 0.8933 ± 0.0002 | 0.8953 ± 0.0033 | 1,181,954 | 454 | 1915 |
| Main study (r8, a64, do 0.05, bs 16, len 96, <=10 ep) | 3 | 0.9035 ± 0.0043 | 0.9035 ± 0.0033 | 887,042 | 774 | 1753 |
| Full fine-tune (reference) | 3 | 0.9040 ± 0.0011 | 0.9057 ± 0.0033 | 278,045,186 | 1564 | 5353 |
