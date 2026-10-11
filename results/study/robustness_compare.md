Accuracy on 600 test comments (true labels); flip = predictions changed vs the clean text.

| Perturbation | TF-IDF + LinearSVC | XLM-R LoRA (selected), raw | XLM-R LoRA (selected), with input guard | XLM-R full FT, raw | TF-IDF + input guard | XLM-R full FT, with input guard |
|---|---|---|---|---|---|---|
| clean | 0.910 | 0.897 | 0.898 | 0.907 | 0.910 | 0.907 |
| sentence: char_repeat | 0.832 (12% flip) | 0.767 (20% flip) | 0.862 (11% flip) | 0.778 (18% flip) | 0.882 (6% flip) | 0.870 (10% flip) |
| sentence: extra_spacing | 0.790 (16% flip) | 0.763 (16% flip) | 0.863 (4% flip) | 0.775 (16% flip) | 0.873 (5% flip) | 0.887 (4% flip) |
| sentence: emoji_insert | 0.892 (3% flip) | 0.902 (1% flip) | 0.903 (1% flip) | 0.902 (1% flip) | 0.892 (3% flip) | 0.902 (1% flip) |
| sentence: punctuation_insert | 0.893 (2% flip) | 0.897 (2% flip) | 0.897 (2% flip) | 0.908 (2% flip) | 0.895 (2% flip) | 0.908 (2% flip) |
| sentence: capitalization | 0.910 (0% flip) | 0.902 (0% flip) | 0.902 (0% flip) | 0.908 (0% flip) | 0.910 (0% flip) | 0.908 (0% flip) |
| sentence: number_substitution | 0.903 (1% flip) | 0.890 (1% flip) | 0.893 (1% flip) | 0.912 (1% flip) | 0.905 (1% flip) | 0.912 (1% flip) |
| sentence: random_case | 0.910 (0% flip) | 0.897 (0% flip) | 0.898 (0% flip) | 0.908 (0% flip) | 0.910 (0% flip) | 0.908 (0% flip) |
| every word: zero-width joiner | 0.688 (28% flip) | 0.458 (50% flip) | 0.898 (0% flip) | 0.455 (51% flip) | 0.910 (0% flip) | 0.907 (0% flip) |
| every word: zero-width non-joiner | 0.450 (50% flip) | 0.458 (50% flip) | 0.898 (0% flip) | 0.455 (51% flip) | 0.910 (0% flip) | 0.907 (0% flip) |
| every word: letter spacing | 0.608 (41% flip) | 0.458 (50% flip) | 0.765 (16% flip) | 0.455 (51% flip) | 0.835 (12% flip) | 0.845 (11% flip) |
| every word: dots between letters | 0.477 (48% flip) | 0.480 (48% flip) | 0.890 (2% flip) | 0.508 (46% flip) | 0.893 (3% flip) | 0.902 (3% flip) |
| every word: dashes between letters | 0.453 (50% flip) | 0.515 (44% flip) | 0.888 (3% flip) | 0.487 (50% flip) | 0.890 (3% flip) | 0.897 (3% flip) |
| every word: emoji inside word | 0.460 (49% flip) | 0.758 (22% flip) | 0.905 (2% flip) | 0.655 (34% flip) | 0.892 (2% flip) | 0.900 (3% flip) |
| every word: emoji wrapped | 0.800 (15% flip) | 0.890 (4% flip) | 0.892 (4% flip) | 0.818 (16% flip) | 0.800 (15% flip) | 0.818 (16% flip) |
| every word: char repetition | 0.493 (46% flip) | 0.643 (35% flip) | 0.758 (21% flip) | 0.570 (43% flip) | 0.820 (14% flip) | 0.775 (21% flip) |
| every word: leetspeak digits | 0.903 (1% flip) | 0.893 (1% flip) | 0.895 (1% flip) | 0.907 (2% flip) | 0.905 (1% flip) | 0.907 (2% flip) |
| every word: star-censored | 0.558 (48% flip) | 0.552 (49% flip) | 0.552 (49% flip) | 0.550 (48% flip) | 0.560 (48% flip) | 0.550 (48% flip) |
| every word: UPPERCASE | 0.910 (0% flip) | 0.897 (1% flip) | 0.898 (1% flip) | 0.908 (0% flip) | 0.910 (0% flip) | 0.908 (0% flip) |
| every word: trailing emoji spam | 0.852 (8% flip) | 0.900 (4% flip) | 0.902 (4% flip) | 0.837 (14% flip) | 0.852 (8% flip) | 0.837 (14% flip) |
| every word: COMBO (repeat+ZWJ+emoji) | 0.452 (50% flip) | 0.572 (45% flip) | 0.758 (25% flip) | 0.507 (48% flip) | 0.822 (13% flip) | 0.757 (22% flip) |
| **mean over attacks** | 0.712 | 0.725 | 0.856 | 0.710 | 0.863 | 0.860 |
