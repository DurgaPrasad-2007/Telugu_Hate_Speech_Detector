# Significance: selected model (A) vs other runs, TEST set

n = 2997 test comments. Delta = macro-F1(A) - macro-F1(other). 95% CI from paired bootstrap (10000 resamples, seed 0). McNemar = exact two-sided binomial test on discordant pairs.

A own macro-F1 = 0.9042, 95% bootstrap CI [0.8936, 0.9148].

| Comparison | Delta macro-F1 | 95% CI | McNemar p | b (A right, other wrong) | c (A wrong, other right) | Verdict |
|---|---|---|---|---|---|---|
| B: XLM-R full FT, seed 2024 | -0.0050 | [-0.0127, +0.0027] | 0.235 | 62 | 77 | no significant difference |
| B: XLM-R full FT, seed 42 | -0.0009 | [-0.0091, +0.0074] | 0.872 | 76 | 79 | no significant difference |
| B: XLM-R full FT, seed 123 | +0.0015 | [-0.0067, +0.0095] | 0.811 | 81 | 77 | no significant difference |
| C: XLM-R DoRA r8 a64, seed 2024 | -0.0004 | [-0.0070, +0.0063] | 1 | 51 | 52 | no significant difference |
| D: XLM-R AdaLoRA r8 a64, seed 2024 | +0.0160 | [+0.0077, +0.0244] | 0.000183 | 104 | 56 | A better |
| E: MuRIL LoRA r8 a64, seed 2024 | +0.0024 | [-0.0060, +0.0107] | 0.634 | 83 | 76 | no significant difference |
| E: MuRIL full FT, seed 2024 | -0.0007 | [-0.0097, +0.0080] | 0.941 | 89 | 91 | no significant difference |
| F: paper XLM-R LoRA r4 a32, seed 2024 | +0.0037 | [-0.0037, +0.0107] | 0.355 | 64 | 53 | no significant difference |
| G: mBERT LoRA r8 a64, seed 2024 | +0.0097 | [+0.0007, +0.0193] | 0.0536 | 120 | 91 | A better |
| G: IndicBERTv2-MLM-only LoRA r8 a64, seed 2024 | +0.0067 | [-0.0024, +0.0157] | 0.168 | 105 | 85 | no significant difference |
| H: TF-IDF (selected by val) baseline | +0.0013 | [-0.0084, +0.0110] | 0.84 | 112 | 108 | no significant difference |

## Seed-level (study_results.csv, XLM-R, 3 seeds each)

LoRA r8 a64 (n=3) vs full FT (n=3): delta mean = -0.0021, Welch t = -0.794, p = 0.472. n=3 per arm, so this test has very low power.
