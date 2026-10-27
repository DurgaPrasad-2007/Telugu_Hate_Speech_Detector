# Final results: what we built, what it beats, and the proof

Task: abusive vs non-abusive Telugu comments (MACD Telugu, 23,980 train / 3,000 val / 3,000 test).
Primary metric: macro-F1. Selection used validation only; the test set was read once per run.

## The verdict in four lines

1. **Parameter-efficient LoRA matches full fine-tuning.** XLM-R + LoRA reaches 0.9035 ± 0.0033 test
   macro-F1 against 0.9057 ± 0.0033 for full fine-tuning (3 seeds each), training 0.32% of the
   weights (0.89 M vs 278 M), using about a third of the GPU memory (1.75 GB vs 5.35 GB) and a
   3.6 MB adapter instead of a 1.1 GB checkpoint.
2. **The difference in accuracy is not statistically real.** On the same 2,997 test comments the
   paired bootstrap 95% CI for LoRA minus full fine-tuning includes zero for all three full-FT seeds
   (McNemar p = 0.24, 0.87, 0.81).
3. **A TF-IDF + LinearSVC baseline is just as accurate on this dataset.** It scores 0.9029 test
   macro-F1 (30 MB, 1.65 ms per comment on one CPU thread) and is statistically
   indistinguishable from the selected model (delta +0.0013, CI [-0.0084, +0.0110], p = 0.84).
   **The transformers do not buy accuracy here; what the study shows is that fine-tuning them
   cheaply costs nothing in accuracy.**
4. **Robustness comes from the input guard, not the model.** Without it every system falls to about
   0.71 mean accuracy under 18 disguise attacks; with the same guard all three reach 0.86.

## What we built (the final system)

`raw comment -> NFC normalisation + anti-obfuscation guard (src/canon.py) -> XLM-R base with a
LoRA adapter (r=8, alpha=64, query/value) -> temperature scaling (T=1.334) -> allow / review / block
thresholds (0.45 / 0.50) -> human review for the borderline band`

Diagram: `docs/figures/architecture.png`. Artifact: `models/best/` (adapter, 3.6 MB; loads on the
`FacebookAI/xlm-roberta-base` weights). Deployable INT8 ONNX: 279 MB, 0.8926 test macro-F1,
8.3 ms median per comment at 4 CPU threads on the dev PC.

## Proof 1: accuracy against every alternative

Test macro-F1, mean ± std over 3 seeds (single run for TF-IDF, which is deterministic):

| System | Test macro-F1 | Trainable params | Peak GPU | Stored size |
|---|---|---|---|---|
| XLM-R full fine-tune | 0.9057 ± 0.0033 | 278.0 M | 5,353 MB | 1.1 GB |
| XLM-R DoRA | 0.9055 ± 0.0009 | 0.91 M | 1,893 MB | 3.6 MB |
| MuRIL LoRA | 0.9052 ± 0.0030 | 0.30 M | 1,592 MB | 1.1 MB |
| **XLM-R LoRA (selected)** | **0.9035 ± 0.0033** | **0.89 M** | **1,753 MB** | **3.6 MB** |
| TF-IDF + LinearSVC | 0.9029 | 517 k features | CPU only | 30 MB |
| IndicBERTv2 LoRA | 0.9017 ± 0.0039 | 0.30 M | 1,747 MB | 1.1 MB |
| mBERT LoRA | 0.8946 ± 0.0015 | 0.30 M | 1,365 MB | 1.1 MB |
| XLM-R AdaLoRA | 0.8891 ± 0.0013 | 1.03 M | 1,760 MB | 4.0 MB |

Figures: `docs/figures/final_frontier.png` (accuracy vs cost), `docs/figures/final_scorecard.png`.

## Proof 2: paired significance (selected model vs each alternative)

Same 2,997 test comments, raw logits, 10,000 paired bootstrap resamples, exact McNemar test.
Full table: `results/study/significance.md`. Figure: `docs/figures/final_forest.png`.

| Selected minus ... | Delta macro-F1 | 95% CI | McNemar p | Reading |
|---|---|---|---|---|
| XLM-R full FT (seed 2024) | -0.0050 | [-0.0127, +0.0027] | 0.235 | no significant difference |
| XLM-R full FT (seed 42) | -0.0009 | [-0.0091, +0.0074] | 0.872 | no significant difference |
| XLM-R full FT (seed 123) | +0.0015 | [-0.0067, +0.0095] | 0.811 | no significant difference |
| XLM-R DoRA | -0.0004 | [-0.0070, +0.0063] | 1.000 | no significant difference |
| MuRIL LoRA | +0.0024 | [-0.0060, +0.0107] | 0.634 | no significant difference |
| MuRIL full FT | -0.0007 | [-0.0097, +0.0080] | 0.941 | no significant difference |
| XLM-R LoRA, paper config | +0.0037 | [-0.0037, +0.0107] | 0.355 | no significant difference |
| IndicBERTv2 LoRA | +0.0067 | [-0.0024, +0.0157] | 0.168 | no significant difference |
| TF-IDF + LinearSVC | +0.0013 | [-0.0084, +0.0110] | 0.840 | no significant difference |
| mBERT LoRA | +0.0097 | [+0.0007, +0.0193] | 0.054 | borderline: CI excludes 0, McNemar does not |
| XLM-R AdaLoRA | +0.0160 | [+0.0077, +0.0244] | 0.0002 | selected is better |

The selected model's own 95% bootstrap CI is [0.8936, 0.9148] (single run, seed 2024). Seed-level
Welch test, LoRA vs full FT: p = 0.47 with n = 3 per arm, which has very little power. No
correction for 11 comparisons was applied; only AdaLoRA survives one.

So the defensible claim is **non-inferiority within about one point**, not superiority. The test
set is 2,997 comments, so differences below roughly one point cannot be resolved by this study.

## Proof 3: robustness, and what actually produces it

Same 600 test comments, 7 sentence-level and 11 per-word disguises (spacing, zero-width
characters, repeated letters, emoji, leetspeak, star-censoring, case). Accuracy against true labels.
Full table: `results/study/robustness_compare.md`.

| System | Clean | Mean under attack | Worst attack |
|---|---|---|---|
| TF-IDF, no guard | 0.910 | 0.712 | 0.450 |
| XLM-R full FT, no guard | 0.907 | 0.710 | 0.455 |
| XLM-R LoRA (selected), no guard | 0.897 | 0.725 | 0.458 |
| XLM-R LoRA (selected) + guard | 0.898 | 0.856 | 0.552 |
| XLM-R full FT + guard | 0.907 | 0.860 | 0.550 |
| TF-IDF + guard | 0.910 | 0.863 | 0.560 |

The guard is the lever: +13 to +15 points of mean accuracy under attack for every system, and
no system gains over another once it is applied. Remaining weak spots for all systems:
star-censored words (0.55) and letter-spacing/repetition combined (0.76-0.84).

## Proof 4: where the models fail, and a data artifact

Computed by `src/paper_extras.py` and `src/template_audit.py` (outputs in `results/study/paper_extras.json`, `results/study/template_audit.json`).

- **Shared errors.** Of the selected model's 287 test errors (seed 2024, raw argmax), 122 (42.5%) are also made by full fine-tuning, DoRA, MuRIL LoRA and TF-IDF (86 labelled abusive, 36 non-abusive). Error overlap with full FT: Jaccard 0.60; with TF-IDF: 0.45. If one of the five were right on every comment, accuracy would be 0.959. These comments are candidates for label noise or context-dependent abuse; they were not re-annotated.
- **Advertisement-like templates.** Seven marker strings (Telugu for "problems", "for any problem", "progeny", "husband and wife", a love-spell term, and the formal "will be done / will be told") flag astrology and spell-service ads. They make up 15.5% / 16.3% / 14.5% of abusive comments in train / val / test, and almost all flagged posts are labelled abusive (1,928 of 1,935 in train; 227 of 227 in test). Every system gets 99.6% of them right. Removing them lowers test macro-F1 by about 0.8 points for all systems (LoRA 0.8964, full FT 0.9019, TF-IDF 0.8949), so the comparison does not change, but scores on this corpus are inflated by an easy subset.
- **Slices.** Transformers beat TF-IDF on comments over 20 words (0.902 / 0.911 vs 0.887); emoji comments are easiest (about 0.93). Single seed, small slices: indicative only.
- **Emoji disguises** are the one attack family where the model matters: the selected LoRA keeps 0.89 to 0.90 accuracy with words wrapped in emoji or trailing emoji spam, against 0.80 to 0.85 for TF-IDF and 0.82 to 0.84 for full FT.
- **Thresholds.** Raising the block threshold from 0.50 to 0.90 lifts block precision from 0.928 to 0.984 but cuts recall from 0.897 to 0.760 and raises the review share from 1.1% to 11.6% (validation).

## What this does and does not support

Supported by the evidence:

- LoRA/DoRA on query and value are an adequate replacement for full fine-tuning for this task.
- AdaLoRA is worse than LoRA here, with significance.
- The input guard roughly doubles the usable accuracy under text disguises for any classifier.
- Calibration works: ECE 0.035 to 0.013 after temperature scaling.

Not supported, and not claimed:

- That the transformer is more accurate than a classical baseline on this dataset (it is not).
- That the selected model beats full fine-tuning or MuRIL LoRA (a tie; selection between them is a
  0.0004 knife-edge on validation).
- Advantages the baseline might lack and this study did not test: implicit or sarcastic abuse,
  unseen vocabulary, code-mixed or Romanised Telugu, other domains. These are hypotheses.

## Practical recommendation

- **Need the smallest, fastest thing that works (a modest laptop):** TF-IDF + LinearSVC with the
  input guard. 30 MB, about 2 ms per comment, 0.90 macro-F1, and it matches the transformers
  under attack. Implementing it in the app is a small change if you want it.
- **Need the parameter-efficient transformer (the research contribution):** XLM-R + LoRA, shipping
  the 3.6 MB adapter. Same accuracy as full fine-tuning, a third of the memory.
- In either case keep the guard and a human-review band.

## Limitations

One dataset and one language; 3 seeds; a 2,997-comment test set; single-seed rank/alpha and
learning-rate grids run on MuRIL; robustness measured on 600 comments with synthetic disguises;
TF-IDF tuned only on C; latency from the dev PC, not a low-end laptop; original IndicBERT not
evaluated (gated).
