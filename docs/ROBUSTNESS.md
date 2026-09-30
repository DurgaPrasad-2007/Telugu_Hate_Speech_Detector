# Robustness

Selected model: `baseline__muril__full__telugu`. All numbers below come from
`src/stress_test.py` and `src/robustness.py` against the real model and real
test/validation comments; raw outputs are in `results/`.

## Brutal obfuscation stress test
300 real abusive test comments (the model already blocked them) and 300 clean
ones, with **every word** disguised. "Caught" = still block or review.
"No guard" = the model without `src/canon.py` (the emoji-stripped second view is
on in both columns); "With guard" = the deployed pipeline.

| disguise | caught, no guard | caught, with guard | clean wrongly blocked | XLM-R candidate, with guard |
|---|---|---|---|---|
| zero-width joiner | 100% | 100% | 0.0% | 100% |
| zero-width non-joiner | 100% | 100% | 0.0% | 100% |
| letter spacing | 1% | 66% | 5.0% | 89% |
| dots between letters | 15% | 99% | 5.7% | 100% |
| dashes between letters | 2% | 98% | 5.3% | 100% |
| emoji inside word | 100% | 100% | 0.7% | 100% |
| emoji wrapped | 100% | 100% | 0.0% | 100% |
| char repetition | 54% | 92% | 11.7% | 97% |
| leetspeak digits | 100% | 100% | 1.7% | 100% |
| star-censored | 100% | 100% | 99.3% | 100% |
| UPPERCASE | 99% | 99% | 0.0% | 100% |
| trailing emoji spam | 100% | 100% | 0.3% | 100% |
| COMBO (repeat+ZWJ+emoji) | 55% | 90% | 14.7% | 97% |

What this shows:
- MuRIL's WordPiece tokenizer already ignores zero-width characters (identical
  tokens with and without them, verified) and copes with emoji and casing. The
  guard matters for **letters separated by spaces, dots or dashes** (1-15% -> 66-99%)
  and stretched letters (54% -> 92%).
- Cost on ordinary text: none measurable. Deployed pipeline val 0.9029, test
  0.9048 macro-F1 vs 0.9029 / 0.9045 without the guard.
- Trade-off of the backbone choice: the XLM-R candidate, with the same guard, caught spaced letters 89% of
  the time vs MuRIL's 66%. MuRIL was selected on validation F1 (see the model card).
- The guard is deliberately narrow: doubled Telugu letters (`నన్ను`, `చచ్చి`) are
  legitimate, so 2x repeats are not collapsed, only runs of 3+.

Known remaining weaknesses (measured, not hidden):
- Doubling *every* letter of clean text wrongly blocks 12-15% of clean comments.
- The model treats star-censored words (`వె**వ`) as abusive (a dataset artifact);
  censoring every word of a clean sentence is blocked ~99%.
- Letters spaced with no word boundaries are caught 66% of the time.
- **Out-of-domain / code-mixed:** zero-shot on the separate code-mixed test set
  (387 comments, never trained on) macro-F1 is 0.76
  vs 0.90 on Telugu; Latin-script comments 0.75.
- Implicit, sarcastic or double-meaning abuse was not tested and is not covered.

## Milder perturbation suite (200 validation comments, baseline accuracy 0.89)
| transform | prediction flips | accuracy vs true label |
|---|---|---|
| char repeat | 8.5% | 0.855 |
| extra spacing | 9.0% | 0.830 |
| emoji insert | 0.0% | 0.890 |
| punctuation insert | 1.0% | 0.890 |
| capitalization | 0.5% | 0.885 |
| number substitution | 2.5% | 0.885 |
| random case | 1.0% | 0.880 |

## Development history
The first hardening pass was developed against the XLM-R candidate, where
zero-width characters, spaced letters and in-word emoji initially defeated the
model (0-32% caught). Its raw log is `results/xlmr_reference/stress_test_BEFORE_defense.log`.
When MuRIL was selected on validation F1, the stress test was re-run from scratch
on MuRIL (numbers above).
