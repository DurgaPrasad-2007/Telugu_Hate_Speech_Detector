# Error Analysis

Populated after the final model is selected and `src/analysis.py --action
error_analysis` is run against `results/error_analysis.csv`. Examples
below are shown masked (`src/text_mask.py`: first/last character kept,
middle starred out) so the abusive words stay unreadable at a glance
while their length/shape is still visible for manual review — never used
for anything the model sees, only for what a human reads here.

Selected model: `baseline__muril__full__telugu`. Its misclassified set is
`results/error_analysis.csv` (287 of 2997 test rows, 9.58%; heuristic split:
188 unclassified, 50 code-mixed, 25 low-confidence, 24 short).

The 60-comment hand reading below was done on the earlier XLM-R candidate's
errors (`results/xlmr_reference/error_analysis.csv`, 285 rows) before MuRIL was
selected. It transfers: 70% of MuRIL's errors are also XLM-R errors, and the
false-negative share is similar (60% vs 63%). MuRIL has more high-confidence
errors (49% at 0.9+ vs 36% for XLM-R), which strengthens finding 3.

## Method
- Export all misclassified test examples with prediction, true label,
  probability (`src/analysis.py:error_analysis`).
- Hand-read 60 of them (random seed 42 sample) and assign a category.

## Findings

Automatic heuristic split (script-mix / confidence / length) on all 285:
| category | count |
|---|---|
| unclassified (needs manual review) | 184 |
| code-mixed/script-mixed | 51 |
| low-confidence/ambiguous | 25 |
| short/ambiguous | 25 |

Hand-reading 60 of these surfaced four recurring, genuinely distinct
failure modes (not evenly split — heaviest first):

1. **Long, unpunctuated rambling comments (dominant failure mode).**
   Comments that run 30-80+ words with no sentence breaks bury the
   abusive clause among neutral filler. Model predicts non-abusive
   (false negative) even at low-to-mid confidence. This is the single
   largest chunk of the "unclassified" bucket -- it isn't code-mixing
   or short-text noise, it's the model failing to attend to one abusive
   clause inside a long non-abusive-sounding stream. Root cause is
   likely `max_length=96` for PEFT runs / the model's general weakness
   on long sequences, not a labeling problem.
2. **Code-mixed / Romanized-English abuse or praise.** Telugu script
   sentences with a key abusive or positive word written in Latin
   script (English word or Romanized Telugu, e.g. an English insult
   mid-sentence) are frequently missed. The XLM-R candidate handled this a bit better
   than the PEFT runs did (its pretraining includes English), but it
   still misses cases where the switched-script word carries most of
   the sentiment.
3. **High-confidence wrong predictions on short comments (both
   directions).** A meaningful fraction of errors are NOT low-confidence
   boundary cases -- several sit at 0.9+ confidence on the wrong label,
   for both false positives (mild venting/complaints flagged as abuse)
   and false negatives (short pointed insults predicted clean). These
   look like genuine model misses / possible label noise, not
   calibration-fixable uncertainty -- consistent with the modest ECE
   improvement from temperature scaling (see `results/calibration.json`)
   not touching this subset.
4. **Emoji-carried sentiment.** A smaller set of comments rely on an
   emoji (😡, 😂, 🤣, 🙁) to convey the abusive/sarcastic tone with mostly
   neutral surrounding text; the tokenizer sees the emoji as a token but
   the model doesn't reliably use it as a sentiment signal.

Short single-clause comments (the "short/ambiguous" bucket) were, on
inspection, mostly genuinely ambiguous out of context (e.g. a bare
phrase that reads negatively only if you know what it's replying to) --
this is closer to irreducible dataset noise than a fixable model gap.

**Practical implication for the `configs/policy.json` review-threshold
band:** most of the errors above sit near, but not at, the model's own
decision boundary; the long-comment failure mode is the exception
worth a targeted fix (chunking/attention pooling over long comments)
if this were taken further.
