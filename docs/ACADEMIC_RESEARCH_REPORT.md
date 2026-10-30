# Telugu abusive-comment detection: an empirical study of backbones, obfuscation defence, and review policy

Every number below is copied from a committed report in this repo (`PROJECT_STATUS.md`, `ROBUSTNESS.md`, `SEEDS_RELABEL_REPORT.md`, `POLICY_REPORT.md`, `FUNCTIONAL_TEST_REPORT.md`). Nothing was re-run for this document. Citations were limited to the 2026 papers already checked against their arXiv abstracts in `APPROACH_SURVEY.md`; no full text was read.

## 1. Setup
Task: binary abusive vs non-abusive classification of short Telugu-script comments. Data: despammed split in `data/clean`; test n = 2770; metric macro-F1 at threshold 0.5. Systems: TF-IDF + LinearSVC (Platt-calibrated), MuRIL-base-cased and XLM-R-base with LoRA+ (lp8, lr 5e-5), and a mean-probability ensemble. Inference adds an anti-obfuscation canonicalizer (`src/canon.py`), temperature calibration, and an allow / review / block policy.

## 2. Findings

### 2.1 Backbones are statistically tied (single seed)
| model | macro-F1 |
|---|---|
| TF-IDF | 0.8960 |
| MuRIL | 0.8943 |
| XLM-R | 0.8944 |
| TF-IDF + MuRIL ensemble | 0.9052 |

Ensemble gain over TF-IDF is +0.0045, paired-bootstrap 95% CI [-0.0038, +0.0131], which includes zero. A 2 MB-class linear model matches the transformers, so serving cost, not accuracy, should drive the choice.

### 2.2 Seed variance is the same size as most effects
Three seeds (1, 2, 42), MuRIL: 0.8967 ± 0.0043 (clean) vs 0.8952 ± 0.0035 (augmented); ensemble 0.9051 ± 0.0009 vs 0.9040 ± 0.0024. Differences of 0.001 to 0.002 are inside seed sd (n = 3, sample sd).

### 2.3 Obfuscation defence
Canonicalization removes zero-width characters, separators inside words, letter spacing, stretched letters, and (new) lookalikes of the anusvara `ం` (Latin o/O, Cyrillic о, Greek ο, Telugu digit zero) when glued to a Telugu letter. On 300 abusive test comments with every word disguised, fraction still blocked or reviewed (MuRIL, no guard -> guard): letter spacing 1% -> 66%, dots 15% -> 99%, dashes 2% -> 98%, repetition 54% -> 92%, combined 55% -> 90%. Zero-width, emoji, leetspeak and uppercase attacks were already near 100% because MuRIL's tokenizer ignores them. Cost on ordinary text: val 0.9029 vs 0.9029, test 0.9048 vs 0.9045.

The homoglyph rule has unit self-checks (`python src/canon.py`) but **no measured recovery rate**; it is not in the table above.

Open weaknesses: star-censored words (`వె**వ`) are treated as abusive (a dataset artifact), blocking about 99% of clean text censored the same way; doubling every letter of clean text wrongly blocks 12-15%; spaced letters are caught only 66% of the time (89% with the XLM-R candidate).

### 2.4 Functional behaviour (HateCheck-style, n = 149 high-confidence cases)
Ensemble pass rate 96.6% ± 0.67 (clean) vs 98.2% ± 0.39 (augmented MuRIL member); negated hate 70% -> 93%. Seed ranges do not overlap (98.0-98.7 vs 96.0-97.3). Per-group n is 5-12, and both the cases and the augmentation templates are model-authored from shared vocabulary, so this gain is in-distribution to the templates. This follows the functional-testing approach of SEAHateCheck (Ng et al., 2026), which also reports slang, implicit hate and counter-speech as weak spots.

### 2.5 Review policy
Conformal thresholds at alpha = 2% wrongful blocks: realized false-block 1.95%; target miss 5 / 10 / 20% realized 4.57 / 8.92 / 19.34%, with review load 21.8 / 13.0 / 3.2%. The guarantee is marginal, so one calibration set scatters around the targets. Routing uncertain items to humans follows the conformal-moderation line of Villate-Castillo et al. (2025).

### 2.6 Explanations
Word attribution is leave-one-word-out probability drop, the same family as the explainer in X-MuTeST (Rehman et al., AAAI 2026). We did not evaluate it against human rationales.

### 2.7 Label noise
687 out-of-fold suspects; a 98-label relabel changed TF-IDF test F1 by +0.0022, CI [-0.0022, +0.0065]. Test labels carry the same noise, so this cannot show whether the flips were correct.

## 3. Threats to validity
- Gold-sheet relabels, functional cases and augmentation templates were written by a Claude model; none is native-speaker verified.
- Out of domain, F1 drops: zero-shot code-mixed set (n = 387) 0.76, Latin-script comments 0.75, vs 0.90 in-domain. No Romanized test set exists.
- All new runs except the 3-seed study are single seed; the ensemble CI includes zero.
- Implicit, sarcastic and double-meaning abuse was not tested.
- `app.py` contains a hard-coded perturbation table (e.g. "mean 72.5% -> 85.6%") that does not match `ROBUSTNESS.md`; treat `ROBUSTNESS.md` as the source until reconciled.
- Raw vs calibrated probability: `raw_prob` is the T = 1 softmax; calibration is a single temperature fit on validation. The UI shows both, but their gap has not been analysed.

## 4. References (2026, abstracts checked in `APPROACH_SURVEY.md`)
- Rehman et al. "X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and a Novel LLM-consulted Explanation Framework". arXiv:2601.03194, AAAI 2026.
- Ng et al. "SEAHateCheck". arXiv:2603.16070, TALLIP (Mar 2026).
- Dang, Mitrovic, Granitzer. "Toward Generalized Cross-Lingual Hateful Language Detection with Web-Scale Data and Ensemble LLM Annotations". arXiv:2604.09625 (LREC 2026 submission): continued MLM pretraining is a candidate next step; not tried here.

Earlier: Villate-Castillo, Del Ser, Sanz. arXiv:2411.04090, Neurocomputing 647 (2025).
