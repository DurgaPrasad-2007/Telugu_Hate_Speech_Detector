# Akshara variant report (measured only)

Command: `.venv\Scripts\python -I src\tfidf_akshara.py`. Test = data/telugu_test.csv (after the loader's leak-drop).
Baseline = models/tfidf/tfidf_best.joblib behind canon.py. Variant = new `src/tfidf_akshara.py`
(char_wb 2-5 + akshara 1-3-gram TF-IDF, LinearSVC C=0.3, trained on telugu train, guard = `akshara.normalize` + `translit_text`).
Argmax predictions, no Platt/thresholds.

Attacks (every word disguised, using aksharas as units, so they apply to Telugu script) on the abusive test rows each
model already flags (n=1376 baseline / 1369 variant) = "caught"; "clean FP" = same attack on label-0 rows predicted abusive.

| metric | baseline+canon | char+akshara |
|---|---|---|
| test macro-F1 | 0.9039 | 0.8995 |
| clean FP rate (unattacked) | 7.24% | 7.66% |
| caught, spaced | 80.3% | 94.4% |
| caught, dash | 89.5% | 89.9% |
| caught, dot | 90.9% | 95.9% |
| caught, star | 76.1% | 99.6% |
| clean FP, spaced | 18.1% | 6.6% |
| clean FP, dash | 12.1% | 7.6% |
| clean FP, dot | 14.2% | 14.3% |
| clean FP, star | 20.0% | 44.2% |
| clean rows altered by guard rule (canon.py vs akshara.normalize, excl. NFC/whitespace) | 2.65% | 2.23% |

Notes
- Macro-F1 is 0.4 pt lower for the variant (single seed, no CI computed; not shown to be significant either way).
- Star: the variant "catches" 99.6% but clean FP is 44.2%. Stars between 2-akshara words are not collapsed
  (only runs of >=3 lone letters are), and the model treats leftover `*` as abusive (the known dataset artifact in docs/ROBUSTNESS.md).
  So the star gain is mostly that artifact, not recovery.
- Dot clean FP is unchanged (14.3%) for the same reason on short words.
- Translit (`src/translit.py`) is only unit-tested (vaddu/voddu/woddu -> one key and one Telugu string); the Telugu-script
  test set has no Romanized text, so its effect on accuracy is unmeasured.
- Attacks here are my per-word akshara separators, not the exact ones in src/stress_test.py/robustness.py
  (robustness.py's transforms need the torch classifier); numbers are not comparable to docs/ROBUSTNESS.md.
