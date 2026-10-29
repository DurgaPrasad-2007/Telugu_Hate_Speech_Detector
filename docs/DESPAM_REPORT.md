# Despam report

`src/despam.py` flags astrology / love-spell template comments and writes cleaned copies to `data/clean/` (plus `data/clean/spam_flagged.csv`). Run `python src/despam.py` (counts) and `python src/despam.py --eval` (baseline). Test: `python tests/test_despam.py`.

Rule: a comment is spam if it contains any of 8 markers (the 7 of `src/template_audit.py` plus `జాతకం`; `వశీకరణ` stem covers `వశీకరణం`), or has char TF-IDF cosine >= 0.8 to a marker-flagged comment in any split. The near-duplicate rule added 0 comments beyond the markers on this data, so the markers alone decide.

Label convention in the CSVs: 0 = abusive, 1 = non-abusive.

## Counts (measured)

| split | n | spam | spam label 0 (abusive) | spam label 1 | kept |
|---|---|---|---|---|---|
| train | 24000 | 1969 | 1954 | 15 | 22031 |
| val | 3000 | 270 | 267 | 3 | 2730 |
| test | 3000 | 230 | 230 | 0 | 2770 |

Marker hits total 2469; near-duplicates added 0.

## TF-IDF (char_wb 2-5 + word 1-2) + LinearSVC, C chosen on matching val macro-F1

| train data | C | test raw (n=3000) | test despammed (n=2770) |
|---|---|---|---|
| raw | 0.3 | 0.9030 | 0.8949 |
| despammed | 0.3 | 0.8277 | **0.8952** |

The honest number is despammed-train on despammed-test: Macro-F1 0.8952 (accuracy 0.8957), 0.78 points below the raw 0.9030. The drop for the raw-trained model on the despammed test (0.8949) matches `template_audit.json`. The despammed-trained model scores 0.8277 on the raw test because it never saw the spam templates; that is expected and not a quality measure.
