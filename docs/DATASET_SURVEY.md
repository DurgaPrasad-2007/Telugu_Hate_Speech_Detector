# Telugu dataset survey (2026-10-29)
VERIFIED = page/API read or rows inspected this session. NOT VERIFIED = not checked. No guessing.

| Dataset | URL | Size / labels | Script | License | Access | Status |
|---|---|---|---|---|---|---|
| mounikaiiith/Telugu-Hatespeech | https://huggingface.co/datasets/mounikaiiith/Telugu-Hatespeech | 24,599/3,510/7,033; yes=263/31/76 (~1% hate) | Telugu (100% rows contain Telugu chars) | CC-BY-4.0 | open, DOWNLOADED to external_data/ | VERIFIED. News-style sentences, very imbalanced; domain differs from ShareChat comments |
| mounikaiiith/Telugu_Sentiment | https://huggingface.co/datasets/mounikaiiith/Telugu_Sentiment | same splits; neutral/pos/neg (train 13213/6635/4751) | Telugu | CC-BY-4.0 | open, DOWNLOADED | VERIFIED |
| mounikaiiith/Telugu_Emotion, Telugu-Sarcasm | https://huggingface.co/datasets/mounikaiiith/Telugu_Emotion , .../Telugu-Sarcasm | ~35K rows, binary-ish "no"/other labels (first rows seen) | Telugu | CC-BY-4.0 | open, not downloaded | VERIFIED exists; label sets not tallied |
| ai4bharat/IndicSentiment (te) | https://huggingface.co/datasets/ai4bharat/IndicSentiment | te test + validation json; product reviews, Positive/Negative | Telugu | not stated in card metadata | open, not downloaded | VERIFIED te files exist; sentiment only, not moderation |
| ai4bharat/Sangraha | https://huggingface.co/datasets/ai4bharat/Sangraha | verified/tel ~15.4 GB, plus synthetic/tel_Latn (romanized wiki) | Telugu + Latin | CC-BY-4.0 | open, too big, not downloaded | VERIFIED; unlabeled text only |
| HOLD-Telugu (DravidianLangTech 2024) | https://aclanthology.org/2024.dravidianlangtech-1.8 | 4,000 train (1,939 hate/2,061 non) + 500 test (per papers) | Latin code-mixed (YouTube) | unknown | NOT FOUND: no download link in any source seen (CodaLab task page not located) | Exists; data NOT obtainable via links found. Contact organizers |
| X-MuTeST Telugu | https://github.com/ziarehman30/X-MuTeST | ~4.5K Telugu, hate/non + token rationales; reported to be HOLD-Telugu transliterated to Telugu script | Telugu | research-only, no redistribution | GATED: Google request form in README | VERIFIED README; data not accessed |
| UnityAI-Guard (arXiv 2503.23088) | https://arxiv.org/abs/2503.23088 | Telugu train 62,309 toxic / 95,282 neutral; test 2,500 each (per paper) | Telugu | unknown | NOT FOUND: paper says release planned; none located | Unverified release |
| FredZhang7/toxi-text-3M | https://huggingface.co/datasets/FredZhang7/toxi-text-3M | ~3M rows, 55 langs | mixed | apache-2.0 (per search snippet) | open | Telugu rows NOT VERIFIED (search endpoint errored) |
| Kaggle saipanda9/telugu-codemixed-hate-speech | https://www.kaggle.com/datasets/saipanda9/telugu-codemixed-hate-speech | unknown | unknown | unknown | needs Kaggle login | page content not readable; NOT VERIFIED |
| DravidianCodeMix (github bharathichezhiyan) | https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset | Tamil/Kannada/Malayalam | - | - | open | VERIFIED NO Telugu |
| Telugu-English abusive (arXiv 2504.21026) | github.com/Ranveer098/Code-Mix-and-Code-Switch | 2,000 Roman comments | Latin | research | liveness NOT checked | unchanged from MODEL_SHORTLIST |
| PolygloToxicityPrompts, Jigsaw multilingual, HASOC Telugu, Bhasha/Hate-Alert, ShareChat/Moj other releases, MACD other splits | - | - | - | - | NOT FOUND for Telugu in searches (Jigsaw multilingual and HASOC not Telugu as far as seen; not exhaustively checked) | NOT FOUND |

## Downloaded (external_data/, untrusted, nothing executed)
- mounikaiiith_Telugu-Hatespeech (about 8 MB), mounikaiiith_Telugu_Sentiment (about 8.5 MB), each with README (source/license).

## Notes
- Only ~370 positives in Telugu-Hatespeech; usable as extra negatives/domain-shift test, weak as hate training data.
- Label polarity differs from ShareChat (0=abusive); map explicitly.
