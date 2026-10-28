# Indic Hate/Abuse Benchmarks and Shared Tasks (2024 to Oct 2026): Macro-F1 Context for a 0.904 Telugu Binary Result

Scope note: search-result snippets and abstracts were used heavily. Most ACL/CEUR/arXiv PDFs could not be decoded by the fetch tool, so most numbers below are from HTML pages, abstracts, or search snippets. Anything not confirmed in a result table is marked UNVERIFIED.

## Q1. Typical SOTA macro-F1 on binary Indic abuse/hate tasks (2025-2026)

### Takeaway
No 2025-2026 published Telugu binary abuse/hate result at 0.904 macro-F1 was found. Reported binary Telugu numbers in the sources I could read are roughly 0.70-0.73 macro-F1 (translated/code-mixed Telugu hate, 2024-2025). Per-language binary numbers across Indic shared tasks cluster around 0.70-0.83. The 0.904 figure is above every comparable number found, so it should be checked for split leakage, dataset difference, or metric/averaging before being used as a benchmark claim.

### Cited Findings
- Telugu-English translated hate speech, binary "hate" vs "non-hate": best reported macro-F1 is 0.7314 for Hindi-Abusive-MuRIL in Table 1; the same paper's Table 2 gives 0.6990 for that model. Both tables are captioned "Translated" and the captions are ambiguous. Plain MuRIL 0.4115 (Table 1) and 0.3338 (Table 2). RoBERTa 0.6772 / 0.3829. Source: [Kakarla & Bulusu Venkata, arXiv 2502.10632 HTML](https://arxiv.org/html/2502.10632v1) and [PDF](https://arxiv.org/pdf/2502.10632) (the PDF version and HTML version disagree on some values; see caveats).
- DravidianLangTech 2024 Telugu code-mixed abusive detection (CUET_Binary_Hackers): best model an Indic sentence-similarity BERT at macro-F1 0.7013 (from search snippet). Other models (MuRIL, XLM-R) were tested but their numbers were not visible. Source: [aclanthology 2024.dravidianlangtech-1.32](https://preview.aclanthology.org/setup/2024.dravidianlangtech-1.32) (page fetch returned the snippet only; the PDF fetch failed to decode).
- DravidianLangTech 2026, Tamil binary abusive vs non-abusive (PrimeLine team): MuRIL 0.83, XLM-RoBERTa 0.79, mBERT 0.77 macro-F1, on the VALIDATION set, "under identical training conditions." Source: [aclanthology 2026.dravidianlangtech-1.51](https://aclanthology.org/2026.dravidianlangtech-1.51/). Not Telugu; reported on validation, not test.
- BLP-2025 Task 1 (Bangla hate speech, 1A/1B binary-style subtasks): most team papers report micro-F1 or weighted micro-F1, clustering 0.70-0.74. Only one macro-F1 figure was seen, about 71% for Retriv (Saha et al.), from a search summary. Sources: [Retriv arXiv 2511.07304](https://arxiv.org/pdf/2511.07304v1) (PDF body not decodable; figures from search summary, UNVERIFIED for macro vs micro), [NSU_MILab / BElite / CUET-NLP_Zenith / Gradient Masters BLP-2025 papers](https://preview.aclanthology.org/new-sigs/2025.banglalp-1.38/), [BElite](https://preview.aclanthology.org/new-sigs/2025.banglalp-1.36/).
- HASOC 2025 (multimodal memes, Hindi/Gujarati/Bangla/Bodo, abusive/hate): one team's own paper reports macro-F1 0.59597 (Hindi, rank 3), 0.61409 (Gujarati, rank 5), 0.60595 (Bangla, rank 6), 0.57184 (Bodo, rank 9). Another team reports 0.5719 for Hindi (rank 7/18). These are team-reported, not the overview leaderboard. Source: search snippets from CEUR FIRE 2025 team papers (URLs not fully retrieved; see Gaps).
- DravidianLangTech 2025 multimodal hate speech (Malayalam, Tamil, Telugu; YouTube speech + transcript): top reported F1s per search snippet are SSNTrio 0.7511 (Malayalam), 0.7332 (Tamil); lowes 0.3817 (Telugu). The snippets did not say whether the metric is macro or weighted. Team ML_Forge reports much lower numbers. Overview: [Jyothish Lal G et al., aclanthology 2025.dravidianlangtech-1.20](https://preview.aclanthology.org/setup/2025.dravidianlangtech-1.20) (the preview page returned 404 on fetch; numbers from search snippets, UNVERIFIED metric). This is a multimodal, multi-class-style task and is not directly comparable to a binary text task.

### Inferences
- Telugu binary abuse/hate in the public literature sits around 0.70-0.73 macro-F1 for the best reported system. A 0.904 claim is about 0.17 above that band. Plausible explanations: random vs. stratified splits, a different/easier dataset (e.g., in-domain YouTube or Twitter with strong lexical cues), validation vs. test confusion, or micro/weighted vs. macro averaging.
- Tamil binary abuse (2026) reaches about 0.83 macro-F1 on validation, which suggests 0.80-0.90 is attainable for Dravidian binary abuse on some datasets. So 0.904 is not impossible, but it needs a clear data/split description to be defensible.
- Shared-task leaderboard numbers for Telugu are weaker than for Tamil/Malayalam in the 2025 multimodal task (0.38 for Telugu vs 0.73-0.75), suggesting Telugu is the harder language in these tasks.

### Gaps
- Official DravidianLangTech 2025 abusive-text overview (Tamil/Malayalam, 2025.dravidianlangtech-1.115) PDF did not decode; the HTML page gave the task framing (abusive vs non-abusive, YouTube comments targeting women) but no leaderboard, team count, or per-language macro-F1. Link: [aclanthology 2025.dravidianlangtech-1.115](https://aclanthology.org/2025.dravidianlangtech-1.115).
- No 2025 or 2026 DravidianLangTech shared task covering Telugu binary abuse with an official overview was found. The 2026 proceedings page exists ([2026.dravidianlangtech-1](https://aclanthology.org/2026.dravidianlangtech-1/)) but its overview paper was not located.
- LT-EDI 2025: no results found (searches returned 2021 and 2022 hope-speech editions only; these are not abuse tasks). Marked as not found.
- IndicHateSpeech (Ghosh et al.) dataset numbers for 2024-2026 were not pulled in this session.
- Bengali/Marathi/Hindi binary macro-F1 leaderboards for 2025-2026 were not confirmed beyond the items above.

## Q2. Which techniques do winners use now?

### Takeaway
Winning or top team papers in 2025-2026 Indic abuse/hate tasks mostly use fine-tuned Indic-specific encoders (MuRIL and its abusive-finetuned variants, IndicBERT, BanglaBERT) and ensembles of 2-4 transformers (soft voting or weighted averaging). Translation to English followed by English encoders was used in the Telugu-English paper. Multi-task learning and speech/multimodal fusion appear in the multimodal task. LLM-based approaches were not seen in any of the top-result sources read in this session (UNVERIFIED that LLMs were absent, only that they were not reported in the sources read).

### Cited Findings
- Retriv (BLP-2025): transformer soft-voting ensemble of BanglaBERT, MuRIL, IndicBERTv2, plus multi-task learning; micro-F1 72.75 (1A) / 72.69 (1B) per the team's arXiv abstract; ~71% macro-F1 for best single model per summary. Source: [arXiv 2511.07304](https://arxiv.org/pdf/2511.07304v1) (PDF body not decoded; details from search summary).
- BLP-2025 NSU_MILab: probability averaging of BanglaBERT, XLM-R, IndicBERT, and Bengali Abusive MuRIL; 0.7235 (1A), 0.6981 then 0.7331 after refinements (metric likely micro-F1 or weighted, UNVERIFIED). Source: search summary of BLP-2025 papers.
- BLP-2025 BElite: weighted ensemble of BanglaBERT, mBERT, XLM-R; micro-F1 0.734 (1A), 0.7315 (1B). Source: [BElite preview](https://preview.aclanthology.org/new-sigs/2025.banglalp-1.36/).
- BLP-2025 CUET-NLP_Zenith: BanTriX, ensemble of BanglaBERT-I, XLM-R, BanglaBERT-II; weighted micro-F1 73.78 on Subtask 1C, 2nd place. Source: [CUET-NLP_Zenith preview](https://preview.aclanthology.org/new-sigs/2025.banglalp-1.38/).
- Telugu-English translated: Hindi-Abusive-MuRIL (a MuRIL checkpoint fine-tuned on abusive Hindi, from the Hate-speech-CNERG family) gave the best macro-F1 (0.7314). Translation via Google Translate. Source: [arXiv 2502.10632](https://arxiv.org/html/2502.10632v1).
- Tamil 2026 PrimeLine: plain MuRIL fine-tuning, no ensemble, no augmentation, no LLM mentioned. Source: [2026.dravidianlangtech-1.51](https://aclanthology.org/2026.dravidianlangtech-1.51/).
- Multimodal DravidianLangTech 2025: SSNTrio top scores; method not read. Team cantnlp used "bag-of-sounds" for multimodal, with a Ridge Classifier at 0.89 on Telugu text reported by KEC_AI_ZEROWATTS (search snippet). Source: [arXiv 2503.07862](https://arxiv.org/html/2503.07862v1) (not fetched in full).
- HASOC 2025 team papers: named models include IndicBERT, MuRIL, XLM-RoBERTa, mBERT (one team paper; scores not visible). Source: CEUR FIRE 2025 team papers via search snippets.

### Inferences
- The dominant recipe is "Indic-pretrained encoder + small ensemble." LLMs were not visible in the top-result sources I read, but the sample is small and biased toward a few papers.
- Translate-to-English is a live option for Telugu (the 2025 paper used it), but it lowered accuracy for some models between table variants, so its benefit is inconsistent.
- Ensembles in the BLP-2025 papers added roughly 1-3 points micro-F1 over single models in the summaries, but no clean ablation tables were read (UNVERIFIED).

### Gaps
- No winning-system description for a Telugu binary abuse task in 2025-2026 was found.
- Data augmentation winners: none found in the sources read.
- LLM-based winners (zero/few-shot or fine-tuned LLMs) for these shared tasks: none found; not verified as absent.

## Q3. MuRIL vs XLM-R head-to-head

### Takeaway
The only direct MuRIL vs XLM-R macro-F1 comparison found is Tamil binary (2026, validation set): MuRIL 0.83 vs XLM-R 0.79 (+0.04). No Telugu-specific head-to-head with XLM-R in a results table was found. In the Telugu-English translated paper, MuRIL was among the weakest models (0.4115), but XLM-R was not evaluated there.

### Cited Findings
- Tamil binary abusive (DravidianLangTech 2026, PrimeLine): MuRIL 0.83, XLM-RoBERTa 0.79, mBERT 0.77 macro-F1 on validation, identical training conditions. Source: [aclanthology 2026.dravidianlangtech-1.51](https://aclanthology.org/2026.dravidianlangtech-1.51/).
- Telugu code-mixed 2024 (CUET_Binary_Hackers): MuRIL and XLM-R were tested, but the snippet did not give their scores; the best model was Indic sentence-similarity BERT at 0.7013. Source: [2024.dravidianlangtech-1.32](https://preview.aclanthology.org/setup/2024.dravidianlangtech-1.32).
- MACI multilingual (2024): a user-aware model reports F1 gains over MuRIL, XLM-R, and M-BERT for Telugu, given only as percentage gains; no absolute scores were visible. Source: [arXiv 2410.21321](https://arxiv.org/pdf/2410.21321) (not decoded).
- Telugu-English (2025): MuRIL 0.4115 and 0.3338 across the two tables; XLM-R and BERT not evaluated. Source: [arXiv 2502.10632](https://arxiv.org/html/2502.10632v1).

### Inferences
- On Tamil, MuRIL leads XLM-R by 0.04 macro-F1 on validation. Telugu evidence is missing for XLM-R; a fair comparison would need same-split runs.
- MuRIL's weak Telugu-English numbers (0.33-0.41) suggest that a plain MuRIL checkpoint is sensitive to setup or to the translated text. The abusive-finetuned MuRIL (Hindi-Abusive-MuRIL) did much better (0.70-0.73), so task-adjacent pretraining seems to matter more than the base checkpoint.

### Gaps
- Telugu MuRIL vs XLM-R macro-F1 on the same split: not found.
- Whether Hindi-Abusive-MuRIL is on Hugging Face under a stable ID: the search did not return a direct "Hindi-Abusive-MuRIL" page. Related checkpoints exist under the Hate-speech-CNERG group, e.g. a Marathi codemixed abusive MuRIL referenced in [Spark NLP](https://sparknlp.org/2024/09/26/marathi_codemixed_abusive_muril_pipeline_mr.html). Check https://huggingface.co/Hate-speech-CNERG directly. UNVERIFIED ID.

## Positioning the 0.904 Telugu binary macro-F1

- Closest Telugu binary comparisons found: 0.7314 (translated Telugu, Hindi-Abusive-MuRIL, 2025), 0.7013 (Telugu code-mixed, sentence-similarity BERT, 2024). 0.904 is about 0.17 higher than these.
- A stronger claim needs: the exact dataset and label source, split method (stratified? group-by-author/video?), whether the 0.904 is on test or validation, macro vs weighted F1, and seed variance (report mean +/- std over seeds). Without these, the number should not be presented as SOTA.
- Suggested framing if the split is clean: "exceeds the best published Telugu binary macro-F1 we found (0.73, Kakarla & Bulusu Venkata 2025) on our split; comparisons are not like-for-like because the datasets differ."

## Key caveats and source-quality notes
- The Telugu-English paper's PDF and HTML versions disagree (RoBERTa 0.6772 vs 0.3829; Hindi-Abusive-MuRIL 0.7314 vs 0.6990), and both tables carry the same caption. Treat as unstable until the authors clarify.
- Several Telugu table rows in that paper are identical (0.3338 for MuRIL, TeluguHateBERT, IndicBERT in Table 2), which may indicate degenerate predictions.
- The 2026 Tamil number is validation-set, not test.
- BLP-2025 and HASOC 2025 numbers are mostly micro/weighted F1 or team-reported; they are not macro-F1 leaderboard results.
- Third-party model mirrors (mixpeek, promptlayer) were not used as evidence for model IDs.

## Sources (fetched or seen in search)
- https://arxiv.org/html/2502.10632v1 (Telugu-English hate, read)
- https://arxiv.org/pdf/2502.10632 (same paper PDF, seen in search)
- https://aclanthology.org/2026.dravidianlangtech-1.51/ (Tamil MuRIL vs XLM-R vs mBERT, read)
- https://aclanthology.org/2025.dravidianlangtech-1.115 (Tamil/Malayalam abusive women-targeting overview, HTML read, no results)
- https://preview.aclanthology.org/setup/2025.dravidianlangtech-1.20 (multimodal overview, 404 on fetch; snippet only)
- https://preview.aclanthology.org/setup/2024.dravidianlangtech-1.32 (CUET Telugu 2024, snippet)
- https://arxiv.org/pdf/2511.07304v1 (Retriv BLP-2025, PDF not decoded)
- https://preview.aclanthology.org/new-sigs/2025.banglalp-1.36/ and .../2025.banglalp-1.38/ (BLP-2025 ensembles, snippets)
- https://arxiv.org/pdf/2410.21321 (MACI, snippet)
- https://arxiv.org/html/2503.07862v1 (cantnlp multimodal, snippet)
- https://sparknlp.org/2024/09/26/marathi_codemixed_abusive_muril_pipeline_mr.html (Marathi abusive MuRIL reference)
- Not read in full (no decodable text): HASOC 2025 overview (FIRE 2025 proceedings, CEUR), LT-EDI 2025 (none found).
