# Multilingual and Indic Encoders on Telugu: Benchmark Evidence (as of 2026-10-28)

Scope: mmBERT, IndicBERT v2/v3, MuRIL, XLM-R, and 2025-2026 Indic/multilingual encoders, judged on Telugu numbers. Items marked **[UNVERIFIED]** could not be confirmed from a primary table or model card.

## Verification of the claim: IndicXTREME Telugu avg ~70.2 (IndicBERT v2) vs 64.2 (MuRIL) vs 56.4 (XLM-R)

### Takeaway
The three numbers are real and appear in the IndicXTREME paper (Table 24, "Results averaged across tasks using preferred metric from the IndicXTREME benchmark"). The row is labeled "IndicBERT", not "IndicBERT v2", and the table averages across tasks rather than reporting a classification-only score. The v2 label is an inference; treat the claim as partly verified.

### Cited Findings
- Table 24 Telugu (te) column, avg-across-tasks: mBERT 43.7; XLMR 56.4; MuRIL 64.2; v1-data 58.0; IndicBERT 70.2; +Samanantar 70.6; +Back-Trans 66.9; IndicBERT-SS 70.1 — [IndicXTREME/IndicBERT v2 paper, arXiv 2212.05409, HTML v3](https://arxiv.org/html/2212.05409v3)
- Table 24 also gives all-language avg: mBERT 39.6, XLMR 44.9, MuRIL 53.3, IndicBERT 57.8, IndicBERT-SS 60.3 — same source.
- Table 4 (all-language averages, per task, not Telugu-specific): IndicBERT v2 / MuRIL / XLM-R — IndicSentiment 88.3 / 85.1 / 84.0; IndicXNLI 73.0 / 72.4 / 69.7; IndicCOPA 62.7 / 58.9 / 60.1; IndicXParaphrase 56.9 / 60.8 / 56.7; MASSIVE Intent 78.8 / 77.2 / 66.6; Naamapadam (F1) 73.2 / 74.3 / 71.7; MASSIVE Slotfill (F1) 56.7 / 57.0 / 50.0; IndicQA (F1) 47.7 / 48.3 / 44.8; FLORES 69.4 / 52.3 / 3.1 — [arXiv 2212.05409 HTML v3, Table 4](https://arxiv.org/html/2212.05409v3). The table has no overall average. A simple mean of the nine columns, which is my own calculation, gives about 67.4 / 65.1 / 56.3.
- The abstract says IndicXTREME has "nine diverse NLU tasks covering 20 languages" and that IndicBERT v2 gives "an absolute improvement of 2 points over a strong baseline" — [arXiv abstract](https://arxiv.org/abs/2212.05409). The abstract does not name the baseline.

### Inferences
- The 70.2 / 64.2 / 56.4 figures are the Telugu rows of the paper's per-language table, so they are not Telugu-specific classification numbers. They average over the IndicXTREME tasks, which include Telugu-covering tasks such as classification, NLI, and QA.
- The "IndicBERT" row is most likely IndicBERT v2, because the paper's main model is v2 and the v1-data row (58.0) is shown separately. **[UNVERIFIED]** label mapping; confirm against the paper's text on the v1 vs v2 rows.
- Any citation should write these as "IndicBERT (v2 paper, Table 24)" and not as "IndicBERT v2 = 70.2 classification".

### Gaps
- The paper does not state which tasks Table 24 averages over; per-task Telugu tables are in the appendix (Tables 14, 24 referenced; not fully read).
- No Telugu-only classification or toxicity number from IndicXTREME exists in the text read.

## Q1: Best backbone for Telugu classification in 2026, with numbers?

### Takeaway
No single 2025-2026 benchmark compares mmBERT, IndicBERT v2, MuRIL, and XLM-R on a Telugu classification or toxicity task. The best-supported Telugu evidence is that MuRIL and IndicBERT v2 beat XLM-R on the IndicXTREME average, and that Indic-specific or Telugu-specific encoders beat XLM-R on some Telugu tasks. This is a conditional answer, not a clear winner.

### Cited Findings
- Telugu code-mixed hate speech (arXiv 2502.10632, Kakarla & Venkata, 2025): models compared are TeluguHateBERT, HateBERT, DeBERTa, MuRIL, IndicBERT, RoBERTa, and Hindi-Abusive-MuRIL. Hindi-Abusive-MuRIL was best in both original and translated settings. In the translated setting its macro-F1 was about 0.73, and IndicBERT about 0.37 — [arXiv 2502.10632](https://arxiv.org/abs/2502.10632v1). The HTML version is at https://arxiv.org/html/2502.10632v1. **[UNVERIFIED]** exact table numbers; the figures come from search-snippet summaries, not the paper's table.
- DravidianLangTech 2024 shared task (Sandalphon team): Telugu-BERT reported F1 0.77 on a transliteration-augmented Telugu code-mixed hate and offensive dataset, first place — [ACL Anthology 2024.dravidianlangtech-1.28](https://preview.aclanthology.org/setup/2024.dravidianlangtech-1.28). **[UNVERIFIED]** from search summary; read the paper for the table.
- DravidianLangTech 2026 (Telugu communicative-style classification, nine classes): MuRIL-based system 7th with Macro-F1 0.1703; the other models' scores are not given in the summary. A separate summary says IndicBERT v2 with partial layer freezing reached the top rank. These two claims conflict on the top-ranked model, so I cannot resolve them — [ACL Anthology, 2026.dravidianlangtech-1](https://aclanthology.org/volumes/2026.dravidianlangtech-1/). **[UNVERIFIED]**, conflicting.
- Telugu morphological tagging (a survey citing a Telugu-only BERT trained on about 8M sentences): gender tagging F1 0.778 for the Telugu-only BERT vs 0.624 for XLM-R and 0.527 for IndicBERT. The survey also says MuRIL and XLM-R over-fragment Telugu words — source is a survey summary, **[UNVERIFIED]**; the primary paper was not located.
- BHASHA 2025 word grouping (not Telugu-specific): MuRIL best of MuRIL, XLM-R, IndicBERT v2, at 58.18% exact match overall — [ACL Anthology, 2025.bhasha-1.18](https://aclanthology.org/2025.bhasha-1.18). **[UNVERIFIED]**, from search summary.

### Inferences
- MuRIL has the most consistent Telugu support across the sources found (the 2026 shared task and the BHASHA 2025 result), and it beats XLM-R on the IndicXTREME average (64.2 vs 56.4). Its position relative to IndicBERT v2 is unresolved: IndicXTREME favors IndicBERT v2 (70.2), while the 2026 shared-task summary is inconsistent.
- For Telugu abuse detection specifically, the strongest evidence is for models pretrained on abusive or Indic-language text (Hindi-Abusive-MuRIL, Telugu-BERT), not for a generic multilingual backbone.

### Gaps
- No head-to-head Telugu toxicity benchmark with mmBERT, IndicBERT v2, MuRIL, and XLM-R under the same split was found.
- The 2026 DravidianLangTech full results table was not retrieved.

## Q2: Does mmBERT cover Telugu, and how does it do vs XLM-R on low-resource classification?

### Takeaway
Yes, Telugu is in mmBERT's pretraining data, but only at a small share (about 0.1-0.5% per stage). The paper reports no Telugu-specific classification numbers, so there is no measured mmBERT-vs-XLM-R result for Telugu.

### Cited Findings
- Table 9 (Appendix D) lists `tel`: Pretrain 2.1B tokens (0.10%), Mid-Training 1.7B (0.29%), Decay-Eng 0.5B (0.48%), Decay-Cont 0.5B (0.52%), Decay-All 0.6B (0.49%), total 5.4B. The table covers "the first 90 languages" — [arXiv 2509.06888 HTML v1](https://arxiv.org/html/2509.06888v1).
- The same HTML lists no per-language Telugu result. Tables 2-7 contain no Telugu row. Table 3 (XTREME) gives averages only: XLM-R base 70.4, mmBERT base 72.8. Table 7 (XNLI/PAWS-X) compares EuroBERT and mmBERT only; its one Indic language is Hindi (XNLI: EuroBERT-210m 60.6, mmBERT-small 67.9, mmBERT-base 71.1). Table 5 (MTEB v2) reports category averages only.
- Abstract claim: mmBERT "significantly outperforms the previous generation of models on classification and retrieval tasks" — [arXiv 2509.06888](https://arxiv.org/abs/2509.06888). It also says it adds "over 1700 low-resource languages to the data mix only during the decay phase."
- Model card (jhu-clsp/mmBERT-base): no per-language list, only aggregate counts (1800+ languages, "1811 languages" metadata). No evaluation numbers. Qualitative XLM-R claim only — [Hugging Face model card](https://huggingface.co/jhu-clsp/mmBERT-base).
- Third-party summaries cite gains of 8-15 F1 points on low-resource tasks (Tigray, Faroese examples). **[UNVERIFIED]**; not checked against the paper.

### Inferences
- The paper's low-resource gains come from the decay-phase inclusion of 1700+ languages; Telugu is in the high-to-mid resource set at 5.4B tokens total, so its gain over XLM-R on Telugu is expected to be smaller than on the truly low-resource languages. This is an inference, not a measured result.
- The "classification" win is measured on XTREME averages and a non-Indic XNLI subset, not on Telugu.

### Gaps
- No mmBERT Telugu classification or toxicity number found. Per-language appendices of the full paper were not read in full.

## Q3: Any new Indic encoders released 2025-2026?

### Takeaway
IndicBERT v3 exists as a model family on Hugging Face, but its release date, language list, and evaluation are unstated, and no paper was found. No 2026 Indic encoder with published Telugu numbers was found.

### Cited Findings
- AI4Bharat HF collection "indicbert-v3": three base encoders, IndicBERT-v3-270M (0.3B), -1B, -4B, described as "state-of-the-art multilingual base encoder language models ... for Indic languages." The page does not list languages (Telugu not confirmed), gives no release year, no linked paper, and no benchmark numbers. Model dates shown as "Jun 12"; the collection shows "updated Jan 6". The year is not shown and the two dates conflict in the snippets — [HF collection](https://huggingface.co/collections/ai4bharat/indicbert-v3). **[UNVERIFIED]** for the year and language coverage.
- Original IndicBERT (v1): multilingual ALBERT, 12 languages, about 8.9B tokens, MIT license — [AI4Bharat page](https://indicnlp.ai4bharat.org/pages/indic-bert/). **[UNVERIFIED]** from search summary.
- Paramanu (arXiv 2401.18034): autoregressive Indic LMs including Telugu, trained from scratch. Not an encoder; latest arXiv version listed as 22 Jan 2026 — [arXiv 2401.18034](https://arxiv.org/abs/2401.18034v1).
- IndicSentEval (2025): probing study over 6 Indic languages including Telugu; Indic-specific multilingual models encode Indic linguistic properties better than universal ones, but universal models are more robust to perturbations — [arXiv 2410.02611](https://arxiv.org/pdf/2410.02611). Probing, not a classification benchmark.
- ModernBERT-style multilingual Indic encoders: no evaluation on Telugu or other Indic languages found.

### Inferences
- The only 2025-2026 Indic-specific encoder with a public release that I could locate is IndicBERT v3, and it is undocumented in the sources read. Do not use it as a Telugu backbone until its paper or language list is found.

### Gaps
- No arXiv paper for IndicBERT v3 found. No 2026 arXiv Indic encoder with Telugu results found.
- ModernBERT-multilingual variants on Telugu: no evidence found.

## Source-quality notes
- Primary and read in full: arXiv 2212.05409 (HTML v3, Tables 4, 24), arXiv 2509.06888 (HTML v1, Tables 3, 5, 7, 9), HF model cards for mmBERT and the IndicBERT v3 collection.
- Search-snippet only (not read in full): arXiv 2502.10632, DravidianLangTech 2024/2026 papers, BHASHA 2025, the survey on Telugu tagging. These are marked [UNVERIFIED] above.
- Date anomalies: the IndicBERT v3 collection shows "Jun 12" for the models and "Jan 6" for the collection in different sources. No year is stated.

## Summary of verified vs unverified
- Verified (primary table read): IndicXTREME Table 24 Telugu values 70.2 (IndicBERT row), 64.2 (MuRIL), 56.4 (XLMR); mmBERT Telugu token count in Table 9; mmBERT's lack of Telugu-specific results.
- Unverified: v2 label on the "IndicBERT" row; Telugu-specific classification numbers from search summaries; IndicBERT v3 release date and languages; mmBERT low-resource gains quoted by third parties.
