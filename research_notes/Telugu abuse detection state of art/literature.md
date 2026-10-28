# Telugu / Indic abuse and hate-speech detection: literature, 2024 to Oct 2026

Coverage is thin: about 8 tool calls, mostly search snippets. Only abstracts were verified, and no results tables were read. Many F1 numbers below are unverified or missing.

## Newest papers (2025-2026) and the models they use

### Takeaway
Telugu-specific work is sparse and small-scale. I found no 2026 arXiv paper that reports macro-F1 on MACD Telugu. The 2025-2026 Telugu papers use encoder models: MuRIL variants, IndicBERT, and multilingual transformers. Several of them add LoRA or classical ML.

### Cited Findings
- DravidianLangTech 2026 (July 2026): "Abusive Content Detection in Telugu-English Code-Mixed Social Media Using Hybrid Transformer Architectures", Reddy and Bellamkonda. It compares traditional ML with transformers, normalizes transliteration, combines TF-IDF and FastText features, and fine-tunes multilingual transformers. The abstract gives no scores, and the dataset and model names were not visible to me. — [ACL Anthology](https://aclanthology.org/2026.dravidianlangtech-1.1/)
- Kakarla and Venkata, arXiv Feb 2025, "Code-Mixed Telugu-English Hate Speech Detection". It compares 7 transformers (TeluguHateBERT, HateBERT, DeBERTa, MuRIL, IndicBERT, RoBERTa, Hindi-Abusive-MuRIL), all fine-tuned with LoRA, on original and Google-Translated English text.
  - Hindi-Abusive-MuRIL was best on both versions.
  - Translation helped most models' accuracy.
  - The abstract has no numbers, and the PDF was unreadable by my tool. — [arXiv](https://arxiv.org/abs/2502.10632)
- Pandey et al., arXiv Apr 2025. Creates a code-mixed Telugu-English dataset (about 2k comments; the Nepali count is garbled in the abstract) and compares LR, RF, SVM, NN, LSTM, CNN and LLMs. It uses 10-fold CV and t-tests. No F1 values were visible to me. — [arXiv](https://arxiv.org/abs/2504.21026v1)
- DravidianLangTech 2025 multimodal hate speech task (Malayalam, Tamil, Telugu). The labels are Gender, Political, Religious, Personal Defamation and Non-Hate. The best Telugu macro-F1 was 0.3817 (team "lowes"), and other teams were lower (0.15 to 0.16). The task is multi-class and multimodal, so it is not comparable to a binary text task. A separate team reports 0.89 text / 0.87 audio with a Ridge classifier, but that setup is unclear and not comparable. — [Overview](https://preview.aclanthology.org/setup/2025.dravidianlangtech-1.20)
- DravidianLangTech 2025 abusive-text-against-women task covered only Tamil and Malayalam, not Telugu. IndicBERT got macro-F1 0.75 there (secondary source: an author page). — [Author page](https://aclanthology.org/people/bojja-revanth-reddy/)
- Other Indic work (not Telugu), 2024-2026:
  - Devanagari hate speech shared task (Hindi/Nepali): MuRIL, IndicBERTv2 and Gemma-2 ensembles. Encoders were strong and the LLM was comparable. — [arXiv 2411.06850](https://arxiv.org/abs/2411.06850), [arXiv 2412.08163](https://arxiv.org/pdf/2412.08163)
  - HASOC 2025 study: MuRIL best (secondary source). — [CEUR T9-17](https://ceur-ws.org/Vol-4173/T9-17.pdf)
  - 2026 Springer chapter on Hindi hostile posts: bagging, boosting and voting over MuRIL, HindiBERT and IndicBERT embeddings. Paywalled, so no scores seen.
  - Survey of low-resource hate speech detection (2024). — [arXiv 2411.19017](https://arxiv.org/pdf/2411.19017)

### Inferences
- Common techniques in this subfield: MuRIL, IndicBERT, XLM-R, ensembles, and occasional LLM prompting. The most recent Telugu work remains small-data and encoder-centric.
- The Telugu-English code-mixed (romanized) papers use different data from MACD, which is native-script.

### Gaps
- No searched source gave 2026 arXiv Telugu results. I did not find the DravidianLangTech 2026 shared-task overviews.
- The scores in the Kakarla, Pandey and Reddy papers were not retrieved.
- LLM prompting or fine-tuning results specific to Telugu were not found.
- HASOC 2024/2025 Telugu tracks, IberLEF, and Semantic Scholar/HF were not searched.

## MACD results and how about 0.90 compares

### Takeaway
MACD (Gupta et al., NeurIPS 2022, which is older than the 2024 window) is the reference dataset. I could not retrieve per-language Telugu F1, so I cannot say with sourced numbers where 0.904 sits.

### Cited Findings
- Older than the window: MACD has 150K ShareChat comments in 5 Indic languages. The Telugu subset is about 30K samples at 52% abusive. Labels are binary, and the label mapping is reported as 0 = abusive and 1 = non-abusive. Check this against the project's own mapping. — [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a7c4163b33286261b24c72fd3d1707c9-Abstract.html), [paper PDF](https://papers.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Paper-Datasets_and_Benchmarks.pdf)
- Older than the window: the authors release AbuseXLMR (XLM-R further pretrained on social-media comments in 15+ Indic languages). They claim it outperforms XLM-R and MuRIL on four Indic datasets. Code is at github.com/ShareChatAI/MACD. — [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a7c4163b33286261b24c72fd3d1707c9-Abstract.html)
- Translation-transfer result (older than the window): MuRIL fine-tuned on English abuse data and tested on English-translated MACD Telugu got F1 34.92. This is only a cross-lingual transfer baseline. — [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Paper-Datasets_and_Benchmarks.pdf)

### Inferences
- A newer Telugu MACD comparison would need the NeurIPS results table. Read it directly before the project claims any placement.
- Because the project's 0.904 is on a 2-class balanced-ish split, a likely sensible claim is "in the range of published Indic binary abuse results". I could not verify the exact range.
- AbuseXLMR is the natural stronger baseline the project should cite or run.

### Gaps
- The MACD per-language F1 table, and any 2024-2026 paper reporting MACD Telugu, were not found.

## Is XLM-R + LoRA standard or modern?

### Takeaway
LoRA is already routine in the Telugu hate speech papers I found, and XLM-R is a common generic multilingual baseline in this subfield. So the project is mainstream, not novel.

### Cited Findings
- Kakarla and Venkata fine-tune all 7 transformers with LoRA for Telugu hate speech (Feb 2025). — [arXiv](https://arxiv.org/abs/2502.10632)
- The survey lists XLM-R among the widely used transformers for Indian-language hate speech. — [arXiv 2411.19017](https://arxiv.org/pdf/2411.19017)
- In the sources I read, MuRIL and Indic-specialized models, such as Hindi-Abusive-MuRIL and AbuseXLMR, are reported as stronger than generic XLM-R. — [arXiv 2502.10632](https://arxiv.org/abs/2502.10632), [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a7c4163b33286261b24c72fd3d1707c9-Abstract.html)

### Inferences
- XLM-R-base + LoRA is a reasonable but generic baseline. Reviewers would expect a MuRIL or AbuseXLMR comparison, and possibly an LLM-prompting comparison.

### Gaps
- I did not find any survey that explicitly labels XLM-R+LoRA as "standard".

## TF-IDF baselines matching transformers

### Takeaway
I found no source confirming that TF-IDF+SVM matches transformers on Telugu. Related evidence points the other way.

### Cited Findings
- At LT-EDI 2025, on Tamil caste and migration hate speech, TF-IDF with SVM or LR was tested against five transformers. The domain-tuned Tamilhate-BERT won with macro-F1 0.88, and the TF-IDF scores were not shown in the snippet. — [LT-EDI 2025 paper](https://aclanthology.org/2025.ltedi-1.18.pdf)
- A homophobia/transphobia entry (SKVtrio) found that TF-IDF combined with BERT embeddings beat either alone across 8 Indian languages. — [ACL Anthology](https://aclanthology.org/people/konkimalla-laxmi-vignesh/)
- The DravidianLangTech 2026 Telugu paper combines TF-IDF and FastText with transformers, which suggests classical features remain in use. — [ACL Anthology](https://aclanthology.org/2026.dravidianlangtech-1.1/)

### Inferences
- The project's finding that TF-IDF+LinearSVC is as accurate on MACD Telugu is plausible for a short, lexically driven binary task, but I found no cited precedent. It would be framed as the project's own empirical result, probably tied to dataset noise or lexical cues.

### Gaps
- No sourced head-to-head showing TF-IDF parity on Telugu was found.
- I found nothing on LoRA vs full fine-tune parity, calibration, INT8 ONNX, or obfuscation guards in Indic abuse work.
