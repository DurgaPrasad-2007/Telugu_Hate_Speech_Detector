# Telugu / Telugu-English Abusive and Hate Speech Detection: 2025-2026 Papers

Scope: 2024 to October 2026. Verification status is marked per item. "Fetched" means read through a web-fetch tool that summarises HTML or PDF; these figures should be checked against the PDF before they are cited in a paper. The arXiv PDF endpoint returned binary content in this session, so the HTML version was used where available. Nothing here was checked against the PDF itself.

## KQ1: Model tables, backbone winner, LoRA, TF-IDF comparison

### Takeaway
Only one of the three target papers reports transformer F1/accuracy tables (Kakarla & Venkata, arXiv 2502.10632), and it uses LoRA. Its winner is Hindi-Abusive-MuRIL, with test accuracy 0.7314 on the original Telugu data. Pandey et al. (arXiv 2504.21026) use classical ML and a fine-tuned LLM on a 2,000-comment Telugu-English set, and their text and Table 5 contradict each other. The Reddy & Bellamkonda 2026 paper (ACL 2026.dravidianlangtech-1.1) gives no numbers in any source I could reach.

### Cited Findings

**A. Kakarla & Venkata, arXiv 2502.10632, "Code-Mixed Telugu-English Hate Speech Detection" (submitted 15 Feb 2025)**
- Abstract: transformers fine-tuned with LoRA; Telugu text translated to English via Google Translate; models compared on both. Models: TeluguHateBERT, HateBERT, DeBERTa, MuRIL, IndicBERT, RoBERTa, Hindi-Abusive-MuRIL. Winner: "Hindi-Abusive-MuRIL outperforms all other models" on both original and translated data. [Source](https://arxiv.org/abs/2502.10632); [Source](https://arxiv.org/html/2502.10632v1) (fetched)
- Table 1 (fetched; caption printed as "Performance of Transformer Models on Translated Telugu Hate Speech Dataset", which appears to be a copy error. Text attributes this table to the original Telugu data). Columns: Test Accuracy / F1 weighted / F1 macro:
  - RoBERTa 0.6773 / 0.6772 / 0.6772
  - DeBERTa 0.4268 / 0.4148 / 0.4150
  - MuRIL 0.4228 / 0.4114 / 0.4115
  - TeluguHateBERT 0.4388 / 0.4250 / 0.4251
  - IndicBERT 0.3687 / 0.3632 / 0.3633
  - HateBERT 0.4509 / 0.4412 / 0.4414
  - Hindi-Abusive-MuRIL 0.7314 / 0.7314 / 0.7314 (best)
  [Source](https://arxiv.org/html/2502.10632v1) (fetched)
- Table 2 (fetched; same caption, translated data per text):
  - RoBERTa 0.5050 / 0.3823 / 0.3829
  - DeBERTa 0.6032 / 0.5657 / 0.5654
  - MuRIL 0.5010 / 0.3344 / 0.3338
  - TeluguHateBERT 0.5010 / 0.3344 / 0.3338
  - IndicBERT 0.5010 / 0.3344 / 0.3338
  - HateBERT 0.4669 / 0.4665 / 0.4665
  - Hindi-Abusive-MuRIL 0.6993 / 0.6990 / 0.6990 (best)
  [Source](https://arxiv.org/html/2502.10632v1) (fetched)
- Reported conclusion: translation raised accuracy for most models but lowered F1, which the authors attribute to lost linguistic nuance. [Source](https://arxiv.org/html/2502.10632v1) (fetched)
- Hyperparameters (fetched): LoRA with AdamW, learning rate 2e-5, batch size 32, 25 epochs. LoRA rank, alpha and dropout are not reported. [Source](https://arxiv.org/html/2502.10632v1) (fetched)
- LoRA: yes. [Source](https://arxiv.org/abs/2502.10632) (abstract)
- TF-IDF or classical baseline: none reported. (fetched)
- Precision and recall: not reported. (fetched)

**B. Pandey, Yadav, Adduru & Rai, arXiv 2504.21026, "Creating and Evaluating Code-Mixed Nepali-English and Telugu-English Datasets for Abusive Language Detection..." (submitted 23 Apr 2025)**
- Models (abstract and fetched body): Logistic Regression, SVM, Random Forest, Neural Network, CNN, LSTM, and an unnamed "LLM". Per the fetched summary, the classical models and CNN/LSTM take 768-dimensional mBERT embeddings as input. [Source](https://arxiv.org/abs/2504.21026); [Source](https://arxiv.org/html/2504.21026v1) (fetched)
- LoRA: not mentioned. The LLM is described as fine-tuned at learning rate 3e-5 for 3 epochs (Telugu) with no named backbone. (fetched)
- TF-IDF: not mentioned as a baseline. (fetched)
- Table 5, "Performance Comparison of Models on Nepali-English and Telugu-English Datasets" (fetched from HTML and reproduced by the fetch tool; values not checked against the PDF). Telugu-English rows, columns Precision / Recall / F1 / Accuracy:
  - Logistic Regression 0.531 / 0.523 / 0.525 / 0.523
  - SVM 0.548 / 0.584 / 0.480 / 0.584
  - Random Forest 0.534 / 0.583 / 0.477 / 0.583
  - Neural Network 0.530 / 0.542 / 0.505 / 0.542
  - CNN 0.522 / 0.524 / 0.483 / 0.524
  - LSTM 0.535 / 0.548 / 0.537 / 0.548
  - LLM 0.910 / 0.907 / 0.907 / 0.907
  [Source](https://arxiv.org/html/2504.21026v1) (fetched)
- Text, Telugu-English results section 6.2 (fetched; quoted figures are as extracted):
  - Logistic Regression: "weighted F1-score of 0.8374 and 83.75% accuracy"
  - SVM: "83.25% accuracy and a 82.86% weighted F1-score"
  - Random Forest: "accuracy of 77.50% and a weighted F1-score 76.31%"
  - Neural Network: "weighted F1-score of 0.8795 and accuracy of 84.25%"
  - CNN: "accuracy of 84.00% and a weighted F1-score of 84.05%"
  - LSTM: "accuracy of 84.25% and a weighted F1-score of 84.09%"
  - LLM: "the highest accuracy (80.50%) was obtained"
  [Source](https://arxiv.org/html/2504.21026v1) (fetched)
- **Conflict:** the text values (about 0.77 to 0.84 accuracy, LLM 80.50%) do not match Table 5 (about 0.48 to 0.54 for classical models, LLM 0.907). The paper does not explain the gap. The table-vs-text conflict is the first thing to check in the PDF. Until it is checked, neither set of numbers should be cited as the result.
- Inference (mine): the Telugu-English set is 60% abusive (see KQ2). An all-abusive predictor gets accuracy 0.60, so Table 5 classical accuracies of 0.52 to 0.58 are at or below the majority baseline. This is consistent with Table 5 and not with the text.
- Nepali-English Table 5 rows (fetched; reference only): LR 0.720, SVM 0.730, RF 0.700, NN 0.740, CNN 0.735, LSTM 0.738, LLM 0.790 (precision/recall/accuracy columns given, F1 0.786 for LLM).

**C. Bojja Revanth Reddy & Sivaiah Bellamkonda, "Abusive Content Detection in Telugu-English Code-Mixed Social Media Using Hybrid Transformer Architectures", ACL DravidianLangTech 2026, aclanthology.org/2026.dravidianlangtech-1.1**
- Confirmed in the ACL volume listing. [Source](https://aclanthology.org/volumes/2026.dravidianlangtech-1/) (fetched)
- Abstract (fetched): transliteration and spelling normalisation; hybrid features combining TF-IDF and FastText embeddings; fine-tuning of multilingual transformers. Compares traditional ML and transformer models. Stated challenges: morphological complexity, contextual ambiguity, limited annotated data. [Source](https://aclanthology.org/2026.dravidianlangtech-1.1/) (fetched)
- Not found on the page: dataset name and size, labels, backbone names, scores, TF-IDF-only baseline, LoRA. The PDF was not retrievable in this session.
- Note: the search snippet also associated Bojja Revanth Reddy with a 2025 DravidianLangTech paper on abusive content against women. That paper is not about Telugu; I did not verify it.

### Inferences
- Among the sources I could read, the only transformer-vs-transformer comparison with LoRA is Kakarla & Venkata. Its Table 1 spread (0.37 to 0.73 accuracy) is wide and some rows sit near chance for a binary task. Several Table 2 rows are identical (MuRIL, TeluguHateBERT, IndicBERT all 0.5010 / 0.3344 / 0.3338), which looks like majority-class collapse. Treat the winner as weakly supported until seeds and variance are reported.
- No 2025-2026 Telugu paper I could read reports a TF-IDF baseline cleanly. Reddy & Bellamkonda use TF-IDF as a feature, not as a comparison baseline.

### Gaps
- Reddy & Bellamkonda 2026: no scores or table text reachable. Next step: fetch https://aclanthology.org/2026.dravidianlangtech-1.1.pdf directly.
- Pandey et al.: the arXiv PDF returned binary; the Table 5 vs text conflict needs the PDF or an extracted text to resolve.
- Kakarla & Venkata: dataset size, source, and LoRA rank/alpha are missing from the HTML version.

## KQ2: Dataset sizes and sources; comparability to a pure-Telugu binary task

### Takeaway
Known sizes are small. Pandey et al. has 2,000 Telugu-English comments (60% abusive), drawn from Twitter, YouTube and Instagram. Kakarla & Venkata's dataset size and source are not stated in the HTML. Neither can be cleanly compared to a pure-Telugu binary task without the source and split sizes.

### Cited Findings
- Pandey et al., Telugu-English: 2,000 comments, 1,200 abusive (60%) and 800 non-abusive; 18,551 tokens; mean 9.28 tokens per comment; sources Twitter, YouTube, Instagram; annotators fluent in English and Telugu; inter-annotator agreement not reported. [Source](https://arxiv.org/html/2504.21026v1) (fetched)
- Pandey et al. internal inconsistency: the abstract says "5 Nepali-English" comments; the body says 5,000 for Nepali and Table 2 says 2,500. Telugu count is consistent at 2,000. [Source](https://arxiv.org/html/2504.21026v1) (fetched)
- Kakarla & Venkata: binary "hate" / "non-hate" Telugu text. Nearly balanced training set; even test split. Dataset name, counts, source and collection method are not stated; Figure 1 gives distribution without numbers. [Source](https://arxiv.org/html/2502.10632v1) (fetched). Note the title says "Code-Mixed", but the body describes "Telugu text". I did not resolve whether the data is code-mixed or script-pure.
- Reddy & Bellamkonda 2026: dataset not described in the reachable abstract. [Source](https://aclanthology.org/2026.dravidianlangtech-1.1/) (fetched)
- HOLD-Telugu 2024 shared task (DravidianLangTech 2024, shared-task overview): dataset of YouTube comments, manually annotated; 23 teams; ranked by macro F1. Dataset size and top score not retrieved. [Source](https://www.amrita.edu/publication/findings-of-shared-task-on-hate-offensive-language-detection-in-telugu-codemixed-text-hold-telugu-dldravidianlangtech-2024/) (fetched)
- DravidianLangTech@NAACL 2025 multimodal hate speech overview (Malayalam, Tamil, Telugu; text plus audio). Search snippet only: best Telugu F1 0.3817 (Team lowes). Not verified against the overview. [Source](https://preview.aclanthology.org/setup/2025.dravidianlangtech-1.20) (search snippet; page returned 404 on fetch)
- KEC_AI_ZEROWATTS (same 2025 task) search snippet: Telugu text model 0.89, audio model 0.87, rank 8 of 145 teams. This conflicts with the best Telugu F1 of 0.3817 in the same snippet set and is unverified; the metric may differ. [Source](https://preview.aclanthology.org/setup/2025.dravidianlangtech-1.20) (search snippet)

### Inferences
- Comparability to a pure-Telugu binary task: Kakarla & Venkata is binary but Telugu-only in the body, with no source. Pandey et al. is binary and code-mixed, with a 60/40 split. The HOLD-Telugu and 2025 multimodal tasks are not clearly binary. Scores are not comparable across these tasks because the metrics differ (macro F1, weighted F1, accuracy, and in one case a multimodal setup).
- At 2,000 comments with a 60% majority, the Pandey et al. Telugu-English task is small and imbalanced. Accuracy is a weak metric for it; macro F1 is needed.

### Gaps
- Kakarla & Venkata split sizes and source.
- HOLD-Telugu dataset size and top-team score.
- 2025 multimodal Telugu sub-task dataset size and the 0.89 vs 0.3817 discrepancy.

## Other 2025-2026 items checked

- **FLAICOL (Mohammed & Kamakshi), ACL 2026.dravidianlangtech-1.4.** Flip-point-guided minority-class augmentation for code-mixed offensive language. Per the search snippet, experiments use Tamil-English, Malayalam-English, and Kannada-English splits, not Telugu. Not a Telugu paper for this purpose. [Source](https://aclanthology.org/2026.dravidianlangtech-1.4/) (search snippet, not fetched)
- **2026 DravidianLangTech volume, Telugu entries.** Most Telugu-titled papers are prompt-style recovery or dialogue tasks, not abuse detection (e.g. 1.5 LIMP, 1.14, 1.25, 1.37, 1.42, 1.45, 1.46, 1.58). The Tamil abuse-against-women track is a separate shared task. [Source](https://aclanthology.org/volumes/2026.dravidianlangtech-1/) (fetched)
- **HASOC 2025.** No Telugu track found. The HASOC 2025 languages reported in the search snippets were Hindi, Gujarati, Bangla and Bodo. [Source](https://hasocfire.github.io/hasoc/2025/) (search snippet; site not fetched)
- **arXiv cs.CL search.** Only the two arXiv papers above were found for 2025 Telugu abuse or hate. The search was limited to one query, so this is not exhaustive.

## Unverified items
- Pandey et al. Table 5 values and the text values: both came from a fetch-tool extraction, and the PDF was not readable. Verify before citing.
- Kakarla & Venkata Table 1 and Table 2: from fetch-tool extraction of the HTML; the table caption duplication is confirmed in the extraction. Verify against the PDF.
- All search-snippet numbers (2025 multimodal Telugu F1 0.3817 and 0.89/0.87) are unverified.
- Reddy & Bellamkonda 2026 scores: not found.

## Next steps
1. Get the PDF text of arXiv 2504.21026 to settle the Table 5 vs text conflict.
2. Get https://aclanthology.org/2026.dravidianlangtech-1.1.pdf to obtain the Reddy & Bellamkonda tables and dataset.
3. Get the Kakarla & Venkata PDF for dataset size, source, and LoRA rank.
4. Fetch the 2025 multimodal overview PDF (DOI 10.18653/v1/2025.dravidianlangtech-1.20) for Telugu sub-task numbers.
