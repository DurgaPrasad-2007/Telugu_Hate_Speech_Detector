# MACD (Gupta et al., NeurIPS 2022 D&B) and AbuseXLMR: Telugu results and dataset notes

Primary source: Gupta et al., "Multilingual Abusive Comment Detection at Scale for Indic Languages", NeurIPS 2022 Datasets and Benchmarks.
- Full PDF: https://papers.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Paper-Datasets_and_Benchmarks.pdf
- Abstract page: https://proceedings.neurips.cc/paper_files/paper/2022/hash/a7c4163b33286261b24c72fd3d1707c9-Abstract.html
- Code and data: https://github.com/ShareChatAI/MACD

Method note: I read the PDF through pdftotext, which lost row labels in some tables. Where a row-to-model mapping is inferred, it is marked. The PDF page images could not be rendered in this environment, so table alignment is verified only against the paper's prose.

## 1. Dataset size, label mapping, splits

### Takeaway
MACD is about 152K comments (Table 2) with a 49% abusive rate. Telugu is about 30K comments at 52% abusive (Table 1). The paper's labels are binary abusive / non-abusive. The GitHub README maps 0 = abusive and 1 = non-abusive, which is the reverse of the common convention, so check this against the CSV before training.

### Cited Findings
- Total 152,422 samples; 74,550 abusive; 77,872 non-abusive; 49% abuse; 92,881 posts; 70,453 users (Table 2) — https://papers.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Paper-Datasets_and_Benchmarks.pdf
- Telugu: "MACD Telugu 30K 52%" (Table 1, same PDF). Table 1 gives the Telugu count only as "30K", so the exact count is not stated.
- Per-language abuse ratios: Hindi 52%, Tamil 46%, Telugu 52%, Malayalam 45%, Kannada 49% (Table 1 and Section 3.3 text, same PDF).
- Labels: "binary labels (abusive and non-abusive) representing the presence/absence of abuse" (Section 3.2, same PDF). The paper does not give a numeric encoding in the text.
- Repo README: "labels 0 and 1 are used for abusive and non-abusive comments, respectively." Data folders: `dataset/` (train/val/test CSV), `dataset_80_10_10/`, `dataset_chrono/`, `dataset_meta/`, `dataset_unbalanced/` (https://github.com/ShareChatAI/MACD). Column names are not documented in the README.
- Default split: "We randomly split MACD in 60:20:20 ratio to form the train, validation and test set" (Section 3.3, "Dataset splits"). Also 80:10:10 random, chronological 60:20:20, and unbalanced 1:5 abusive:non-abusive (same PDF).
- Per-language split sizes for Telugu are not given in the paper text.

### Inferences
- Telugu's 30K figure is approximate. With the 60:20:20 default, the Telugu test set is about 6K comments, but this is an estimate from "30K", not a reported number.
- The label direction needs checking: the README's 0 = abusive mapping and the paper's abusive-first ordering in the prose may not match.

### Gaps
- Exact Telugu sample count, Telugu per-split counts, and the column names of the CSV are not in the paper or README. Verify by loading the CSV.

## 2. Telugu results: XLM-R, MuRIL, mBERT, AbuseXLMR, baselines

### Takeaway
In the paper's random 60:20:20 monolingual setup, AbuseXLMR's Telugu macro-F1 is 89.44 (Table 3). The paper's best labelled Telugu macro-F1 is 90.00 for XLM-R joint training (Table 7). The paper never reports a Telugu macro-F1 of 0.904. Its closest figures are 90.00 and 89.44. The paper's numbers are in percent, so the project's ~0.904 is on a different scale or a different setup.

### Cited Findings (Table 3, monolingual, random 60:20:20; values are Telugu Acc / macro-F1 in %)
- mBERT: 86.10 / 86.08 (Table 3, same PDF)
- XLM-R: 86.52 / 86.49 (Table 3). Note: the paper's text says XLM-R is best for Telugu among the contextual baselines, which conflicts with the numbers below (see caveat).
- MuRIL: 89.08 / 89.07 (Table 3)
- AbuseXLMR: 89.50 / 89.44 (Table 3). The paper says AbuseXLMR "outperforms all the other models for all the languages" (Section 4).
- Two unlabelled rows in Table 3's Telugu column: 89.47 / 89.42 and 91.40 / 91.37. The row labels were lost in extraction, so I cannot assign them. The TF-IDF (LR) and TF-IDF (SVM) rows have no Telugu numbers in the extracted text.

### Caveats
- Table 3 and Table 7 conflict. Table 7 labels 89.08 / 89.07 as "XLM-R (mono)" for Telugu, but Table 3 gives that same value to MuRIL. Table 7's "XLM-R (pr-mono)" (89.50 / 89.44) equals Table 3's AbuseXLMR row, which is consistent with the paper's description of AbuseXLMR as continued XLM-R pretraining. So the Table 7 label is probably right for the mono and pr-mono rows, and Table 3's MuRIL/XLM-R labels may be swapped. Unverified: check the image of Table 3 in the PDF.
- The text says "XLM-R shows best performance for Hindi, Telugu and Malayalam" among mBERT, XLM-R and MuRIL (Section 4, same PDF). Table 3 as extracted shows MuRIL above XLM-R for Telugu, so the table and the text disagree in some reading.

### Other Telugu tables (lower confidence)
- Table 7 (mono / joint / pretrain), Telugu macro-F1: XLM-R (mono) 89.07; XLM-R (joint, all five languages' training splits combined) 90.00 (Accuracy 90.05); XLM-R (pr-mono) 89.44. The paper says joint training helps mBERT and XLM-R. Same PDF, Section 4 "Joint training experiments" and "Pretraining experiments".
- Table 4 (AbuseXLMR, Telugu column, Acc / macro-F1): 91.63 / 91.60, 90.42 / 90.38, 94.65 / 88.86. The row labels were lost. Inferred mapping: 80:10:10 random = 91.60, chronological = 90.38, unbalanced = 88.86. This rests on the paper's text that F1 drops on chronological and unbalanced splits, and accuracy rises on unbalanced. Unverified.
- Table 6 (few-shot, XLM-R and MuRIL, Telugu row): values for 5/25/100/250/500 samples are listed, but the row-to-model assignment is ambiguous in extraction. Do not cite without checking the PDF image.
- Table 5 (zero-shot cross-lingual): the Telugu rows are unreliable after extraction. Not used here.

### Comparison with ~0.904 (user's figure)
- No labelled Telugu macro-F1 in the paper equals 0.904. The nearest are 0.9000 (XLM-R joint, Table 7) and 0.8944 (AbuseXLMR, Table 3). The unlabelled 0.9137 in Table 3 may be a higher model, but it is unverified.
- Source of ~0.904 in this repo: `E:/Telugu_Hate_Speech_Detector/.agents/teamwork/explorer_m0_arch_3/handoff.md` line 368 lists an "XLM-RoBERTa-base + LoRA" baseline with a value of 0.904 and another of 0.612. The metric is not named in that line, and it is not from the MACD paper. Treat it as the project's own unverified result.
- If the project's 0.904 is macro-F1 on the MACD Telugu test split, it sits roughly 0.5 to 1 point above the paper's AbuseXLMR (0.8944) and 0.4 points above the XLM-R joint result (0.9000). Such a comparison needs the same split and the same metric, so it is not valid without checking.

### Inferences
- AbuseXLMR has the best labelled Telugu macro-F1 (89.44) among the paper's monolingual baselines, but the gap to MuRIL (89.07) is about 0.4 points on one split with one seed. The paper reports no variance or seed information in the text I extracted, so the gap is within plausible noise.
- XLM-R joint training across languages (90.00) is slightly above all monolingual Telugu rows, which suggests cross-language signal helps in Telugu.

### Gaps
- Baselines: no TF-IDF Telugu value could be assigned to a row. No fastText result found.
- Seeds, error bars and hyperparameters: not extracted from the main text. The appendix (Section A.5) was not read because it is not in the extracted text.

## 3. Duplicates, noisy labels, leakage

### Takeaway
No source I found reports duplicates or train/test leakage in MACD. The paper says nothing about deduplication. The paper does report annotation quality (agreement coefficient 0.69 to 0.73 across languages; the symbol was lost in extraction and is most likely kappa, unverified). A 2026 follow-up reports label noise, but on a different dataset (the Moj/MACI challenge data), not on MACD.

### Cited Findings
- Annotator agreement: "observe kappa = 0.73, = 0.72, = 0.71, = 0.69, and = 0.70 for Hindi, Tamil, Telugu, Malayalam, and Kannada" (Section 3.2, the symbols were lost in extraction but the values read as kappa). Telugu kappa = 0.71. Same PDF.
- Annotation process: 53 annotators; two native speakers per language; disagreements resolved by a senior annotator (Section 3.2, same PDF).
- Balanced sampling: "After annotation, we sampled comments across categories and languages for creating a well-balanced and diverse dataset" (Section 3.2, same PDF). This sampling may affect the joint distribution of labels and topics across splits.
- User bias: "we threshold the number of comments" per user (Section 3.1). The paper says this mitigates user bias but gives no threshold value in the extracted text.
- Follow-up with label-noise evidence, but on a different dataset: Rastogi et al., "Multi-Stage Training for Abusive Comment Detection in Indic Languages" (https://arxiv.org/pdf/2605.22380). It describes the Moj app challenge data (700,000 comments, 15+ languages). It estimates 9-10% annotation error in its training set, and says flipping labels "got 97% F1 on CV but 88-89% on the leaderboard". This paper does not use MACD, so do not apply its noise estimate to MACD.
- Repo README: no statement on duplicates, cleaning or licensing (https://github.com/ShareChatAI/MACD).
- Supplemental PDF (https://papers.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Supplemental-Datasets_and_Benchmarks.pdf): the fetch returned compressed binary, so its content could not be read. It may contain data-prep details.

### Inferences
- The ~0.90 range reported across models (Section 2) is close to the kappa-implied noise ceiling. This is only a rough check, not a measured ceiling.
- Without a dedup step documented, near-duplicate comments across splits cannot be ruled out. Check with an exact and near-duplicate scan of the project's CSV before comparing numbers.

### Gaps
- Whether dedup was done: not stated anywhere I could read.
- Supplemental PDF content: unreadable from this environment.

## 4. AbuseXLMR

### Takeaway
AbuseXLMR is XLM-R continued-pretrained with masked language modelling on 5M social-media comments from ShareChat (Section 3.4, same PDF). The paper reports that it beats XLM-R and MuRIL on several Indic datasets. I found no independent replication and no model card on the Hugging Face hub in my searches.

### Cited Findings
- "we randomly sample 5M comments out of the complete corpora and use these sampled comments for continued pretraining of the XLM-R model using masked language modelling" (same PDF, pretraining section).
- Table 9 (cross-dataset, accuracy / macro-F1): MOLD Marathi, AbuseXLMR 90.72 / 89.50; Bengali 90.78 / 89.73; HASOC Hindi 84.98 / 84.75. XLM-R on the same datasets: 88.96 / 87.70; 90.47 / 89.32; 82.85 / 82.39. MuRIL: 89.60 / 88.38; 90.63 / 89.59; 84.45 / 84.22. The paper says AbuseXLMR outperforms both MuRIL and XLM-R on all three (Section 4).
- Table 8: MACD-pretraining followed by HASOC fine-tuning improves XLM-R's macro-F1 from 82.82 (baseline) to 83.99 (Pretrained-MACD). Same PDF.
- Pretraining corpus: 5M is the sample size; the paper also says the "pretraining corpus" is "15+ Indic languages" (same PDF).
- Model release: the paper says a model is released. The repo README links to a pretrained AbuseXLMR model (https://github.com/ShareChatAI/MACD). I did not verify the link target.

### Inferences
- The gain of AbuseXLMR over MuRIL on MACD Telugu is small (about 0.4 macro-F1 points). The paper's claim is stronger on Hindi and Malayalam than on Telugu.

### Gaps
- Hugging Face model card, licence and exact checkpoint name: not verified.
- Independent replication: none found.

## Summary of unverified or conflicting items
1. Table 3 row labels for XLM-R vs MuRIL conflict with Table 7 and the paper's text. Not resolved from extracted text.
2. Table 4 and Table 6 row assignments: inferred or ambiguous.
3. Label convention: the paper does not give a numeric mapping; the README says 0 = abusive. Verify on the CSV.
4. The project's 0.904 is not from the paper. Its metric and split are unknown.
5. No source reports duplicates or leakage in MACD.

## Sources consulted
- https://papers.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Paper-Datasets_and_Benchmarks.pdf (primary; read via pdftotext)
- https://proceedings.neurips.cc/paper_files/paper/2022/hash/a7c4163b33286261b24c72fd3d1707c9-Abstract.html
- https://github.com/ShareChatAI/MACD (README via WebFetch)
- https://arxiv.org/pdf/2605.22380 (different dataset: Moj/MACI challenge)
- https://papers.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Supplemental-Datasets_and_Benchmarks.pdf (unreadable)
