# TF-IDF + Linear SVM Parity with Transformers on Hate/Abuse Detection, and Benchmark Leakage

Scope: 2020 to Oct 2026, with emphasis on recent work. Verification tags used below:
- [FETCHED] full text or abstract page read in this session
- [SNIPPET] only a search-result snippet was seen; numbers not checked against the paper
- [UNVERIFIED] mentioned from memory or secondhand; no source read
- Tool note: most PDF fetches returned raw binary, so most claims are [SNIPPET]. Treat exact numbers as provisional until the PDFs are opened.

## Q1. How often do papers find TF-IDF/linear SVM parity with transformers, and under what conditions?

### Takeaway
There is no systematic count, and I did not build one. Across about 12 comparative studies reviewed, TF-IDF + linear models were competitive or won in at least one setting in roughly a third of them. Transformers led in most, usually on the minority hate class or on macro-F1. Parity shows up most often on small or noisy datasets, short and lexically explicit text, and where transformer fine-tuning is unstable.

### Cited Findings
- HASOC 2019 (SVM vs BERT, English): BERT clearly ahead on sub-task 1 but not on sub-task 3, where the SVM was competitive. [SNIPPET] — [KMI-Panlingua at HASOC 2019](https://ceur-ws.org/Vol-2517/T3-14.pdf)
- Comparative hate-speech study (English): TF-IDF SVM macro-F1 0.64, "comparable to some transformer models"; transformers were best, especially on the minority hate class. [SNIPPET] — [arXiv 2202.09517](https://arxiv.org/pdf/2202.09517)
- FIRE 2020 BERT paper (Vol-2826): BERT 88.33 vs TF-IDF SVM 81.56 macro-F1 in one setting, so transformer ahead. [SNIPPET] — [CEUR Vol-2826 T2-11](https://ceur-ws.org/Vol-2826/T2-11.pdf)
- Indonesian product-review sentiment benchmark (preprint, May 2026; non-hate, Austronesian, low-resource): Linear SVC on TF-IDF reached 97.60% accuracy and 0.5510 macro-F1; fine-tuned IndoBERT reached 88.70% accuracy and 0.5088 macro-F1. The authors attribute the gap mainly to IndoBERT being trained on a sampled subset while the baselines used the full corpus, so this is a confounded comparison. Macro-F1 near 0.55 next to 97.6% accuracy also suggests heavy class imbalance. [FETCHED] — [arXiv 2605.03439](https://arxiv.org/abs/2605.03439)
- A 38-model hate-speech evaluation (arXiv 2509.14266, English, Sep 2025) found traditional methods such as SVM competitive on some data, with RoBERTa best on small sets. [UNVERIFIED — taken from the sibling note `modern_techniques.md`; not re-checked here] — [arXiv 2509.14266](https://arxiv.org/pdf/2509.14266)
- Simple-baseline caveat for TF-IDF: the NAACL 2022 "Features or Spurious Artifacts?" paper reports that simple data-centric baselines are "surprisingly effective," but its PDF was not readable, so I have no numbers. [FETCHED abstract] — [aclanthology 2022.naacl-main.221](https://aclanthology.org/2022.naacl-main.221/)
- Non-hate-speech evidence on small labelled sets: a 2025 blog benchmark (long documents, not hate speech) (about 100–150 labeled samples) found LR on TF-IDF at 79 F1 against 82 for BERT-base. Blog, not peer-reviewed. [SNIPPET] — [procycons benchmark 2025](https://procycons.com/en/blogs/long-document-classification-benchmark-2025/)

### Inferences
- The conditions where parity appears most often are: small training sets (hundreds to a few thousand items), explicit slur or profanity-driven labels, short text, and minority-class F1 being low for all models. In these cases a linear model with word and char n-grams can match a transformer that fits the same lexical cues.
- Parity claims are easy to get when the transformer baseline is weak. The Telugu Kakarla paper (see Q2) shows several fine-tuned transformers near chance, which would make any linear baseline look competitive.
- I did not find a study that sweeps training-set size systematically on Indic data. This is the cleanest way to test the parity condition and is a gap in the literature I found.

### Gaps
- No systematic meta-count of parity results; the fraction above is a convenience sample of about 12 studies, not a survey.
- Many sources are conference or shared-task papers whose PDFs I could not open, so the numbers come from snippets.

---

## Q2. Telugu and Dravidian examples (also Tamil, Malayalam, Kannada)

### Takeaway
Telugu has no rigorous head-to-head showing TF-IDF + SVM parity with transformers that I could verify. The shared-task evidence is mixed: linear TF-IDF models are often competitive or top-of-team, but transformers lead the leaderboards. For Tamil, one 2021 team found TF-IDF char n-grams + logistic regression best. The most useful Telugu result I found, Kakarla & Venkata 2025, reports transformer fine-tuning numbers that look unstable.

### Cited Findings
- Telugu, HOLD-Telugu (DravidianLangTech, ACL 2024.dravidianlangtech-1.8): 23 teams. Team MUCS reported an LR model on TF-IDF of subword, word and char n-grams as its best model, with macro-F1 0.6501 (15th rank). [SNIPPET] — [ACL 2024.dravidianlangtech-1.8](https://aclanthology.org/2024.dravidianlangtech-1.8.pdf); team paper [MUCS, snippet] — see the findings paper for the rest
- Telugu, DravidianLangTech 2023 (Tamil and Telugu): SVM, LR and LinearSVC on TF-IDF; macro and weighted F1 "significantly better" on the Telugu data than on Tamil and Tamil-English. [SNIPPET] — [ACL 2023.dravidianlangtech-1.32](https://aclanthology.org/2023.dravidianlangtech-1.32.pdf)
- Telugu-English aggression corpus (3,361 annotated tweets): baseline RBF-kernel SVM F1 0.67. The snippet does not confirm the feature set. [SNIPPET] — [academia.edu record](https://www.academia.edu/76916564/Corpus_Creation_and_Baseline_System_for_Aggression_Detection_in_Telugu_English_Code_Mixed_Social_Media_Data)
- Telugu-English abuse (Pandey et al., arXiv 2504.21026, Apr 2025): about 2,000 Telugu-English comments; compares LR, RF, SVM, NN, LSTM, CNN and LLMs with 10-fold CV and t-tests. No F1 values were visible. [SNIPPET] — [arXiv 2504.21026](https://arxiv.org/abs/2504.21026)
- Telugu, Sandalphon (DravidianLangTech-EACL 2024): compares classical ML, DL and transformers; Telugu-BERT best at F1 0.77 on the augmented set. [SNIPPET; no URL recovered — gap]
- Telugu, Kakarla & Venkata (arXiv 2502.10632, Feb 2025), binary hate/non-hate, fine-tuned with LoRA. The paper has **no classical baseline** (SVM, LR, NB and TF-IDF are only in related work). [FETCHED arXiv HTML] — [arXiv 2502.10632](https://arxiv.org/html/2502.10632v1)
  - Original Telugu, macro-F1: Hindi-Abusive-MuRIL 0.7314; RoBERTa 0.6772; HateBERT 0.4414; TeluguHateBERT 0.4251; DeBERTa 0.4150; MuRIL 0.4115; IndicBERT 0.3633. Accuracy for IndicBERT is 0.3687, below the 0.5 chance rate for a balanced binary task.
  - Translated (English), table 2: three models (MuRIL, TeluguHateBERT, IndicBERT) all show identical accuracy 0.5010 and identical weighted and macro F1 near 0.334, which is the signature of majority-class collapse.
  - The table captions say "Translated" for both tables, which is ambiguous. I read them as original and translated respectively, based on the text.
- Telugu, MACD / ShareChat (NeurIPS 2022 Datasets and Benchmarks): 150K comments on 92,881 posts by 70,453 users, multilingual. Telugu subset: 30K samples, 52% abuse. Code at ShareChatAI/MACD. [SNIPPET] — [NeurIPS 2022 datasets paper](https://papers.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Paper-Datasets_and_Benchmarks.pdf). Note: the name is "Multilingual Abusive Comment Detection," not "multi-aspect."
- Tamil, DravidianLangTech 2021 offensive-language team paper: TF-IDF char n-grams + logistic regression best for Tamil; MuRIL embeddings + random forest best for Malayalam. The paper's Malayalam text and tables conflict, so treat numbers as unverified. [SNIPPET] — [ACL 2021.dravidianlangtech-1.37](https://aclanthology.org/2021.dravidianlangtech-1.37.pdf)
- Tamil/Malayalam/Kannada 2021 shared-task winners were transformer ensembles: SJ_AJ used mBERT + XLM-R ensemble, ranked 1st on Kannada, 2nd Malayalam, 3rd Tamil. indicnlp@kgp weighted F1 0.97 Malayalam, 0.77 Tamil, 0.72 Kannada (model type not confirmed in the snippet). [SNIPPET] — [arXiv 2102.01051](https://arxiv.org/abs/2102.01051), [arXiv 2102.07150](https://arxiv.org/abs/2102.07150). The organizers' findings paper is at [ACL 2021.dravidianlangtech-1.17](https://aclanthology.org/2021.dravidianlangtech-1.17/); I did not open it.
- Tamil, LT-EDI 2025 (caste/migration hate): TF-IDF with SVM and LR compared against five transformers; the domain-tuned Tamil hate BERT was best at macro-F1 0.88. [SNIPPET] — [LT-EDI 2025 paper](https://aclanthology.org/2025.ltedi-1.18.pdf)
- Hindi/Bengali/Tamil/English personal project (GitHub, not peer-reviewed): TF-IDF LR (word 1–2 + char 2–5 grams) vs fine-tuned MuRIL. Mean F1 0.794 vs 0.801, with TF-IDF ahead on English and Hindi and MuRIL ahead on Bengali and Tamil. [SNIPPET] — [GitHub Angshuman1211](https://github.com/Angshuman1211/multi-lingual-social-media-post-hate-speech-detector)
- MuRIL vs IndicBERT on a Devanagari hate task: MuRIL 0.6904 F1 vs IndicBERT 0.5924. [SNIPPET] — [CHIPSAL 2025](https://aclanthology.org/2025.chipsal-1.28.pdf)

### Inferences
- The Telugu and Tamil evidence points to a pattern. Linear TF-IDF (especially char n-grams) is competitive for the mid-field, and the top-of-leaderboard systems are transformer ensembles or domain-tuned encoders (MuRIL, Hindi-Abusive-MuRIL). For a single-team comparison, a TF-IDF model that beats a weak transformer baseline is not evidence that transformers are bad.
- Agglutinative morphology is a plausible reason char n-grams work for Telugu. I found one Tamil result consistent with this but no Telugu study that tests it directly.
- The 2025 Telugu transformer results may be underperforming because of fine-tuning instability, not because the task needs a linear model. The collapse rows in the Kakarla table support this but do not prove it.

### Gaps
- No verified Telugu study reports a head-to-head TF-IDF + SVM vs. transformer comparison with paired significance tests.
- I could not open the HOLD-Telugu findings paper, the Sandalphon paper, the Pannerselvam 2023 Telugu-English TF-IDF + SMOTE paper, or the 2026 Telugu DravidianLangTech paper (Reddy & Bellamkonda, ACL 2026.dravidianlangtech-1.1). Results for these are unverified.
- The sibling note in `research_notes/Telugu abuse detection state of art/` states the same gap (no Telugu TF-IDF parity precedent); this note agrees with it.

---

## Q3. Dataset artifacts, lexical shortcuts, near-duplicate leakage and annotation noise

### Takeaway
Hate-speech benchmarks are known to contain lexical shortcuts, mainly slurs and identity terms, and near-duplicates across splits. Several recent papers show that these inflate reported scores. For Telugu, the MACD and HOLD-Telugu benchmarks are at risk for the same reasons, but I did not find a verified Telugu leakage audit.

### Cited Findings
- Spurious vs authentic lexical artifacts in hate speech (NAACL 2022): cross-platform study on English data from four platforms; distinct spurious artifacts "require different treatments" for robustness and fairness; simple data-centric baselines are "surprisingly effective." Releases baselines and artifact code. [FETCHED abstract] — [aclanthology 2022.naacl-main.221](https://aclanthology.org/2022.naacl-main.221/)
- Identity-term false positives: Kennedy et al. (ACL 2020) show BERT classifiers flag benign uses of identity terms; explanation regularization reduces false positives on NYT articles (non-hate by construction) without losing in-domain performance. [SNIPPET] — [ACL 2020.acl-main.483](https://aclanthology.org/2020.acl-main.483)
- HateCheck (ACL 2021): 29 functional tests, 18 hate and 11 non-hate contrast tests (including reclaimed slurs and negation). Shows that held-out accuracy can overstate generalisation. [SNIPPET] — [arXiv 2012.15606](https://arxiv.org/abs/2012.15606); data at paul-rottger/hatecheck-data (GitHub, from memory, [UNVERIFIED] URL)
- Latent Hatred (arXiv 2109.05322): most public hate datasets were seeded with hate lexicons or explicit phrases; the paper reports about 20% profanity in the hate class across datasets. Single-source claim. [SNIPPET] — [arXiv 2109.05322](https://arxiv.org/pdf/2109.05322)
- Annotator bias: Measuring Hate Speech corpus has 135,556 labels from 8,472 annotators with socio-demographic data. [SNIPPET] — [arXiv 2410.07991](https://arxiv.org/pdf/2410.07991)
- Duplicates across 20 social-media NLP datasets (arXiv 2410.03545): "varying levels of data duplication"; duplication can cause label inconsistency, data leakage and overestimated SOTA results. [FETCHED abstract; method and numbers not verified] — [arXiv 2410.03545](https://arxiv.org/abs/2410.03545)
- Pretraining and test overlap: HateModerate (arXiv 2307.12418) keeps only test items absent from each model's training data. [SNIPPET] — [arXiv 2307.12418](https://arxiv.org/pdf/2307.12418)
- Reproduction study (IJISRT, 2026): 96.8% of test items appear verbatim in training under the original protocol; the reported accuracy of 0.9804 falls under corrected evaluation. Low-profile venue; verify before citing. [SNIPPET] — [IJISRT 26SEP325](https://ijisrt.com/assets/upload/files/IJISRT26SEP325.pdf)
- Train-test overlap as a general text-classification problem: Elangovan et al. (EACL 2021) quantify overlap in benchmarks and its effect on reported performance. [SNIPPET; authorship and the exact paper ID were not verified] — [aclanthology 2021.eacl-main.113](https://preview.aclanthology.org/fix_video/2021.eacl-main.113.pdf)
- Near-duplicates: n-gram matching misses near-duplicates with noise or abridgement; neural similarity catches more (NEWS-COPY, ICLR 2022). [SNIPPET; mapping of the arXiv ID to NEWS-COPY is unverified] — [arXiv 2210.04261](https://arxiv.org/abs/2210.04261)
- Practitioner example: a 9-point eval gain vanished after deduplication, with about 6.2% of eval rows flagged at Jaccard 0.7. Anecdotal. [SNIPPET] — [DEV Community post](https://dev.to/marcuswwchen/a-9-point-eval-gain-vanished-when-we-deduped-train-against-test-3baj)

### Inferences
- The MACD Telugu subset has 52% abuse and comments grouped under 92,881 posts. Comments on the same post are likely similar and share vocabulary, so a random comment-level split can leak. A group split by post is needed, and I did not find a published check of this for MACD.
- Telugu slur or abuse lexicons would give a shortcut. The test is a lexicon-only classifier (presence of any lexicon term); if it matches the model's F1, the benchmark is largely lexical.
- Kakarla's translated results show near-identical numbers across three models. This is a sign that a model can learn to predict the majority class, which a shortcut-heavy data set makes easier to hide.

### Gaps
- No verified Telugu or Dravidian paper runs a near-duplicate or group-split audit on MACD or HOLD-Telugu.
- The quantitative leakage numbers above come from snippets. The IJISRT and dedup-blog numbers are not peer-reviewed or checked.

---

## Q4. Recommended diagnostics and significance reporting

### Takeaway
Run four checks before claiming parity: a cross-split duplicate audit with a group-aware split, a lexicon-only and top-feature shortcut check, paired significance tests on the same test items, and multi-seed transformer runs with collapse detection. Use the Dror et al. framework for multi-dataset claims.

### Cited Findings
- Dror et al. (TACL 2017, Replicability Analysis): paired bootstrap for parsing, paired McNemar for binary sentiment, Wilcoxon signed-rank for POS, Steiger test for word similarity; argues that multi-comparison needs a dedicated framework. [SNIPPET] — [arXiv 1709.09500](https://arxiv.org/pdf/1709.09500), [TACL Q17-1033](https://aclanthology.org/Q17-1033/)
- Dror et al. (ACL 2018, Hitchhiker's Guide): a survey of ACL and TACL 2017 papers found significance testing often ignored or misused. [SNIPPET] — [ACL P18-1128](https://preview.aclanthology.org/menus/P18-1128/)
- Dror, Peled-Cohen & Shlomov, *Statistical Significance Testing for NLP* (Morgan & Claypool, 2020) covers test choice and multiple comparisons. [SNIPPET; book listing] — [publisher listing](https://tenlong.com.tw/products/9781681737959)
- HateCheck (see Q3) provides a functional test suite that can be ported to Telugu. [SNIPPET]

### Inferences (recommended diagnostics, my synthesis)
1. **Cross-split duplicate audit.** Exact-match and MinHash/Jaccard (threshold around 0.7–0.8, state it) between train and test, after normalising Telugu script and transliteration. Report how many test items are removed and re-score all models on the cleaned split.
2. **Group-aware split.** Split by post or user (MACD exposes post and user IDs). Compare random vs group-split scores; a large drop indicates leakage.
3. **Shortcut checks.** (a) Lexicon-only baseline: predict hate if any term from a slur or abuse list appears; report its F1 next to the model's. (b) Top-weighted n-grams from the linear SVM, checked for identity or slur terms. (c) HateCheck-style Telugu contrast set: reclaimed or quoted slurs, negation, identity terms in benign context.
4. **Significance on the same test items.**
   - Paired bootstrap (about 10,000 resamples) on the difference in macro-F1 between two systems. Report the 95% CI of the difference.
   - McNemar's exact test on paired correct/incorrect predictions for accuracy-style comparison.
   - For multi-dataset claims, use the Dror et al. replicability analysis rather than one test per dataset.
5. **Multi-seed.** At least 5 seeds for each transformer (vary init and data order), mean ± std, and report all runs including failures. Count majority-class collapse runs separately (see Kakarla table). [UNVERIFIED: the "Show your work" recommendation for reporting seed variance is from memory and not checked here]
6. **Metrics.** Macro-F1 and minority-class F1 together, plus accuracy with class priors. Accuracy alone is misleading for imbalanced Telugu data (MACD is near balanced at 52%, but HOLD-Telugu and others may not be).
7. **Annotation noise.** Re-annotate a random sample (about 200 items) with a second annotator; report Cohen's kappa. [UNVERIFIED as a specific recommendation; standard practice from memory]

### Gaps
- No Telugu paper reviewed reports paired bootstrap or McNemar tests on its own results.
- No verified source gives a standard threshold for "meaningful" parity margin on hate detection.

---

## Source status summary
- Fetched and read (abstract or HTML): NAACL 2022 artifacts (abstract), arXiv 2502.10632 (HTML, full tables), arXiv 2605.03439 (abstract), arXiv 2410.03545 (abstract).
- PDF fetches that returned raw binary (no text): NAACL 2022 PDF, HOLD-Telugu 2024 PDF. Their content above comes from snippets only.
- Everything else is [SNIPPET] or [UNVERIFIED] as tagged.
- Not checked at all: MACD GitHub repository, Sandalphon paper, Pannerselvam 2023, Reddy & Bellamkonda 2026 Telugu paper, the 2021 Dravidian findings paper.

## Recommended next verifications (for the report writer or the project)
1. Open the HOLD-Telugu 2024 findings and the MACD NeurIPS PDF in a text-capable tool (e.g. `pdftotext`) to get the actual baseline table.
2. Run the cross-split duplicate audit and the lexicon-only baseline on the project's own MACD Telugu split. This is the test that would confirm or refute the project's TF-IDF parity result.
3. Locate the Reddy & Bellamkonda 2026 Telugu paper (ACL 2026.dravidianlangtech-1.1) and check whether it has a classical baseline.
