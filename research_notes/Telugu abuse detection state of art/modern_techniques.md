# Modern techniques for low-resource toxicity / hate-speech classification (state as of Oct 2026) - judging XLM-R-base + LoRA

Method note: ~10 web searches, mostly snippet-level; full papers were NOT opened. Anything not directly cited is a Gap. Many questions (DoRA/PiSSA/LoRA-GA, Sarvam/Airavata/Krutrim, ShieldGemma, Llama Guard) were not searched and are listed as gaps rather than asserted.

## 1. What is SOTA for low-resource multilingual toxicity/hate classification (encoders, LLMs, guard models)?

### Takeaway
No single SOTA recipe was found for Telugu. Evidence shows fine-tuned Indic-aware encoders (MuRIL variants) remain strong, few-shot LLMs can match them on some subtasks, and the "modern" encoder (mmBERT) and guard models (Qwen3Guard, PolyGuard) exist but are not shown on Telugu toxicity.

### Cited Findings
- mmBERT (JHU, arXiv Sep 2025; ICML 2026 proceedings): ModernBERT-style encoder, 3T+ tokens, 1,800+ languages, Gemma 2 tokenizer, 307M params total (110M non-embedding) for base; low-resource languages added only in the final decay phase; authors claim large gains over prior multilingual encoders (author-reported, no independent evaluation found) - [HF blog](https://huggingface.co/blog/mmbert); [arXiv](https://arxiv.org/abs/2509.06888); [ICML](https://proceedings.mlr.press/v306/marone26a.html)
- EuroBERT covers only 15 languages; mmBERT base/small beat EuroBERT-210m even on EuroBERT's own languages (NLI-type tasks, not toxicity) - [mmBERT paper](https://arxiv.org/pdf/2509.06888). Implication: EuroBERT is irrelevant for Telugu.
- Qwen3Guard (arXiv 2510.14276, Oct 2025): Generative and Stream variants, 0.6B/4B/8B, Apache 2.0, claims up to 119 languages and dialects, SOTA claims on multilingual safety benchmarks; training translated content into extra languages. Per-language numbers unverified - [arXiv](https://arxiv.org/abs/2510.14276v1). These are prompt/response safety guards, not social-media comment hate classifiers.
- PolyGuard (arXiv 2504.04377, Apr 2025): 17 languages (includes Hindi; Telugu not listed in the snippets seen), PolyGuardMix 1.91M samples, claims +5.5% over open and commercial safety classifiers; classifies prompt harm, response harm, refusal - [arXiv](https://arxiv.org/abs/2504.04377v1); [model list](https://docs.mozilla.ai/any-guardrail/api-reference/index/content-safety/poly-guard)
- Telugu specific: Code-mixed Telugu-English hate detection (arXiv 2502.10632, Feb 2025) compared TeluguHateBERT, HateBERT, DeBERTa, MuRIL, IndicBERT, Hindi-Abusive-MuRIL with LoRA and ensembling; Hindi-Abusive-MuRIL best - [arXiv](https://arxiv.org/pdf/2502.10632)
- X-MuTeST (arXiv 2601.03194, Jan 2026, AAAI 2026): LLM reasoning plus explainability-guided training for Indic hate; token-level rationales incl. 4,492 Telugu samples. Details not verified beyond a search snippet - [arXiv](https://arxiv.org/abs/2601.03194)
- Devanagari shared task: few-shot Gemma-2 9B about equal to fine-tuned IndicBERT v2 (F1 0.7588 vs 0.7582 test, subtask B); ensemble best at 0.7652 - [arXiv 2411.06850](https://arxiv.org/html/2411.06850v1) (this is Nov 2024, flag as older)
- Code-mixed Hindi-English (CHiPSAL 2025): Indic-instruction-tuned LLMs fell behind; best was fine-tuned Gemma-2 variant - [ACL Anthology](https://preview.aclanthology.org/setup/2025.chipsal-1.5)
- 2025 WMT paper over 17 languages: translation-based approach is effective for multilingual toxicity detection - [ACL Anthology](https://aclanthology.org/2025.wmt-1.15.pdf)
- XLM-R has been reported to lag mBERT/RoBERTa in some single-language toxic classification studies (older, 2021) - [arXiv 2111.00981](https://arxiv.org/pdf/2111.00981)

### Inferences
- XLM-R-base is a 2019-era encoder; mmBERT (2025) is the obvious "modern encoder" comparator and is a cheap swap, but I found no toxicity results for it. Not using it is a defensible gap, not an error, but a reviewer may ask.
- MuRIL (Indian-language-specific pretraining) is the more field-standard Telugu encoder baseline than XLM-R; absence of MuRIL/IndicBERT comparison would look dated. (This repo already ships a MuRIL INT8 model per git log.)
- Guard models target LLM prompt/response safety, so they are adjacent, not a replacement, for comment-level abuse classification.

### Gaps
- No Telugu result for Llama Guard, ShieldGemma, Qwen3Guard, PolyGuard found; not searched for Sarvam, Airavata, Krutrim, Gemma/Qwen3 fine-tuning on Telugu toxicity.
- No ModernBERT/mmBERT/EuroBERT vs XLM-R toxicity head-to-head found.
- No 2026 LLM-judge benchmark for Indic toxicity opened.

## 2. Where does LoRA stand; is LoRA on XLM-R-base modern or routine?

### Takeaway
LoRA on an encoder is routine, not novel; evidence points to near-parity with full fine-tuning on classification with large compute savings. Newer variants (DoRA, PiSSA, LoRA-GA) were not verified.

### Cited Findings
- Multilingual news classification (PLoS ONE 2024): PEFT (LoRA/adapters) cut training time, parameters, VRAM with little loss in quality; full FT better for zero-shot unseen languages under joint training - [arXiv 2308.07282](https://arxiv.org/pdf/2308.07282) (older, 2023-24)
- Persian NLP comparison of full FT, LoRA, AdaLoRA, DoRA: LoRA and DoRA very competitive with full FT at much lower compute (summary only) - [JAD](https://jad.shahroodut.ac.ir/article_3392.html)
- Roman Urdu hate speech: PEFT vs prompt engineering comparative study, LoRA beneficial in low-resource (no numbers seen; arXiv 2608.21408, ID not verified by me) - [arXiv](https://arxiv.org/pdf/2608.21408)
- Telugu code-mixed hate paper already used LoRA on encoders in Feb 2025 - [arXiv 2502.10632](https://arxiv.org/pdf/2502.10632)
- "Fine Tuning Methods for Low-resource Languages" (arXiv 2510.04139) surfaced but contents not read - [arXiv](https://arxiv.org/pdf/2510.04139)

### Inferences
- A 278M-parameter encoder is cheap enough to fully fine-tune on one GPU, so LoRA there is a convenience choice; claim it for efficiency, not accuracy. Report a full-FT control.

### Gaps
- No evidence gathered on QLoRA/DoRA/PiSSA/LoRA-GA vs LoRA for encoders; no Telugu-specific PEFT-vs-full-FT parity numbers.

## 3. Data-centric and robustness techniques

### Takeaway
LLM-based augmentation and targeted minority-class augmentation are active and show modest gains mainly when baseline data is small; robustness to transliteration/code-mixing is being studied mostly as evaluation. Calibration plus deferral to humans is emerging in 2026 work.

### Cited Findings
- Unfiltered-LLM semantic augmentation: about +4-6% F1 in low-resource, +1-2% on full data (abstract-level) - [Springer 2025](https://link.springer.com/article/10.1007/s00607-025-01518-8)
- Targeted augmentation of minority-class code-mixed Tamil/Malayalam/Kannada-English hate samples helps transformers (FLAICOL, DravidianLangTech 2026) - [ACL events](https://aclanthology.org/events/dravidianlangtech-2026/)
- GPT-based augmentation for code-mixed Indonesian hate tweets - [ETASR](https://etasr.com/index.php/ETASR/article/view/14342)
- Malayalam-English sentiment: synthetic data helped only when baseline was weak (secondary snippet, hence moderate confidence) - [search result summary above, DravidianLangTech listing](https://aclanthology.org/events/dravidianlangtech-2026/)
- Bangla LLM robustness under transliteration perturbations; BanTH translate/transliterate-then-classify prompting - [arXiv 2411.00691](https://web3.arxiv.org/abs/2411.00691)
- Code-mixed workflow instability in moderation: same content in clean-English vs code-mixed form gets different actions; abstention framed as human-review routing (arXiv 2606.05654, Jun 2026) - [arXiv](https://arxiv.org/pdf/2606.05654)
- Fair and calibrated toxicity detection with robust training and abstention (arXiv 2605.14074, May 2026): deferral only works if confidence tracks correctness uniformly across groups - [arXiv](https://arxiv.org/pdf/2605.14074)
- Older: uncalibrated abusive-content scores align poorly with humans (Perspective recalibration, 2020) - [ACL](https://preview.aclanthology.org/fix_video/2020.nlpcss-1.14.pdf)

### Inferences
- Reporting calibration (ECE) and a coverage-vs-accuracy/macro-F1 curve with human-review routing would read as current (2026); transliteration (Romanized Telugu) and obfuscation stress tests are the likely expected robustness check.

### Gaps
- No 2026 adversarial/obfuscation-robustness benchmark for Telugu found; back-translation evidence for Indic hate not found; I read only snippets of these papers.

## 4. Do strong classical baselines (TF-IDF+SVM) match transformers on small Indic datasets?

### Takeaway
Mixed: TF-IDF+SVM is a respectable baseline and sometimes close, but transformers (especially Indic-specific) typically win, mostly on the minority hate class.

### Cited Findings
- 38-model hate speech evaluation (arXiv 2509.14266, Sep 2025, not Indic): traditional methods (SVM, CatBoost) remain competitive (F1 above 88% on some data); RoBERTa best on small sets - [arXiv](https://arxiv.org/pdf/2509.14266)
- Comparative study: TF-IDF SVM macro-F1 0.64 comparable to some transformers, but all transformers beat other TF-IDF/GloVe models by large margin (English, 2022) - [arXiv 2202.09517](https://arxiv.org/pdf/2202.09517)
- LT-EDI 2025 Tamil caste/migration hate: TF-IDF SVM/LR compared to transformers; domain-adapted Tamil hate BERT best at macro-F1 0.88 - [ACL Anthology](https://preview.aclanthology.org/credits/2025.ltedi-1.18)

### Inferences
- Always include a TF-IDF (char n-gram) + SVM baseline; for agglutinative Telugu, character n-grams are the fair choice. A transformer that does not clearly beat it should be reported honestly.

### Gaps
- No Telugu-specific 2025-26 study showing TF-IDF+SVM matching transformers was found.

## Overall judgment (inference)
XLM-R-base + LoRA is routine/generic as of Oct 2026, not outdated-wrong but not modern. Strengthen with: MuRIL/IndicBERT and mmBERT comparators, a full-FT control, a TF-IDF baseline, a few-shot LLM (Gemma/Qwen) reference, calibration plus review-routing, and Romanized/code-mixed stress tests.
