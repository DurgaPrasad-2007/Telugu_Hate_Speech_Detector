# PEFT, Calibration, and Efficient Deployment for Small-Encoder Text Classification (2024 to Oct 2026)

Scope: (a) LoRA/DoRA/AdaLoRA/IA3 vs full fine-tuning for encoder classification; (b) calibration and selective prediction for toxicity; (c) INT8 quantization, distillation, SetFit; (d) three-way allow/review/block moderation practice. Status tags: [verified] = source page or text fetched and read; [snippet] = seen only in search-result text; [unverified] = not checked.

Today is 2026-10-28. Searches were run in this session; no tables were parsed from PDFs that failed to decode.

---

## Verified ID checks

- arXiv 2605.14074 EXISTS [verified]. "Fair and Calibrated Toxicity Detection with Robust Training and Abstention," Mokshit Surana, cs.LG, v1 submitted 13 May 2026. https://arxiv.org/abs/2605.14074
  - Relevance to (b): fairness-focused. Covers temperature scaling, confidence-based abstention, per-identity thresholds, ERM vs reweighting vs Group DRO. Key claims as reported in the abstract: subgroup calibration gaps can hide behind good overall calibration; temperature scaling struggles when miscalibration is uneven; abstention helps non-identity content more than identity-mentioning content.
  - It is NOT a conformal-prediction paper. Do not cite it as conformal evidence.
- arXiv 2606.05654 EXISTS [verified] but is NOT an encoder-PEFT or calibration paper. "When Surface Form Changes Moderation Decisions: A Paired Study of Code-Mixed Workflow Instability," Suraj Babu Thimma Krishnaram, Yibo Hu, Karthikeyan Saravanan. cs.SE, v1 4 June 2026, v2 5 June 2026. https://arxiv.org/abs/2606.05654
  - Relevance to (d) only. See section D below.
- arXiv 2510.04139 exists [snippet]: "Fine Tuning Methods for Low-resource Languages" (Bakkenes, Wang, Johansson). Decoder-model focus (Gemma, GPT-SW3, BLOOM, NLLB references). Not an encoder comparison. Results could not be read.

---

## A. PEFT vs full fine-tuning for encoder classification

### Takeaway
LoRA on an encoder is a routine baseline, not a novel contribution. Parity is not guaranteed. The best controlled encoder evidence found (XLM-R Large, multilingual news classification) shows PEFT typically a few F1 points below full fine-tuning, with the gap narrowing as data shrinks and with per-task variation. Nothing found directly tests LoRA r=8 on XLM-R-base for Telugu or other Indic languages.

### Cited Findings
- XLM-R Large, LoRA vs adapters vs full fine-tuning, multilingual news article classification (PLoS ONE, 2024; Razuvayevskaya et al.). Verified by reading the PMC full text. https://pmc.ncbi.nlm.nih.gov/articles/PMC11068208/
  - Backbone: XLM-RoBERTa Large. XLM-R Base appears only in an appendix. Methods: LoRA, Pfeiffer adapter (Houlsby in appendix), BitFit dropped. IA3 not evaluated.
  - Trainable params: FFT ~560M; LoRA ~3.2M / ~4M / ~2M across the three sub-tasks; adapters ~26M / ~7M / ~7M.
  - Mean ± SD over 3 seeds, Table 6, sub-task 1 F1-macro: FFT 59.9 ± 3.1; LoRA 57.9 ± 6.3; Adapter 58.0 ± 2.0.
  - Sub-task 2 F1-macro: FFT 49.2 ± 7.4; LoRA 45.3 ± 7.1; Adapter 47.1 ± 7.9.
  - Sub-task 3 F1-micro: FFT 41.8 ± 8.6; LoRA 42.9 ± 9.5; Adapter 42.2 ± 9.5 (LoRA slightly above FFT on this one).
  - Languages: seen in training = English, French, German, Italian, Polish, Russian. Test-only ("surprise") = Spanish, Greek, Georgian.
  - Authors' conclusions (as reported in the fetched text): FFT generally strongest on sub-tasks 1 and 2; LoRA competitive or slightly ahead on sub-task 3; PEFT cut trainable parameters roughly 140x to 280x and training time about 32% to 44%; full FT beats PEFT on zero-shot cross-lingual transfer in the joint multilingual setting; differences often drop below 1% as training data shrinks.
  - Caveat: the sub-task 1 gap (about 2 F1-macro points) is within the seed SD for LoRA (6.3). Not a clean win for FFT.
- Same study, an earlier arXiv version listed as 2308.07282 appeared in search results [snippet]: https://arxiv.org/pdf/2308.07282 . Treat it as the same study unless checked.
- DoRA: arXiv 2402.09353, "DoRA: Weight-Decomposed Low-Rank Adaptation," Shih-Yang Liu et al., cs.CL, first submitted 14 Feb 2024, later versions exist (v4, v6). [snippet] https://arxiv.org/abs/2402.09353v4
  - Authors' abstract-level claim: improves learning capacity and stability over LoRA with no added inference overhead. Reported experiments are LLaMA, LLaVA, and VL-BART (commonsense reasoning, visual instruction tuning, image/video-text). No encoder-classification numbers were read. [unverified for encoders]
  - Hugging Face PEFT added DoRA in v0.9.0 (per HF paper page snippet). https://huggingface.co/papers/2402.09353
- AdaLoRA and IA3 on encoder classification: not found in this session's searches. [unverified / gap]
- Other low-resource PEFT evidence:
  - Frontiers in Big Data 2025 (DOI 10.3389/fdata.2025.1677331) [snippet only]: DistilBERT on English AG News and Amazon Reviews, LoRA/IA3/ReFT vs full FT under low-resource conditions. Not multilingual. https://public-pages-files-2025.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1677331/epub
  - "LoRA vs Full Fine-tuning: An Illusion of Equivalence," arXiv 2410.21228 [snippet]: LoRA-trained weights differ in spectral structure and forget less pre-training behavior. LLM-focused, not encoder classification.
  - A Zenodo "AI-generated" report on LoRA vs full FT for XLM-R-large on XTREME exists (https://zenodo.org/record/21520835). Generated by an autonomous system; NOT used as evidence. [unverified, exclude]

### Inferences
- LoRA r=8 on XLM-R-base is a standard baseline, not a notable contribution on its own. A paper claiming it would need a matched-budget comparison, multiple seeds, and per-language or per-class results.
- Parity is not expected in general. Expect a small gap (roughly 0 to 2 F1 points on this kind of task) that is comparable to seed variance on small data. Parity is more plausible when the classifier head and most data are in-language and the training set is moderate.
- For our reporting, an honest sanity check would be full FT vs LoRA r=8 with 3 to 5 seeds, reporting mean ± SD. Differences smaller than seed SD should not be called wins.

### Gaps
- No encoder LoRA vs FFT study for Telugu or Dravidian languages was found.
- No verified XLM-R-base (not Large) LoRA r=8 numbers were found.
- AdaLoRA and IA3 on encoder classification: no verified numbers.
- The 2024 PLoS ONE full-text table was readable only through the PMC mirror; the main-text parameter-count row for LoRA is paraphrased from the fetched table.

---

## B. Calibration and selective prediction for toxicity

### Takeaway
Temperature scaling is cheap and does not change argmax, so it fixes ECE but not ranking. Recent toxicity-specific evidence (2605.14074) says calibration can be uneven across identity subgroups, which aggregate ECE hides. No conformal toxicity paper was found in this session.

### Cited Findings
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017 (arXiv 1706.04599). Temperature scaling divides logits by a single scalar T; accuracy is unchanged; ECE uses M equal-width confidence bins. [snippet, search summaries of the paper] https://arxiv.org/abs/1706.04599 , https://proceedings.mlr.press/v70/guo17a.html
  - Exact ECE values were not read. [gap]
- Fair and Calibrated Toxicity Detection with Robust Training and Abstention, arXiv 2605.14074 (Surana, 13 May 2026). [verified abstract] https://arxiv.org/abs/2605.14074
  - Calibration gaps can be hidden by good overall calibration.
  - Temperature scaling struggles when miscalibration is uneven across subgroups.
  - Abstention helps non-identity content more than identity-mentioning content.
  - Group DRO equalizes calibration across groups but worsens global miscalibration; reweighting widens calibration gaps.
- Dynamically Scaled Activation Steering, arXiv 2512.03661 [snippet]: a toxicity gate (PCA + logistic regression) had ECE about 0.025 to 0.049 before scaling, fitted temperatures 0.93 to 1.70, and temperature scaling added essentially nothing. https://arxiv.org/pdf/2512.03661 . Shows that a well-calibrated gate may not need TS.
- Cross-domain selective prediction (arXiv 2603.08907) [snippet]: on intent-classification data, TS reduced ECE from about 0.52 to 0.040 on one dataset and gave a more usable rejection-threshold range. Non-toxicity, so only indirect. https://arxiv.org/pdf/2603.08907
- Uncertainty-Driven Reliability survey, arXiv 2508.07556 [snippet]: argues TS is monotone and cannot change ranking, so it does not fix selective-classification gaps. Also notes disagreement in the literature on whether calibration helps abstention. https://arxiv.org/pdf/2508.07556
- Adaptive ECE and binning: Nixon et al., "Measuring Calibration in Deep Learning," arXiv 1904.01685 (2019; some secondary sources cite 2020) [snippet]. Adaptive (equal-count) binning gives more stable metric rankings across bin counts. Standard ECE is a biased estimate and depends on binning. https://arxiv.org/abs/1904.01685 , https://ar5iv.labs.arxiv.org/html/1904.01685
  - Full text not read. [unverified detail]

### Inferences
- For a three-way (allow / review / block) policy, the decision quality depends on the two thresholds and the review-band size. ECE alone does not capture it. Report per-band precision, review rate, and error rate in the review band.
- If we add TS, it will not change the decision for argmax-only logic but will change where thresholds fall. Calibrate on a held-out split, not the training split.
- Conformal prediction (not verified in this session) would need a calibration split and gives coverage guarantees per class, not per subgroup. Subgroup coverage must be checked separately (per 2605.14074).

### Gaps
- No conformal prediction result for toxicity classifiers was found in 2024 to 2026 searches.
- No toxicity-specific TS vs ECE number from an encoder (XLM-R or MuRIL) was found.
- Telugu-specific calibration evidence: none found.

---

## C. INT8 quantization, distillation, SetFit

### Takeaway
A 1.2-point macro-F1 drop from INT8 dynamic quantization (our 0.9042 to 0.8926) is within the range reported for dynamic PTQ on XLM-R-base in published or public results, but the comparison is weak because the exact setups differ. Expect roughly 0.5 to 2 points on classification and larger losses on some languages.

### Cited Findings
- O'Neill and Dutta, "Self-Distilled Quantization," ACL 2023 (arXiv 2307.05972). Reported in a search summary: XLM-R Base, XNLI, INT8 post-training dynamic quantization. English 83.9 (FP32) to 80.1 (PTQ dynamic); average across languages 73.9 to 72.5 (about -1.4). The average drop figure is the summary's reading of the table, not checked against the PDF. [snippet] https://arxiv.org/pdf/2307.05972 . Authors also note multilingual models generalize less well after quantization when fine-tuned on English only.
- Intel, INT8 dynamic-quantized XLM-R base fine-tuned on MRPC (Hugging Face model card) [snippet]: eval-F1 0.9010 (FP32) vs 0.8966 (INT8), about -0.4 points; model size 1061 MB to 354 MB. https://huggingface.co/Intel/xlm-roberta-base-mrpc-int8-dynamic-inc
- Q8BERT (arXiv 1910.06188, Zafrir et al.) [snippet, numbers conflict]: quantization-aware training gives about 4x compression with relative error under 1% on most GLUE tasks (RTE excluded). A secondary summary lists DQ (dynamic quantization) drops from about 1.4% (SST-2) to 9.13% (RTE). The full PDF did not decode, so these per-task numbers are unverified. Paperswithcode and SQuAT list conflicting CoLA numbers for Q8BERT (65.0 vs 58.5), so do not cite a single Q8BERT figure without checking. https://arxiv.org/pdf/1910.06188
- Zero-shot dynamic quantization for transformers (arXiv 2211.09744) [snippet]: plain INT8 loses more on some tasks (for example BERT-base CoLA 58.0 to 48.3), largely recovered by a clipping method. https://arxiv.org/pdf/2211.09744
- Marchisio et al., 2024, "How Does Quantization Affect Multilingual LLMs?" (EMNLP Findings) [snippet]: quantization impact varies by language; non-Latin-script languages hurt most. Decoder LLMs, so indirect evidence. https://preview.aclanthology.org/watermark/2024.findings-emnlp.935
- PyTorch dynamic quantization tutorial [snippet]: dynamic = weights static INT8, activations quantized per batch; per-channel weight quantization helps accuracy. https://pytorch.org/tutorials/prototype/graph_mode_dynamic_bert_tutorial.html
- SetFit, "Efficient Few-Shot Learning Without Prompts," arXiv 2209.11055 (Tunstall et al., 2022) [snippet]: contrastive fine-tuning of a Sentence Transformer then a classification head; abstract says comparable to PEFT and PET and an order of magnitude faster to train. Multilingual via a multilingual sentence-transformer body. Full tables not read. https://arxiv.org/abs/2209.11055 , https://huggingface.co/blog/setfit
- MuRIL, arXiv 2103.10730 (Google, 2021) [snippet]: Indian-language model with vocabulary built for IN languages; XTREME average 68.6. https://arxiv.org/pdf/2103.10730
  - No INT8 quantization results for MuRIL were found. [gap]

### Inferences
- Our drop: 0.9042 to 0.8926 macro-F1 is -1.16 points (about -1.3% relative). This is consistent with the -0.4 (MRPC, F1) and -1.4 (XNLI average) figures above, and it is not a red flag on its own.
- But one run per setting is not enough. With the seed SDs seen in Section A (roughly 2 to 9 F1 points on small data), a 1-point gap needs several seeds or a paired bootstrap on the test set before it is called a real drop.
- Quantization can also shift per-language or per-class performance more than the average. Break down the drop per class (especially the review and minority classes) and per language if possible.
- Dynamic quantization is a reasonable first deployment step for encoders on CPU because it needs no calibration set. Static quantization (calibrated) often recovers more accuracy but needs a representative set; this was not verified for XLM-R.
- Distillation and SetFit were not benchmarked against our setup. SetFit is a plausible alternative for low-data regimes but its multilingual numbers for Telugu were not found.

### Gaps
- No verified static-vs-dynamic INT8 comparison for XLM-R-base classification with per-class numbers.
- No verified Q8BERT per-task table (PDF did not decode; secondary numbers conflict).
- No MuRIL quantization results.
- No Telugu-specific quantization results.

---

## D. Moderation policy practice for three-way allow / review / block

### Takeaway
Industry and research practice is a two-threshold, cascade design: confident-allow, confident-block, and a human-review middle band. Jigsaw recommends human oversight rather than full automation. No authoritative numeric cut-offs for hate speech were found. The most relevant recent workflow evidence is 2606.05654.

### Cited Findings
- arXiv 2606.05654 (Krishnaram, Hu, Saravanan; cs.SE; 4 June 2026, revised 5 June 2026) [verified abstract]. Hate-speech moderation as a routing workflow (allow / flag / human review). Paired clean English vs Tamil-English code-mixed items, thresholds tuned on clean English. About 26.5% of items change decision under code-mixing. Human-review share rises from about 14% to about 30%. False flags on non-hateful content rise from about 7% to about 10%. Tamil-only degrades more. A disagreement-based deferral rule cuts automatic errors on stressed inputs, but only by sending more items to review. https://arxiv.org/abs/2606.05654
  - Relevance: directly on point for our code-mixed Telugu setting, and the "review share" metric is the one to report.
  - Preprint, not peer reviewed as far as checked. [unverified peer status]
- Jigsaw / Perspective API guidance [snippet]: Jigsaw discourages full automation and recommends human oversight. https://coralproject.net/?p=571 (search-result link, not the primary Jigsaw page) [weak source]
- Threshold guidance [snippet]: one research study suggests 0.7 to 0.9 for Perspective research use, with tuning per application. Not a policy setting. [weak source]
- Lost in Moderation, arXiv 2503.01623 [snippet]: commercial moderation APIs over- and under-moderate group-targeted hate speech and linguistic variation. https://arxiv.org/pdf/2503.01623 (not read in full)
- Digital Guardians, arXiv 2501.01256 [snippet]: GPT-4, Perspective, and OpenAI Moderation API on German reader comments. https://arxiv.org/pdf/2501.01256 (not read in full)
- Calibration-linked: 2605.14074 Section B (abstention is unequal across identity-mentioning content).

### Inferences
- Three-way policy should be evaluated as a workflow: report the allow, review, and block rates; the precision and error rate inside each band; and the review-band size. A classifier with good macro-F1 can still send too much to review.
- Thresholds should be chosen on a held-out in-language set and checked for code-mixed and Telugu-only inputs (per 2606.05654).
- Include a subgroup check on identity-mentioning content (per 2605.14074).

### Gaps
- No public, authoritative three-way moderation threshold table for Telugu or hate speech was found.
- Industry policy documents (Meta, Google, OpenAI) were not searched in this session.

---

## Answers to the key questions

1. Is LoRA r=8 on XLM-R-base notable or routine?
   Routine. It is a standard baseline. The closest controlled evidence (XLM-R Large, PLoS ONE 2024) shows LoRA about 2 F1-macro points behind FFT on the main sub-task, within seed noise, and ahead on one sub-task. Parity is typical in the small-gap sense, not guaranteed. A paper would only be notable with matched seeds, per-language results, and low-resource or code-mixed evaluation.

2. Is a 1.2-point INT8 dynamic drop (0.9042 to 0.8926 macro-F1) normal?
   Plausibly yes. Public numbers for XLM-R-base dynamic PTQ show drops of about 0.4 (MRPC F1) and about 1.4 (XNLI average, with about 3.8 on English). Our drop sits in that range. Confirm it is larger than seed variance and check per-class and per-language breakdowns before calling it normal.

3. Best practices to report.
   - Mean ± SD over at least 3 to 5 seeds, plus a paired bootstrap or McNemar-type test for the quantized vs FP32 comparison.
   - Macro-F1 and per-class F1, especially for review and minority classes.
   - Per-language F1 for Telugu, code-mixed, and English splits.
   - Calibration: ECE with the binning scheme stated (equal-width or adaptive), bin count, and sensitivity to bin count (per Nixon et al.); reliability diagram; per-subgroup ECE (per 2605.14074).
   - Selective prediction: coverage vs risk curve at the chosen review rate.
   - Workflow metrics for three-way decisions: allow / review / block rates, review-band precision, and automatic error rate (per 2606.05654).
   - Deployment: model size, latency on named CPU hardware, batch size, thread count, and the quantization mode (dynamic vs static, per-channel or per-tensor).
   - Parameter counts and trainable fraction for each PEFT method.
   - Exact checkpoints, library versions, and random seeds.

---

## Unverified items to recheck before citing

- Q8BERT per-task DQ drops (1.4% to 9.13%): PDF did not decode; secondary source only.
- Self-Distilled Quantization average (73.9 to 72.5): read from search summary, not the PDF table.
- Dynamic Scaled Activation Steering ECE and temperature values: search summary only.
- DoRA encoder results: none found.
- SetFit multilingual numbers: none found.
- Frontiers 2025 low-resource PEFT numbers: not read.
- Nixon et al. ECE pitfalls: abstract-level only.
- 2606.05654 peer-review status.
