# Approach survey (2025-2026) for Telugu hate-speech detection

Status tags: VERIFIED = arXiv abstract page fetched and read in this session (abstract only, not full text; no numbers beyond the abstract were checked). UNVERIFIED = seen only in a search snippet, or my own knowledge.

Ceiling finding: TF-IDF+SVC, XLM-R+LoRA and MuRIL+LoRA+ all tie at ~0.895 macro-F1, so the gains are in labels, selective prediction and moderator tooling, not architecture.

## 1. LLM annotators / label auditing / noisy labels
- Jadhav, Shanbhag, Thakurdesai, Sinare, Joshi. "On Limitations of LLM as Annotator for Low Resource Languages". arXiv 2411.17637 (v2 Mar 2025). https://arxiv.org/abs/2411.17637 . VERIFIED. Marathi: GPT-4o and Llama 3.1 405B annotate worse than fine-tuned BERT (10.2% and 14.1% accuracy gap). Apply: do NOT use an LLM as a label replacement; use it only to flag disagreements with existing labels (run a local 7-9B 4-bit model on the 8GB GPU over the test/val rows where the classifier is confidently "wrong"; human-review those).
- Dang, Mitrovic, Granitzer. "Toward Generalized Cross-Lingual Hateful Language Detection with Web-Scale Data and Ensemble LLM Annotations". arXiv 2604.09625 (Mar 2026, LREC 2026 submission). https://arxiv.org/abs/2604.09625 . VERIFIED. Continued MLM pretraining (~+3% macro-F1) and a LightGBM ensemble of 4 LLM labelers; synthetic labels help small models most. Apply: continued MLM pretraining of MuRIL on unlabeled Telugu comments (a few hours on 4060 Ti, LoRA or short run); use the ensemble-of-labelers idea as a meta-feature (classifier prob + LLM vote).
- Hasan et al. "Are LLMs Ready to Replace Bangla Annotators?" arXiv 2602.16241. https://arxiv.org/abs/2602.16241 . VERIFIED but WITHDRAWN by authors (methodological discrepancies). Do not cite.
- Confident learning / cleanlab / small-loss / co-teaching for hate speech: NO 2025 hate-speech paper found. UNVERIFIED (classic methods, Northcutt et al. 2021 and Han et al. 2018 are from memory). Apply: 5-fold out-of-fold probs from the existing TF-IDF+SVC (cheap, minutes), `cleanlab.filter.find_label_issues`, human-audit top 200, report macro-F1 on a hand-verified clean subset. This is the highest value-per-hour item.

## 2. Synthetic data / transliteration augmentation
- Kakarla, Venkata. "Code-Mixed Telugu-English Hate Speech Detection". arXiv 2502.10632 (Feb 2025). https://arxiv.org/abs/2502.10632 . VERIFIED. LoRA on TeluguHateBERT, HateBERT, DeBERTa, MuRIL, IndicBERT, RoBERTa; training on Google-Translated English often beat native Telugu; Hindi-Abusive-MuRIL best. Apply: repo already has src/translit.py; add a Telugu-script <-> Roman view and a translate-to-English view as test-time augmentation, check whether the ensemble beats 0.895.
- Tabassum et al. "Language Detection in Telugu Code-mixed Text using Transliteration-Augmentation", DravidianLangTech 2024. https://aclanthology.org/2024.dravidianlangtech-1.28.pdf . UNVERIFIED (snippet only; 2024, outside window).

## 3. Calibrated / conformal selective classification, HITL moderation
- Villate-Castillo, Del Ser, Sanz. "A Collaborative Content Moderation Framework for Toxicity Detection based on Conformalized Estimates of Annotation Disagreement". arXiv 2411.04090, Neurocomputing 647 (2025). https://arxiv.org/abs/2411.04090 . VERIFIED. Multitask toxicity + disagreement head, conformal prediction, moderator-set thresholds route to human review. Apply: src/conformal_policy.py already exists; add a split-conformal set (alpha=0.1) on a held-out calibration split and route "both-classes" or empty sets to review; report coverage vs accuracy-on-auto curve.
- Urbinati, Lai, Frenda, Stranisci. "Are you sure? Measuring models bias in content moderation through uncertainty". arXiv 2509.22699, Findings of EMNLP 2025. https://arxiv.org/abs/2509.22699 . VERIFIED. Conformal set size as a bias proxy across annotator groups. Apply: compute average set size per target-group/dialect slice; a free fairness audit from the same conformal code.

## 4. Explainability for moderators
- Rehman et al. "X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and a Novel LLM-consulted Explanation Framework". arXiv 2601.03194, AAAI 2026. https://arxiv.org/abs/2601.03194 . VERIFIED (abstract). Word-level rationales for Hindi, Telugu (4,492), English; explainer = probability drop when removing uni/bi/trigrams, unioned with LLM explanations. Code at github.com/ziarehman30/X-MuTeST (UNVERIFIED, from search snippet). Apply: occlusion n-gram explainer is model-agnostic, ~30 lines, works on the TF-IDF model on CPU; highlight tokens in app.py; score against their Telugu rationales if the data downloads.

## 5. PEFT / distilled CPU serving
- No 2025 paper found distilling MuRIL/XLM-R for Indic hate speech. UNVERIFIED gap. Apply: src/distill.py exists; distill the MuRIL+LoRA+ / ensemble soft labels into the INT8 MuRIL already packaged, or into TF-IDF+linear (soft-label regression), and measure latency vs the 0.895 tie. Since TF-IDF ties already, the CPU argument favors TF-IDF as the serving model with the transformer as an offline auditor.

## 6. Cross-lingual transfer
- Ghorbanpour, Dementieva, Fraser. "Data-Efficient Hate Speech Detection via Cross-Lingual Nearest Neighbor Retrieval with Limited Labeled Data". arXiv 2505.14272, EMNLP 2025. https://arxiv.org/abs/2505.14272 . VERIFIED (abstract). Retrieve nearest labeled examples (MMR-deduplicated) from a multilingual pool, ~200 examples sometimes suffice. Apply: embed Tamil/Kannada/Malayalam/Hindi hate sets with a multilingual encoder, retrieve neighbors of the hardest Telugu errors, add as training data, ablate.

## 7. Fairness / functional tests
- Ng et al. "SEAHateCheck". arXiv 2603.16070 (Mar 2026, TALLIP). https://arxiv.org/abs/2603.16070 . VERIFIED. HateCheck-style suite extended with LLM-drafted, expert-validated cases; slang, implicit hate, counter-speech are weak. Apply: write ~15 functionalities x 10 Telugu templates (slur use/mention, negation, counter-speech, reclaimed, caste and religion targets), LLM-draft, hand-validate, run on all three models.
- HateCheckHIn (arXiv 2205.00328) and Multilingual HateCheck (arXiv 2206.09917) are 2022 precedents. UNVERIFIED (snippets only). No 2025 Telugu/caste functional-test paper found.

## Ranking (novelty x feasibility x user value)
1. Conformal abstention + HITL routing with moderator-set thresholds (3). Feasible in hours, direct user value.
2. Cleanlab-style label audit with out-of-fold probs plus LLM second opinion (1). The only lever on a data-limited ceiling.
3. HateCheck-style Telugu functional + slice-wise conformal set-size bias audit (7+3). Genuinely novel for Telugu.
4. Occlusion n-gram rationales in the UI (4). Cheap, moderator-visible.
5. Continued MLM pretraining + cross-lingual kNN retrieval augmentation (1,6). Costly, uncertain gain.

## Design
Data: unlabeled Telugu comments -> continued MLM; out-of-fold TF-IDF probs -> cleanlab flags -> local LLM second opinion -> human audit of flagged rows -> versioned "clean-v2" labels and a hand-verified gold slice.
Training: keep TF-IDF+SVC and MuRIL+LoRA+; add translit/translation views; ensemble; distill into the INT8/TF-IDF student for CPU.
Policy: split-conformal on the ensemble, alpha chosen by moderator, outputs {hate, not-hate, review}; per-slice coverage reported.
UI (app.py): verdict, conformal status, occlusion-highlighted tokens, a one-click label correction that appends to an audit queue feeding the next cleanlab pass.
Eval gate: HateCheck-Telugu pass rates and per-slice set-size before shipping any model.
Caveat: all findings come from abstracts only; none of the claimed gains were reproduced here.
