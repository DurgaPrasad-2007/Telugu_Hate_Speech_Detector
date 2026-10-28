# Adversarial Obfuscation and Robustness for Indic / Telugu Toxicity Detection, and Input Normalization Defenses

Research window: up to October 2026. Status tags: [verified] = primary page or full text checked with WebFetch; [snippet] = seen only in search result excerpts, not opened; [unverified] = secondary or self-reported, not checked against the paper.

---

## Q1. Is rule-based canonicalization before inference an established defense or novel? Evidence of effectiveness?

### Takeaway
Canonicalization is not novel. Normalization-style defenses against character attacks have been published since at least 2019 (learned denoiser) and 2022 (rule-based Adversarial Text Normalizer, NAACL industry). The evidence for generic character attacks is reasonable, but it covers only the perturbation classes the normalizer was built for. I found no published evaluation of canonicalization on Telugu or Romanized Telugu toxicity data.

### Cited Findings
- Kurita et al. (2019) show realistic character attacks (character scrambling, homoglyph substitution, dictionary-based near-neighbor replacement) plus distractor injection cut recall of toxicity classifiers substantially. Table 2 on the Jigsaw 2018 and OffensEval 2019 test sets: Logistic Regression 0.822 to 0.344 (-58.2%) on Jigsaw; BERT 0.914 to 0.597 (-34.7%) on Jigsaw and 0.721 to 0.342 (-52.6%) on OffensEval. [verified] — https://arxiv.org/pdf/1912.06872 (ar5iv HTML: https://ar5iv.labs.arxiv.org/html/1912.06872)
  - Note: Table 4 in the same paper lists BERT Jigsaw C+D recall as 0.604, not 0.597. The two tables disagree slightly. [verified]
- Kurita et al. defend with a Contextual Denoising Autoencoder (CDAE), a learned character-level denoiser, not a rule table. With adversarial training, CDAE recall is 0.733 (Jigsaw) and 0.596 (OffensEval) under C+D attacks. Character-only attacks cut CDAE recall to 0.642 (Jigsaw, -28.1%) and 0.354 (OffensEval, -52.2%). Distractors alone cut it by 9.4% and 4.5%. [verified] — https://ar5iv.labs.arxiv.org/html/1912.06872
- Bitton, Pavlova, Evtimov (2022) propose the Adversarial Text Normalizer (ATN), a lightweight, model-agnostic preprocessing unit that undoes character-level manipulations with no retraining. Evaluated on hate speech detection and NLI. The abstract claims it "restores baseline performance on attacked content." [verified abstract] — https://arxiv.org/abs/2206.04137 ; NAACL industry track: https://aclanthology.org/2022.naacl-industry.30.pdf
  - The abstract does not give before/after numbers. The full PDF was not checked, so no figures are cited here. [unverified]
  - Search snippet claims a red-team finding that attack types outside the ATN rules (e.g., replace_similar_chars, simulate_typos) drive the performance gap. Not verified against the paper. [snippet]
- A 2025 paper on special-character attacks against open LLMs recommends pre-tokenization normalization (NFC/NFKC, zero-width removal, homoglyph mapping) as a first line of defense. [snippet] — https://arxiv.org/html/2508.14070v1
- Plain Unicode normalization is weak against homoglyphs. A benchmark reported in translit documentation (titled "Fire Extinguishers Full of Gasoline", eight preprocessing tools, six attack types, three tasks including toxicity) reports that NFC, NFKC, NFKD, and casefold give no protection against homoglyphs and negligible protection against the rest. TR39 confusable mapping recovered the tested homoglyph pairs. [unverified: secondary summary only; original paper not located] — https://translit.readthedocs.io/en/latest/security/adversarial-defense.html
- Mimecast's guidance says NFKC handles compatibility characters (fullwidth, Roman numerals) but not cross-script look-alikes such as Cyrillic letters, which need UTS #39 mapping. [snippet] — https://www.mimecast.com/blog/5-keystrokes-that-defeat-a-privacy-filter/
- Zero-width characters and bidi overrides are not removed by character normalization alone and need explicit stripping. [snippet, consistent with the translit summary above]
- Kahu & Ahuja (2025, "All You Need is Leet") report a composite underscore plus leetspeak attack evaded Perspective API and HateSonar for 86.8% of hateful texts. Zero-width insertion was imperceptible but produced large edit distances. [snippet; abstract not opened] — https://arxiv.org/abs/2505.16263
- A 2025 survey (arXiv 2512.17367) reports a visually-confusable-glyph and zero-width attack dropped an ALBERT hate-speech classifier from 91.8% to 13.1% accuracy. [snippet; not opened] — https://arxiv.org/pdf/2512.17367
- Korean obfuscation work (KOTOX, arXiv 2510.10961) releases rule-based obfuscation transformations derived from real examples, and reports that training on the dataset improves robustness to obfuscated toxic text without hurting clean-text performance. No numbers in the abstract. [verified abstract] — https://arxiv.org/abs/2510.10961

### Inferences
- Rule-based canonicalization is an established pattern (ATN, normalization in LLM defense guidance), not a new idea. What is less established is a systematic comparison of rules versus learned denoisers versus adversarial training on the same Indic data.
- The strongest evidence is for homoglyph, zero-width, and leetspeak classes. Canonicalization is much less clear for semantic or phonetic changes, such as a word spelled differently in Romanized Telugu, where no fixed character rule can cover all variants.
- For a Telugu-script pipeline, the main risk is the false-positive side. Mapping look-alike characters and folding case changes legitimate Telugu text, and Telugu script has many combining marks that naive folding would damage. This is an inference from general normalization behavior, not a tested result.

### Gaps
- No paper found that applies rule-based canonicalization to Telugu, Hindi, or other Indic toxicity classifiers and reports before/after numbers. [gap]
- The ATN and "Fire Extinguishers" numbers could not be retrieved from primary sources. [gap]
- No paper found that measures canonicalization false-positive cost on Indic scripts. [gap]

---

## Q2. What perturbation suites exist for Indic languages? Attacks on Indic classifiers?

### Takeaway
Attack and perturbation work for Indic languages exists but is thin, and almost none of it targets toxicity classifiers. The closest work is: AdvCodeMix (sentiment, code-mixed Bengali/Hindi English), Ghosh et al. (NAACL Findings 2025, linguistically grounded phonological and orthographic attacks across 12 Indic languages), IndicSentEval (probing plus 13 perturbation types), and a 2026 Romanization-variation benchmark covering Bengali, Hindi, Tamil, and Telugu. No Telugu-specific adversarial hate-speech benchmark was found.

### Cited Findings
- AdvCodeMix (Das et al., CODS-COMAD 2022; arXiv 2111.00350) is a black-box attack on code-mixed sentiment classifiers. The attack perturbs tokens chosen by importance and keeps sentence structure intact. F1 drops are about 51% on Bengali-English models and 53% on Hindi-English models, and the authors say larger drops are possible with more perturbed tokens. [verified] — https://arxiv.org/abs/2111.00350 ; https://arxiv.org/pdf/2111.00350
  - Conflict: a search snippet describes AdvCodeMix as using phonetic and word-to-transliteration perturbations. The fetched abstract page describes perturbations generally and does not confirm phonetic or transliteration methods. Treat the phonetic claim as unverified until the full PDF is read. [conflict]
  - AdvCodeMix is a sentiment task, not toxicity. Its transfer to toxicity classifiers is untested in the sources found.
- Ghosh, Dabre, Bhattacharyya (Findings of NAACL 2025): described as the first study of adversarial robustness of PLMs for Indic languages. Builds phonological and orthographic attacks, covers 12 languages from 3 families, and 3 tasks. The abstract says PLMs are "slightly less susceptible" to linguistic attacks than non-linguistic attacks, and that "even constrained attacks are effective." [verified abstract; coverage details from search snippet] — https://aclanthology.org/2025.findings-naacl.468/ ; PDF: https://aclanthology.org/2025.findings-naacl.468.pdf
  - Per-model numbers were not retrieved (PDF not readable in this session). [gap]
- IndicSentEval (arXiv 2410.02611; ICON 2024, later IJCNLP-AACL 2025 Findings) is a probing benchmark of about 47K sentences across 6 Indic languages with 13 perturbation types. Snippets report universal multilingual models (InfoXLM, BLOOM, mGPT, XGLM, mT5) are more resilient to perturbations than BERT-style models (mBERT, IndicBERT, MuRIL). [snippet; not opened] — https://arxiv.org/html/2410.02611v2
- Code-mixed Hinglish paper (arXiv 2310.18155): reports phonetic-perturbation accuracy of mBERT 65.31% and IndicBERT 49% on a sentiment task, and that proposed pretraining reduces the drop. [snippet; not opened] — https://arxiv.org/pdf/2310.18155
- Romanization-variation benchmark (Singh, ACL MRL 2026, Budapest, Oct 2026, pp. 366-376): argues many-to-one Indic romanization means single-canonical-transliteration evaluations overestimate robustness. Builds a 4,000-set benchmark over Bengali, Hindi, Tamil, and Telugu with native-script, human-Romanized, and human-attested equivalence views. Evaluates 12 models from 3 families. Representation distance predicts inconsistency better than token fertility (ROC-AUC 0.774 vs 0.634 on a small instruction model). [verified] — https://aclanthology.org/2026.mrl-1.29/
  - This is the most relevant recent work for Telugu-specific perturbation. It is a representation-stability study, not a toxicity classifier evaluation.
- MulTypo (ACL 2026, acl-long.729): a multilingual typo generator that simulates human typing errors with language-specific keyboard layouts. Reports that instruction tuning can increase brittleness under noise and that low-resource languages are less robust. Code: https://github.com/cisnlp/multypo [snippet]. Fetch of the ACL page returned 404; whether Telugu or other Indic languages are covered was NOT confirmed. [unverified]
- Transliteration benchmarking (arXiv 2505.19851) adds manual noise to romanized words, using 100 common words from each of 10 Indian languages, to test transliteration under noisy input. [snippet] — https://arxiv.org/pdf/2505.19851
- Telugu-English abusive-content dataset (arXiv 2504.21026, Nepali-English and Telugu-English sets, 2,000 annotated Telugu-English comments): notes that Romanized Telugu has inconsistent spelling. It does not run adversarial tests. [snippet] — https://arxiv.org/pdf/2504.21026
- Telugu code-mixed hate-speech work (Kakarla & Venkata, arXiv 2502.10632, Feb 2025): fine-tunes transformers with LoRA and compares original-Telugu training against machine-translated English. Does not test adversarial inputs. [snippet] — https://arxiv.org/pdf/2502.10632
- Hindi/Tamil: HateCheckHIn (arXiv 2205.00328) proposes functional tests for Hindi hate-speech models; this is closer to robustness testing than most work found [snippet]. The 2025 Tamil and Malayalam DravidianLangTech result (Macro-F1 0.7039 Tamil) is a standard classifier, not adversarial. [snippet]. No Hindi or Tamil attack paper on toxicity classifiers was found. [gap]
- IndicJR (arXiv 2602.16832, Feb 2026; EACL 2026 Industry): a judge-free jailbreak benchmark over 12 Indic and South Asian languages that addresses script mixing and romanization. Jailbreak robustness, not toxicity classification. [snippet]. https://arxiv.org/abs/2602.16832
- HinTox (Hugging Face model card): a Hinglish abuse detector that claims robustness to misspellings, leetspeak, and obfuscation. Model-card claim only; no independent evaluation. [unverified] — https://huggingface.co/Keshav0av/HinTox

### Inferences
- The strongest Indic perturbation suites are Ghosh et al. (linguistic, NAACL 2025) and IndicSentEval (probing, 13 perturbations). Both are general-purpose, not toxicity-specific. A toxicity-specific Telugu suite would need to combine homoglyph/zero-width/leetspeak classes (from Kurita and Kahu & Ahuja) with the Telugu phonetic and Romanization variation classes (from Ghosh, IndicSentEval, and the 2026 Romanization benchmark). This combination does not appear to exist.
- Romanized Telugu variation is probably the highest-impact vector for a Telugu-English hate-speech model, because it is common in real data and no fixed character rule covers it. This is an inference, not a measured result.

### Gaps
- No Telugu-specific adversarial hate-speech benchmark found. [gap]
- Ghosh et al. per-language and per-model numbers not retrieved. [gap]
- MulTypo's language list not confirmed. [gap]
- No published IndicBERT versus MuRIL versus XLM-R robustness numbers on a toxicity task. [gap]

---

## Q3. Defenses beyond canonicalization: adversarial training, character-aware models (ByT5/CANINE), transliteration-first pipelines

### Takeaway
The evidence for character-aware and denoising defenses is real but comes from English. Byte-level models (ByT5) are reported to handle noise better than subword models on general tasks. The one toxicity result for character-level denoising (Kurita et al.) favors CDAE over BERT on character attacks, but BERT is stronger against distractors. I found no CANINE toxicity-robustness result and no ByT5 toxicity-robustness result on Indic data.

### Cited Findings
- ByT5 paper (Xue et al., 2021; arXiv 2105.13626): reports ByT5 outperforms mT5 across noising schemes, and is robust to uppercase and repetitions. General tasks, not toxicity. [snippet] — https://arxiv.org/pdf/2105.13626
- MANTa (arXiv 2212.07284) reports a ToxicComments accuracy table: T5-base 91.5, Charformer-base 92.7, MANTa-LM base 93.2. This is clean-data accuracy, not adversarial robustness. The numbers cite Tay et al. (2021) for the first two. [snippet] — https://arxiv.org/pdf/2212.07284
- Kurita et al. (2019): adversarial training on synthetic noise plus CDAE is more resilient to character perturbations than BERT; BERT is stronger on distractors. [verified, see Q1]
- Bespalov et al. (ACL Industry 2023, "Towards Building a Robust Toxicity Predictor"; arXiv 2404.08690 is a later version): reports SOTA toxicity classifiers are vulnerable, with attack success rates above 98% in multilabel cases. [snippet] — https://arxiv.org/pdf/2404.08690 ; https://aclanthology.org/2023.acl-industry.56
- Adversarial retraining is described as better suited to semantic changes, while normalization is task-agnostic and complementary (ATN abstract). [verified abstract] — https://arxiv.org/abs/2206.04137
- MaskPure (arXiv 2406.13066) proposes stochastic purification as an alternative to deterministic normalization, tested against DeepWordBug and TextFooler. [snippet] — https://arxiv.org/pdf/2406.13066
- Transliteration-first pipelines: a 2024 DravidianLangTech Telugu code-mixed study found the transliteration-augmented method handles linguistic variation best, and a related 2024 shared-task paper reports F1 0.77 with Telugu-BERT on augmented data. [snippet; attribution of the 0.77 figure not confirmed] — https://aclanthology.org/2024.dravidianlangtech-1.28 ; https://aclanthology.org/2024.dravidianlangtech-1.23
  - A 2026 workshop paper on hybrid transformers for Telugu-English abusive content says it uses preprocessing to normalize transliterations and spellings. [snippet] — https://aclanthology.org/2026.dravidianlangtech-1/
  - These are clean-data gains from augmentation, not adversarial robustness results.

### Inferences
- Byte-level and character-level models are a plausible defense for Romanized Telugu spelling variation (no fixed token vocabulary), but no result on Indic toxicity supports this yet. Treat as a hypothesis to test.
- Transliteration-first pipelines (Romanize or transliterate to a single script, then classify) merge spelling variants, which helps spelling variation. But the Romanization-variation work above suggests a single canonical transliteration can overstate robustness, so a transliteration-first pipeline needs its own robustness test.
- Adversarial training with homoglyph, leetspeak, and Romanization-variation augmentation looks like the most direct defense, but no Telugu study with this setup was found.

### Gaps
- No CANINE result on toxicity robustness found. [gap]
- No ByT5 result on Indic toxicity or adversarial robustness found. [gap]
- No direct comparison of canonicalization versus adversarial training versus byte-level models on the same Indic toxicity data. [gap]

---

## Q4. 2025-2026 developments

### Takeaway
2025-2026 work is mostly on general LLM and English-language attacks (leetspeak, zero-width, homoglyph). Indic-specific 2026 work is on Romanization variation (MRL 2026), typo robustness (MulTypo, ACL 2026, language coverage unconfirmed), and jailbreak benchmarks (IndicJR, Feb 2026). I found no 2026 paper on adversarial robustness of Telugu toxicity classifiers.

### Cited Findings
- Kahu & Ahuja, "All You Need is Leet" (arXiv 2505.16263, May 2025; Virginia Tech): black-box, 86.8% evasion of Perspective and HateSonar with the composite attack. [snippet] — https://arxiv.org/abs/2505.16263
- Survey, arXiv 2512.17367 (2025): ALBERT 91.8% to 13.1% under glyph and zero-width attack. [snippet] — https://arxiv.org/pdf/2512.17367
- Romanization-variation benchmark (MRL 2026): [verified, see Q2]
- MulTypo (ACL 2026): [snippet, see Q2]
- IndicJR (Feb 2026): [snippet, see Q2]
- KOTOX (2025): [verified abstract, see Q1]
- Special-character attacks on LLMs (arXiv 2508.14070, 2025): recommends normalization as first line of defense. [snippet]
- A 2025 arXiv paper on toxicity in otome game communities (arXiv 2609.08009) reports about 40% of toxic spans on Weibo use variation strategies, that evasive posts get more engagement, and proposes layered preprocessing (confusable mapping, Pinyin handling, emoji substitution). It estimates a 19.8% false-negative rate on Weibo. The modules are proposed, not evaluated. [snippet; ID and date not checked] — https://arxiv.org/pdf/2609.08009

### Inferences
- The field is moving from "does this attack work" toward "does normalization cover the attack" and "does the evaluation use one canonical spelling." The Romanization-variation paper is the clearest example.
- Indic-specific attack work in 2026 is about LLMs and Romanization, not about classic toxicity classifiers. That leaves a gap for Telugu toxicity.

### Gaps
- No 2026 Telugu toxicity adversarial benchmark found. [gap]
- Searches did not return results from after October 2026. Coverage is limited to what the search index returned.

---

## Summary Answers to Key Questions

1. Rule-based canonicalization: established as a family (ATN 2022, normalization in LLM defense guidance) but not as an Indic-specific result. It covers homoglyph, zero-width, and leetspeak attacks well in the sources found. It does not cover semantic or Romanization variation. No Telugu before/after evidence exists in the sources found. Novelty claim for a Telugu pipeline would be a contribution, not a re-check of an existing result.
2. Perturbation suites for Indic: Ghosh et al. (NAACL 2025 Findings), IndicSentEval (ICON 2024 / AACL 2025), Romanization-variation benchmark (MRL 2026), MulTypo (ACL 2026, coverage unconfirmed), HateCheckHIn (Hindi functional tests). No Telugu toxicity suite found.
3. 2025-2026: Leetspeak evasion (2025), Romanization variation (2026), typo robustness (2026), Indic jailbreak robustness (2026). No 2026 Telugu toxicity adversarial result found.

## Unverified Items to Check Before Citing
- AdvCodeMix phonetic/transliteration method (conflict between snippet and fetched abstract). Read arXiv 2111.00350 PDF.
- Ghosh et al. NAACL 2025 numbers. Read PDF at https://aclanthology.org/2025.findings-naacl.468.pdf
- ATN before/after numbers and attack taxonomy. Read PDF at https://arxiv.org/pdf/2206.04137
- "Fire Extinguishers Full of Gasoline" benchmark: original paper not located.
- MulTypo language coverage: ACL page returned 404; check https://github.com/cisnlp/multypo
- HinTox and the Bespalov et al. 2023 numbers.
- IndicSentEval per-model robustness numbers.
- Telugu-BERT 0.77 F1 attribution.
- arXiv 2609.08009 ID and date.
