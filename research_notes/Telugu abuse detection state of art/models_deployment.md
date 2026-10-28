# Indic foundation models and efficient deployment for Telugu text classification (as of Oct 2026)

Method note: built from ~9 web searches (search-engine summaries, few full-page fetches). Items marked [secondary] or [unverified] need primary-source checks.

## Which backbones are current for Telugu, and which benchmarks show the best?

### Takeaway
IndicBERT v2 (AI4Bharat) leads MuRIL and XLM-R on the IndicXTREME average for Telugu. Direct Telugu hate/abuse comparisons are thin, and the one found says abuse-tuned MuRIL beats plain MuRIL and IndicBERT. Newer options (IndicBERT-v3, mmBERT, Sarvam-30B/105B) exist but lack Telugu classification evidence.

### Cited Findings
- IndicXTREME Telugu average (preferred metric) reported as roughly IndicBERT 70.2, MuRIL 64.2, XLM-R 56.4 (IndicBERT-SS 70.1). Extracted by a search summary from the IndicBERT v2 paper table; not verified against the PDF, and a per-task Telugu breakdown was not found — [Towards Leaving No Indic Language Behind (arXiv 2212.05409)](https://arxiv.org/pdf/2212.05409)
- IndicBERT v2 was pretrained on IndicCorp v2 (~20.9B tokens); variants: MLM-only, MLM+SAM-TLM, SS, MLM-Back-TLM; collection last updated Oct 2024 — [HF IndicBERT v2 collection](https://huggingface.co/collections/ai4bharat/indicbert-v2)
- AI4Bharat HF org lists IndicBERT-v3 (270M, 1B, 4B encoders) and IndicTrans3-beta; no release date or benchmarks found — [AI4Bharat HF](https://huggingface.co/ai4bharat)
- Telugu hate speech paper (code-mixed Telugu-English, arXiv 2025): Hindi-Abusive-MuRIL best (~0.73 F1/acc on translated data), plain MuRIL ~0.42, IndicBERT lowest ~0.37; abuse-domain pretrained models transfer better. Single paper, peer-review status unchecked, and the scores look low/odd for plain MuRIL — [arXiv 2502.10632](https://arxiv.org/pdf/2502.10632)
- Tamil caste/migration hate task LT-EDI 2025: MuRIL-large top macro-F1 0.8092 (Tamil, indirect) — [CUET_N317 @ LT-EDI 2025](https://preview.aclanthology.org/credits/2025.ltedi-1.18)
- mmBERT (ModernBERT-style multilingual encoder, 3T tokens, 1800+ languages, Gemma 2 tokenizer, ICML 2026) — [PMLR](https://proceedings.mlr.press/v306/marone26a.html), [HF blog](https://huggingface.co/blog/mmbert)
- Sarvam-30B (sparse MoE, ~2.4B active non-embedding) and Sarvam-105B announced 18 Feb 2026, Apache 2.0, weights on HF/AIKosh in early March 2026, tokenizer for 22 languages [secondary] — [OpenSourceForU](https://www.opensourceforu.com/2026/03/sarvam-releases-30b-and-105b-llms-under-apache-2-0/); adoption friction (missing tooling/formats) reported — [Inc42](https://inc42.com/?p=551736)
- Gemma 3 IndicGenBench avg: 1B 41.4, 4B 57.2, 12B 61.7, 27B 63.4 (generation, not classification) — [Gemma 3 report](https://arxiv.org/pdf/2503.19786)
- IndicIFEval (14 Indic languages): Qwen3-14B ~70.0, Qwen3-32B ~66.0, Gemma-3-27B-IT ~63.3 (instruction following; Telugu column not confirmed) — [arXiv 2602.22125](https://arxiv.org/pdf/2602.22125)
- LLM-vs-specialized hate detection question is raised in Hindi-English work (IndoHateMix) — [arXiv 2506.12744](https://arxiv.org/html/2506.12744v1)

### Inferences
- For a small Telugu classifier, encoder fine-tuning (IndicBERT v2 / MuRIL / XLM-R) remains the sensible default; Sarvam/Gemma/Qwen are generative and too heavy for CPU unless heavily quantized.
- mmBERT and IndicBERT-v3 are the "2026-fresh" encoders worth a baseline run, but no Telugu numbers exist in what I found.

### Gaps
- No Telugu per-task IndicXTREME table verified; no head-to-head LLM vs fine-tuned encoder on Telugu classification; no mmBERT or IndicBERT-v3 Telugu scores; no confirmed IndicLID 2025+ update; Sarvam claims are from secondary sources.

## Is XLM-R-base competitive vs MuRIL and IndicBERT v2 for Telugu?

### Takeaway
On the IndicXTREME average, XLM-R is the weakest of the three for Telugu (as reported above), but it is not uncompetitive on all measures; evidence for hate/abuse specifically is lacking.

### Cited Findings
- IndicXTREME Telugu avg XLM-R ~56.4 vs MuRIL ~64.2 vs IndicBERT ~70.2 — [arXiv 2212.05409](https://arxiv.org/pdf/2212.05409) (see caveat above; size class of XLM-R in the table unverified)
- Telugu perplexity on RESPIN transcripts: base MuRIL 62.98, fine-tuned MuRIL 15.41, XLM-R 12.07 (lower better); model-card, not peer reviewed, and not a classification metric — [HF RESPIN_LanguageModels](https://huggingface.co/SpireLab/RESPIN_LanguageModels/blob/main/README.md)
- A 2024 Telugu code-mixed offensive-language shared-task system evaluated MuRIL and XLM-R; scores not retrieved — [DravidianLangTech 2024](https://preview.aclanthology.org/setup/2024.dravidianlangtech-1.32)

### Inferences
- XLM-R-base is a defensible, widely supported baseline (esp. for Romanized/code-mixed text, given its broad CC100 data - general knowledge, not sourced), but expect it to trail IndicBERT v2/MuRIL on native-script Telugu; justify it with own-dataset results.

### Gaps
- No direct XLM-R vs MuRIL vs IndicBERT v2 Telugu hate/abuse benchmark found.

## Current CPU/edge deployment practice; is LoRA + INT8 ONNX FastAPI typical or advanced?

### Takeaway
ONNX Runtime officially recommends dynamic quantization for transformers, so INT8 ONNX on CPU is mainstream. I found no sources on the prevalence of LoRA-merge + ONNX services, SetFit, or GGUF for Indic classifiers.

### Cited Findings
- ORT docs: dynamic quantization recommended for RNN/transformer models, static for CNNs; API `quantize_dynamic`; CPU supports U8U8, U8S8, S8S8; no retraining support — [ORT quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- Sentence-Transformers ONNX dynamic quantization presets (e.g. avx2) should match deployment CPU family [third-party] — [simplified.guide](https://www.simplified.guide/sentence-transformers/onnx-model-quantize)
- Attention in BERT can be sensitive to quantization; validate accuracy [anecdotal blog] — [dev.to](https://dev.to/shalinibhavi525sudo/the-engineers-toolkit-how-to-shrink-and-accelerate-transformer-models-for-edge-ai-o9d)

### Inferences
- INT8 dynamic ONNX is standard; adding LoRA is a training-time choice (merge before export, so serving looks the same) and is moderately common practice - unsourced. Calling the stack "advanced" overstates it; "current standard, well-executed" is safer. Large vocab models (MuRIL ~197k, XLM-R 250k) are embedding-heavy, so INT8 gains are mostly on the encoder layers - unsourced inference.

### Gaps
- No sources on distillation, SetFit, GGUF/llama.cpp for Telugu classification, or Sarvam GGUF availability (Inc42 hints tooling/formats lacking).

## Production moderation patterns: thresholds, human-in-the-loop, canonicalization

### Takeaway
Two-threshold (allow/review/block) routing with calibration and human review is the established pattern in the literature; Indic-specific obfuscation-robustness work exists but is sparse.

### Cited Findings
- Double-threshold policy: auto-accept / human review / auto-reject, with diminishing returns from extra review — [arXiv 2601.05974](https://arxiv.org/html/2601.05974v1)
- Learning to defer in content moderation: static thresholds ignore model uncertainty, reviewer capacity and selection bias — [arXiv 2402.12237](https://arxiv.org/html/2402.12237v3)
- Practitioner guide: starting auto-accept ~0.90 (vendor rule of thumb, not validated); tune on frozen holdout, shadow-mode threshold changes, monitor reviewer reversals/appeals — [Transloadit](https://transloadit.com/guides/ai-moderation-systems/)
- Calibration (temperature scaling) and confidence-based abstention reduce selective risk in LLM moderation guards — [arXiv 2601.05974v2](https://arxiv.org/pdf/2601.05974v2) (attribution of this finding to that exact paper is from a search summary; verify)
- AdvCodeMix: phonetic/transliteration-swap attacks on code-mixed text (sentiment, not hate) — [arXiv 2111.00350](https://arxiv.org/pdf/2111.00350)
- MoH pipeline: language ID + Roman-to-Devanagari transliteration before mBERT/MuRIL for Hinglish hate — [MoH](https://deepai.org/publication/ceasing-hate-withmoh-hate-speech-detection-in-hindi-english-code-switched-language)
- 2026 DravidianLangTech Telugu-English abuse work notes obfuscation via spelling variants and Romanization; a 400-example adversarial benchmark paper exists — [DravidianLangTech 2026](https://aclanthology.org/2026.dravidianlangtech-1/)
- AI4Bharat IndicXlit (transliteration) available via AIKosh — [AIKosh](https://aikosh.indiaai.gov.in/home/models/details/indicxlit.html)

### Inferences
- Canonicalization (Unicode NFC/NFKC, strip zero-width chars, leetspeak/homoglyph mapping, optional Roman-to-Telugu transliteration via IndicXlit) fits the literature, but no published guidance or standard specifically for Telugu was found; zero-width/leetspeak handling is unsourced engineering practice.

### Gaps
- No Telugu- or Indic-specific published guidance on calibrated thresholds; no paper combining obfuscation attacks with multi-lingual Indic hate moderation; no validated threshold values.
