# LLM-based toxicity / hate / abuse detection for Indian languages (Telugu focus, 2025 to Oct 2026)

Scope note: about 14 tool calls. Most numbers come from full-page fetches of arXiv HTML, but those fetches were summarized by an intermediary model, so table values should be spot-checked against the PDF before any quote. Search-only items are marked "(snippet only, unverified)". Items I could not check are in Gaps.

## Do LLMs beat fine-tuned encoders on Telugu/Indic abuse detection?

### Takeaway
On the only Telugu head-to-head I could read in table form (X-MuTeST, AAAI 2026), zero-shot prompted LLMs (GPT-4o, Llama-3.1, Mistral-24.07) trail fine-tuned MuRIL and XLM-R by about 0.2 macro-F1 on Telugu, Hindi and English. Nothing I found shows a fine-tuned LLM beating a fine-tuned Indic encoder on Telugu. The evidence is one paper on a small dataset with no seeds reported, plus qualitative abstracts.

### Cited Findings
- X-MuTeST (arXiv 2601.03194, submitted 6 Jan 2026, AAAI 2026) uses Telugu data from HOLD-Telugu (DravidianLangTech 2024, YouTube comments): 4,492 records (2,131 hate / 2,361 non-hate). The original is Latin script; the paper transliterates it to Telugu script with IndicXlit. Hindi is HASOC 2020/2021 (6,004 records); English is HASOC (6,334). Source: [arXiv HTML](https://arxiv.org/html/2601.03194) (extracted via WebFetch; spot-check).
- Telugu, Table 3 (macro-F1 / accuracy as reported): MuRIL-Rationale-XMuTeST 0.8733 / 0.8740; MuRIL-XMuTeST (no rationale) 0.8591 / 0.8598; MuRIL-LIME 0.8591 / 0.8598; XLMR-Rationale-LIME 0.8450 / 0.8455; BERT-ILP 0.8221 / 0.8213; BERT-HateXplain-LIME 0.8084 / 0.8089; zero-shot GPT-4o 0.6417 / 0.6484; zero-shot Llama-3.1 0.6319 / 0.6362; zero-shot Mistral-24.07 0.5895 / 0.6402. Source: [arXiv HTML](https://arxiv.org/html/2601.03194).
- Hindi, Table 4: fine-tuned MuRIL-Rationale-XMuTeST 0.8217 macro-F1; XLM-R-Rationale-XMuTeST 0.8456; zero-shot GPT-4o 0.5542; Llama-3.1 0.5049; Mistral-24.07 0.7099. Source: [arXiv HTML](https://arxiv.org/html/2601.03194).
- English, Table 5: fine-tuned MuRIL-Rationale-XMuTeST 0.8385; zero-shot GPT-4o 0.7633; Llama-3.1 0.7362; Mistral 0.7643. Source: [arXiv HTML](https://arxiv.org/html/2601.03194).
- The paper's own proposed system, "X-MuTeST with LLM," scores 0.8849 (Telugu) and 0.8623 (Hindi) macro-F1. Its architecture, and exactly how the LLM is used, were not confirmed from my fetch. Treat it as a proposed method, not an LLM-only classifier. Source: [arXiv HTML](https://arxiv.org/html/2601.03194).
- The paper reports Accuracy, F1, and Macro-F1 for classification. It does not report weighted-F1, and it does not state the prompt text, number of shots, or random seeds. Source: [arXiv HTML](https://arxiv.org/html/2601.03194).
- WOAH 2025 (Ghorbanpour, Dementieva, Fraser; "Can Prompting LLMs Unlock Hate Speech Detection across Languages?") says zero-shot and few-shot prompting "lag behind fine-tuned encoder models on most of the real-world evaluation sets" but generalize better on functional tests. Models named: LLaMA, Aya, Qwen, BloomZ. The abstract covers eight non-English languages, but the page I fetched did not name them, so Telugu inclusion is unverified. Source: [ACL Anthology](https://aclanthology.org/2025.woah-1.39/) (abstract only, no tables).
- Marathi (arXiv 2411.17637): GPT-4o ranked highest among GPT-4o, Gemini 1.0 Pro, Gemma 2 and Llama 3.1 as annotators, but still trailed fine-tuned BERT baselines (snippet only, unverified). Source: [arXiv](https://arxiv.org/html/2411.17637v2).
- DweshVaani (CHiPSAL 2025, religious hate in Hindi-English code-mixed text): Indic instruction-tuned LLMs fell behind, and the best model was a fine-tuned Gemma-2 (snippet only, unverified). Source: [ACL preview](https://preview.aclanthology.org/setup/2025.chipsal-1.5).
- Bengali cross-script transfer: MuRIL macro-F1 0.414 and IndicBERT 0.397 (snippet only, unverified). Encoders only, no LLM. Source: [arXiv 2204.12543 series search result](https://arxiv.org/pdf/2204.12543).

### Inferences
- Under zero-shot prompting, the gap is large (about 0.2 to 0.3 macro-F1 on Telugu and Hindi, and about 0.07 on English). It is consistent across the three languages in X-MuTeST and matches the WOAH 2025 qualitative finding.
- The X-MuTeST zero-shot LLM result is a lower bound for LLMs. Few-shot or fine-tuned LLMs were not tested in that paper, so "LLMs lose" is established only for zero-shot prompting.
- Telugu-specific LLM vs encoder evidence is one paper, one dataset, and one split. Treat the sign as likely and the size as uncertain.

### Gaps
- No fine-tuned LLM (LoRA or full) on Telugu abuse detection was found. The main question is still open for fine-tuned LLMs.
- The 2025-2026 guard models (Qwen3Guard, PolyGuard, Llama Guard 4, ShieldGemma) were not evaluated on Telugu in any source I could read. See the guard-model Gaps under the third question.
- No seeds or confidence intervals are reported in X-MuTeST.
- No table in WOAH 2025 was reachable, so no per-language numbers are available from it.

## Telugu-specific results

### Takeaway
Telugu abuse and hate results in 2025-2026 are scarce. The best-documented setup is X-MuTeST on HOLD-Telugu (about 4.5K), where MuRIL with rationale supervision reaches about 0.87 macro-F1. The DravidianLangTech 2025 shared task covered multimodal Telugu hate speech, not text-only abuse, and I could not read a ranked Telugu text result with an LLM.

### Cited Findings
- DravidianLangTech@NAACL 2025 overview, "Overview of the Shared Task on Multimodal Hate Speech Detection in Dravidian languages" (DOI 10.18653/v1/2025.dravidianlangtech-1.20), covers Tamil, Malayalam and Telugu from YouTube speech and text. The team-reported ranking figure I saw for Telugu was best F1 0.3817 (snippet only, unverified). Source: [ACL preview](https://preview.aclanthology.org/setup/2025.dravidianlangtech-1.20).
- KEC_AI_GRYFFINDOR (team paper, DravidianLangTech 2025) reports Ridge Classifier F1 0.89 for Telugu text, which is far above the overview's ranking figure. Team-reported only, not comparable to the organizers' metric (snippet only, unverified). Source: [ACL preview](https://preview.aclanthology.org/setup/2025.dravidianlangtech-1.31).
- The shared task on abusive text targeting women covered Tamil and Malayalam only, not Telugu (snippet only). Source: search results, DravidianLangTech.
- Repo baseline (local, verified from `README.md`): XLM-R-base with LoRA (r=8, alpha=64, 887k trainable parameters) on the MACD Telugu split. Macro-F1 test 0.9035 ± 0.0033 over 3 seeds. Shipped INT8 ONNX test macro-F1 0.8926. Full fine-tune of XLM-R reaches 0.9057 ± 0.0033 test. The README states that the code-mixed data is not evaluated.
- The repo's `reports/Telugu abuse detection state of art.md` says no source was found showing TF-IDF parity on Telugu, and that the MACD Telugu literature was not placed against 0.904 in published tables. Note that the repo's earlier note says MuRIL INT8 was deprecated (`onnx_model_muril_int8/`, broken per-tensor INT8). The shipped model is XLM-R, not MuRIL.

### Inferences
- X-MuTeST and the repo use different Telugu datasets (HOLD-Telugu, 4.5K, versus MACD Telugu, about 30K). Their scores (about 0.87 and about 0.90 macro-F1) cannot be compared directly.
- No 2025-2026 Telugu-specific text-only result in my reading reports a fine-tuned LLM or guard model against a fine-tuned Telugu encoder. This is the main gap for the target question.

### Gaps
- The MACD Telugu literature was not read in table form. The repo's own note says the same.
- The DravidianLangTech 2025 overview's per-team Telugu table was not read.
- I did not check whether any Telugu paper in 2025-2026 evaluates Gemma, Qwen, or Sarvam (Sarvam-M, Airavata, Krutrim) on Telugu toxicity. Searches on those names were not run.

## Guard models (multilingual and Indic) and LLM-judge/annotation studies, and cost/latency

### Takeaway
None of the guard models I could check reports Telugu with per-language numbers. PolyGuard's only Indic language is Hindi, and it does not evaluate Llama Guard 4, ShieldGemma or Qwen guards. IndicGuard (June 2026) is the most relevant recent Indic guard, and its Telugu numbers were not extractable. On cost, the repo's CPU encoder has no per-query API cost, and the LLM API price I could confirm is about $0.10 per million input tokens for Gemini 2.5 Flash-Lite (aggregator, unverified against Google's page).

### Cited Findings
- PolyGuard (arXiv 2504.04377, COLM 2025): 17 languages in PolyGuardMix (1.91M samples). Hindi is the only Indic language. Its Tables 1 to 5 aggregate across languages and datasets and report F1, Exact Match, Jaccard and recall. Section G says the datasets "do not extend to low-resource languages." Baselines are Aegis-Defensive, MD Judge, Llama Guard 2 and 3, DuoGuard, Perspective API, OpenAI Omni Moderation and Google Moderation. Llama Guard 4, ShieldGemma and Qwen guards are not evaluated. Source: [arXiv HTML](https://arxiv.org/html/2504.04377v1) (WebFetch extraction).
- IndicGuard (arXiv 2606.22841, dated 2026-06-23 by file metadata; authors Bramhecha, Deshmukh, Bodhale, Borate, Joshi of L3Cube Pune): a secondary snippet lists ten Indic languages including Telugu, a Gemma-3-4B-IT base, and a claim of beating CultureGuard. The PDF fetch returned compressed binary and no tables were readable, so these claims are unverified. Sources: [arXiv PDF](https://arxiv.org/pdf/2606.22841); [Hugging Face model](https://huggingface.co/l3cube-pune/IndicGuard); [Hugging Face dataset](https://huggingface.co/datasets/l3cube-pune/IndicGuard).
- Qwen3Guard (arXiv 2510.14276, Oct 2025): Generative and Stream variants in 0.6B, 4B and 8B. The report claims coverage of 119 languages and dialects, with RTP-LX and PolyGuard used for multilingual evaluation. Per-language numbers were not read. This is from a secondary summary, so treat it as unverified. Source: [arXiv abstract](https://arxiv.org/abs/2510.14276v1).
- Llama Guard 4 (12B): language list not confirmed from the official card. A third-party aggregator lists Meta's 12-language Llama 4 set (includes Hindi, not Telugu), but attributes it to the base models, not Guard 4 (unverified). Source: [Meta model card](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama-guard-4/) (not fetched).
- ShieldGemma: not searched.
- LLM-as-annotator (Bangla, arXiv 2602.16241): 17 LLMs zero-shot as Bangla hate-speech annotators; reports annotator bias and instability, and larger models did not reliably annotate better than smaller task-aligned ones (snippet only, unverified). Source: [arXiv](https://arxiv.org/abs/2602.16241).
- Gemini 2.5 Flash-Lite pricing: $0.10 per million input tokens, $0.40 per million output tokens, cached input about $0.01 per million (third-party aggregators; not checked against Google's pricing page). Sources: [anotherwrapper.com](https://anotherwrapper.com/llm-pricing/gemini-2.5-flash-lite); [pricepertoken.com](https://pricepertoken.com/pricing-page/model/google-gemini-2.5-flash-lite).
- Repo CPU serving: README says inference runs on CPU; latency was measured only on the development PC and not on a low-end laptop. The exact ms figure was not extracted in this pass (see Gaps).

### Inferences
- The guard models are built for English-first and high-resource multilingual safety, and Telugu coverage is not demonstrated in any of them. A Telugu-specific abuse detector will not inherit guard-model quality for free.
- Per query, the encoder is effectively free after startup, while an API LLM costs per token and adds network latency. At the Gemini 2.5 Flash-Lite list price, the cost of classifying 1M short comments is roughly on the order of tens of dollars (input-only estimate, assuming about 30 tokens per comment). This is arithmetic from the list price, not a measured figure.

### Gaps
- No Telugu per-language numbers for PolyGuard, Qwen3Guard, Llama Guard 4 or ShieldGemma were retrieved. Stated as a finding: no guard model I could read reports Telugu.
- IndicGuard's language list, base model and Telugu scores were not extractable from the PDF. Needs a re-fetch of the arXiv HTML version.
- Usable LLM-as-judge vs human agreement on Telugu was not found. The closest evidence is Marathi (snippet) and Bangla (snippet).
- The repo's measured INT8 CPU latency (ms per comment) was not extracted; check `docs/` or `results/final_best/` in the repo.
- GPT-4o-mini and Claude/Gemini list prices were not confirmed in this pass. Gemini figures come from aggregators.
- Sarvam, Airavata, Krutrim, Gemma 3, Llama 4 and Qwen 3 on Telugu toxicity were not searched.
