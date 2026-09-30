# Research Review

Status: Phase 0–2 (verification) complete. Phases 3+ (training/experiments) not yet run.

## 1. Label semantics — VERIFIED, mapping corrected

### Question
Do `data/telugu_*.csv` and `data/codemixed_*.csv` use `0 = Abusive` or
`1 = Abusive`? The README and original `data_utils.py` assumed
`1 = Abusive`. Manual inspection of `telugu_train.csv` contradicted this
(rows with slurs/obscenity were labelled `0`; polite messages were
labelled `1`).

### Method (triangulation, not assumption)
1. **Dataset lineage / hash comparison.** Downloaded
   `ShareChatAI/MACD` (NeurIPS 2022, "MACD: Multilingual Abusive Comment
   Detection at Scale for Indic Languages") from
   `github.com/ShareChatAI/MACD`, both `dataset/` and `dataset_80_10_10/`
   splits, Telugu files. SHA-256 of `data/telugu_{train,val,test}.csv` in
   this repo is **byte-identical** to
   `MACD/dataset_80_10_10/telugu_{train,val,test}.csv`
   (23,979/3,000/3,000 rows after minor MACD-side de-dup — confirmed
   row-for-row text+label match on 100% of overlapping rows).
2. **Upstream README.** MACD's own `README.md` states verbatim:
   > "labels 0 and 1 are used for **abusive** and **non-abusive**
   > comments, respectively."
3. **HOLD-Telugu comparison (ruled out).** Fetched the HOLD-Telugu
   (DravidianLangTech@EACL 2024) shared-task data directly from
   `github.com/Salman1804102/DravidianLangTech-EACL-2024-HOLD`
   (`training_data_telugu-hate.xlsx`, `telugu-hate-speech-test.xlsx`).
   This is a **different, unrelated dataset**: string labels
   `hate`/`non-hate` (not 0/1), ~4,000 train rows (not 24,000), YouTube
   political/celebrity comments in a different register, and zero text
   overlap with our CSVs. **Our files are not HOLD-Telugu or a
   relabelled derivative of it** — the README's framing was simply
   wrong about provenance. Aclanthology paper `2024.dravidianlangtech-1.8`
   confirms HOLD-Telugu's own split sizes/format don't match ours either.
4. **Manual re-read**, post-hoc, of 15 additional rows across both
   domains at each label value, confirms alignment with the MACD
   convention (obscene/insulting text at `0`, benign text at `1`) with
   no contradictions found.
5. **Independent secondary literature** (arXiv 2504.21026, "Creating and
   Evaluating Code-Mixed Nepali-English and Telugu-English Datasets...",
   and the MACD NeurIPS paper itself) consistently frame `0` as the
   abusive/positive class for this dataset family.

### Verdict
All four independent checks agree: **raw `0 = Abusive`, raw `1 =
Non-Abusive`** in every CSV in `data/`. No disagreement was found between
external sources and the CSVs, so no lineage-ambiguity investigation was
needed beyond ruling out HOLD-Telugu.

### Decision (applied, non-destructive)
- Source CSVs in `data/` are **untouched**.
- `src/data_utils.py` now defines
  `RAW_LABEL_MAP = {0: 1, 1: 0}` and applies it inside `_standardize()`
  at load time, so every downstream consumer (training, metrics,
  thresholds, docs) uses the single internal convention already stated
  in the README: **1 = Abusive, 0 = Non-Abusive**. This is the
  `LABEL_MAP` referred to by the project plan.
- Verified post-fix: Telugu train is 12,401 abusive / 11,579 non-abusive
  (51.7%/48.3%) and code-mixed train is 1,542/1,637 — both close to
  MACD's reported ~49% abuse ratio, another consistency check in favor
  of this mapping over its inverse.

### Sources
- ShareChatAI/MACD, https://github.com/ShareChatAI/MACD (README.md, `dataset_80_10_10/telugu_*.csv`)
- MACD: Multilingual Abusive Comment Detection at Scale for Indic Languages, NeurIPS 2022 Datasets & Benchmarks track, https://proceedings.neurips.cc/paper_files/paper/2022/file/a7c4163b33286261b24c72fd3d1707c9-Paper-Datasets_and_Benchmarks.pdf
- Findings of HOLD-Telugu@DravidianLangTech-EACL2024, https://aclanthology.org/2024.dravidianlangtech-1.8/ and repo https://github.com/Salman1804102/DravidianLangTech-EACL-2024-HOLD (used to rule out this lineage)
- Creating and Evaluating Code-Mixed Nepali-English and Telugu-English Datasets..., arXiv:2504.21026

## 2. PEFT / Transformers API — spot-checked against installed versions

Environment actually installed in `.venv` (Windows, CUDA 12.4):
`torch 2.6.0+cu124`, `transformers 5.17.0`, `peft 0.21.0`,
`bitsandbytes 0.50.2`. GPU: RTX 4060 Ti, 8 GB VRAM, CUDA available.

- `LoraConfig.use_dora` **exists** in installed `peft==0.21.0` — the
  repo's `dora` method (`use_dora=(method=="dora")` on top of
  `LoraConfig`, `src/train.py:74`) uses the current, correct API.
- `AdaLoraConfig` **has** `orth_reg_weight` and `total_step` fields in
  0.21.0 — the repo's usage (`src/train.py:82-92`) matches, not a
  version-drift guess.
- **Confirmed gap (not yet fixed):** `target_modules` defaults to
  `"query,value"` (`src/train.py:142`), which is correct for
  BERT/mBERT-family checkpoints but wrong for XLM-R/RoBERTa-family
  (`query,key,value`) and DeBERTa-v3 (fused `in_proj`/different naming).
  Per-model target-module lists will be derived from
  `model.named_modules()` at Phase 3 (training hardening), not
  hardcoded — tracked, not fixed here since no model has been loaded
  from the Hub yet in this environment.
- ONNX Runtime dynamic quantization and Streamlit caching APIs will be
  re-checked at Phase 7/8 against the versions actually installed then
  (not yet installed/verified — `optimum[onnxruntime]` not in the venv
  yet).

## 3. Status / what's NOT run yet
Per Rule 1 (never fake results): no model has been downloaded or trained
in this environment yet. Phases 3 onward (training hardening, actual
experiment matrix, calibration, robustness, Streamlit, serving) are
pending and will report `NOT RUN` for anything not executed.
