# Model & data shortlist (web-verified, Oct 2026)

Only items read directly from the HF / arXiv / ACL page are marked VERIFIED.

## Encoders
| Model | Params | License | Telugu | Max len | Status |
|---|---|---|---|---|---|
| jhu-clsp/mmBERT-base | 307M (110M non-embedding) | MIT | `tel` in card metadata only | 8192 | exists; Telugu unconfirmed in text |
| ai4bharat/IndicBERTv2-MLM-only | 278M | MIT | "23 Indic languages", Telugu not named | n/s | exists; Telugu unconfirmed in text |
| google/muril-base-cased | BERT-base | Apache-2.0 | yes (17 langs) | 512 | VERIFIED |
| l3cube-pune/telugu-bert | n/c | n/c | Telugu-only | n/c | exists (search only) |

No Telugu/multi-Indic ModernBERT found; mmBERT is the only ModernBERT-style candidate.

## Small LLMs
| Model | Params | License | Telugu | Status |
|---|---|---|---|---|
| sarvamai/sarvam-1 | 2B (card) / 3B (safetensors) | non-commercial | yes | VERIFIED |
| google/gemma-3-1b-it | 1B | Gemma terms, gated | not named | exists |
| Qwen/Qwen3-1.7B | 1.7B | Apache-2.0 | not named | exists |

QLoRA fit on 8 GB is an UNVERIFIED estimate (Sarvam-1 at 3B would be tight).

## Data
- HOLD-Telugu (2024.dravidianlangtech-1.8): exists, 4,000 train / 500 test per a participant paper; NO verified download link or license.
- Telugu-English abusive (arXiv 2504.21026): 2,000 Roman-script comments; footnote repo github.com/Ranveer098/Code-Mix-and-Code-Switch (liveness unchecked); research use only.
- No HF-hosted Telugu hate dataset found.

## Citations spot-check
All six exist at their stated IDs. Caveats: arXiv 2602.11181 is titled "Code Mixologist: A Practitioner's Guide to Building Code-Mixed LLMs" (RESEARCH_REVIEW.md P17 calls it "Playbook" with different authors); 2606.18852 is about implicit hate generally, not Telugu; 2607.23319 and 2606.18852 only softly confirmed. Author/affiliation/result claims in RESEARCH_REVIEW.md were not checked against the papers.
