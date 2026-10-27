# nano-research: quick review (browse capability + security)

Repo: https://github.com/karthik-ak-Git/nano-research (Rust, CDP-driven Chromium agent). Source-level skim of `src/security/*`, `browser/session.rs`, `browser/launcher.rs`, `agent/service.rs`. Not a full audit; nothing was built or run.

## Verdict
Architecture is reasonable (raw CDP, loopback-only debug port, ephemeral port, temp profile, DOM caps, untrusted-content fencing). **Nothing catastrophic, but the headline security claims ("ZERO VULNERABILITIES", "ENFORCED SSRF") are overstated.** The issues below are real and worth fixing, ordered by severity.

## Findings

### 1. SSRF guard is bypassable (HIGH) — `src/security/ssrf.rs`, `src/browser/session.rs:94-113`
- Guard only checks the **literal host string**. A domain like `evil.com` that resolves to `127.0.0.1` / `169.254.169.254`, or a DNS-rebinding host, passes. No DNS resolution check.
- **Redirects are not validated.** A public page can 302 to `http://169.254.169.254/...`. Guard runs only on the initial `navigate()`.
- Links clicked, `window.location` changes, and sub-requests (iframes/fetch) in the page never go through the guard.
- `navigate()` validates `url`, then sends the **raw original `url`** to `page_navigate`, not the normalized `safe_url`. Parser-differential risk; send `safe_url`.
- Fix: enforce at the network layer via CDP `Fetch.enable` + `Fetch.requestPaused` (check every request, including redirects, after resolving the IP), or a resolving proxy. Re-validate on `Page.frameNavigated`. Also check IPv4-mapped IPv6 (`::ffff:127.0.0.1`) and decimal/hex IP forms in `is_disallowed_*`.

### 2. LLM-controlled arbitrary JS execution (MEDIUM-HIGH) — `src/actions/executor.rs:114`
- `BrowserAction::Evaluate { script }` runs model-chosen JS in the page. Combined with prompt injection from page content, an attacker page can steer the agent to run scripts, navigate (see #1), or exfiltrate data seen in the session.
- Fix: remove `Evaluate` from the model-facing action set, or allowlist/templated scripts only. Don't reuse a logged-in browser profile.

### 3. API key leaked via URL query string (MEDIUM) — `src/agent/service.rs:~150`
- Gemini call uses `...:generateContent?key={api_key}`. Keys in URLs end up in logs, proxies, and `reqwest` error messages (which include the URL). Use the `x-goog-api-key` header instead. Make sure errors/`info!` logs never print the URL.
- Also: `reqwest::Client::new()` per call, no timeout. Reuse one client and set a timeout.

### 4. Prompt-injection defense is weak and mislabeled (MEDIUM) — `src/security/prompt_guard.rs`
- 10 hard-coded English phrases, case-insensitive substring match. Trivially bypassed (paraphrase, other languages, Unicode homoglyphs, split text). It only *tags*, never blocks.
- The "cryptographic boundary tagging" is `DefaultHasher` (SipHash, non-crypto, unkeyed) of the content itself. It provides no integrity and no unforgeable delimiter; page text containing `</untrusted_web_content>` can break out of the fence (no escaping of the closing tag).
- Fix: escape/strip the closing tag in content, use a random per-request nonce in the tag, and rely on capability limits (#1, #2) rather than phrase lists. Drop the "cryptographic" wording.

### 5. Self-audit is tautological (LOW, but misleading) — `src/security/mod.rs`
- `run_self_audit()` hardcodes status strings ("ENFORCED…", "ROBUST (ZERO VULNERABILITIES DETECTED)") and tests only trivial cases (literal `169.254.169.254`, `127.0.0.1`). It cannot detect #1. Same for README claims like "0.0% fabrication risk", "Anthropic-aligned". Remove or soften; add tests for redirects / resolving hosts / IPv6-mapped.

### 6. Misc robustness (LOW)
- `dom_guard.rs` `truncate_text` slices `&s[..max_text_len]` by bytes: **panics on a multibyte UTF-8 boundary** (Norwegian å/ø/æ text, i.e. this repo's actual target data). Use `s.char_indices()`/`floor_char_boundary`. Verified by reading code; not run.
- `intelligence/mod11.rs:16` `to_digit(10).unwrap()` — panics on non-digit input if reached unsanitized; confirm `clean` is filtered.
- `orchestration/coordinator.rs:142` `sem.acquire().await.unwrap()` — fine unless semaphore closed; use `?`.
- Launcher: ephemeral port is bound then dropped before Chrome starts (small race, local only). Debug port is loopback-bound, which is good. Check the temp profile dir is cleaned up and not world-readable; confirm no `--no-sandbox` is added elsewhere (not seen in launcher args shown).
- Heuristic fallback in `agent/service.rs` silently runs when no API key is set; make that explicit in output so results aren't mistaken for LLM-verified data.

## What's fine
- Loopback-only CDP, dynamic port, isolated `--user-data-dir`.
- Scheme allowlist (http/https), sensitive-port block, private/loopback/metadata IP block for literal IPs.
- DOM node/depth/text caps.
- Keys read from env, `.env.example` has no secrets.

## Suggested priority for Antigravity
1. Network-level SSRF enforcement (#1) + pass `safe_url`.
2. Remove/limit `Evaluate` (#2).
3. Move Gemini key to header, add client timeout (#3).
4. Fix UTF-8 panic in `truncate_text` (#6) — one-line, user-visible.
5. Escape closing tag + nonce in `encapsulate_untrusted_content`; tone down claims (#4, #5).

---

# Part 2: System design, workflow, and approach vs 2026 competitors

Based on reading `agent/{service,dual_brain,prompts,token_optimizer}.rs`, `orchestration/coordinator.rs`, `dom/*`, `specialists/*`, `budgets.rs` (~5.3k lines of Rust). Competitor info comes from third-party 2026 blog comparisons (vendor/blogger-reported, not independently verified): [Skyvern](https://www.skyvern.com/blog/browser-use-alternatives/), [dev.to comparison](https://dev.to/agdex_ai/ai-native-browser-automation-in-2026-browser-use-stagehand-steel-and-playwright-mcp-compared-160n), [alexcloudstar](https://alexcloudstar.com/blog/ai-browser-agent-stagehand-browser-use-playwright/), [devtoollab](https://devtoollab.com/blog/best-ai-browser-automation-tools).

## 1. What it actually is (system design)
Two things share one repo:
1. **Generic browser agent** (`browser/`, `cdp/`, `dom/`, `actions/`, `agent/service.rs`): raw CDP over WebSocket, DOM + DOMSnapshot perception, indexed-element prompt, JSON action loop (click/input/navigate/evaluate/done), budgets, popup watchdog. It is a small Rust port of the Browser Use design (same `thinking / evaluation_previous_goal / memory / next_goal / action` schema and `[12]` indexed elements).
2. **Norwegian company dossier pipeline** (`orchestration/coordinator.rs` + `specialists/*`): Mod-11 check -> Brreg REST API -> optional website recon via the browser -> verification -> "dual brain" -> synthesis. Most facts come from the **official registry JSON API, not browsing**. Web recon is optional and defaults to `false`.

## 2. Biggest finding: the "Dual-Brain" is not an LLM
`SingleModelDualBrain::execute_system_2_deliberative` calls four local functions (`analyze_gaps`, `reconcile_forensics`, `map_ownership`, `synthesize_narrative`) that are **hard-coded rules and `format!` templates**. There is no model call in `dual_brain.rs` (an `api_key` field is stored and never used). The "executive narrative", "forensic flags" and "confidence_score" are templated strings over registry fields. `sensory_heuristic` is digit scanning.
- The README sells this as a "Single-Model Dual-Brain cognitive architecture". It is a deterministic report formatter. **The marketing is misleading.**
- The only real LLM path is `agent/service.rs::call_llm_api` (Gemini or OpenAI-style). Claude is advertised but there is **no Anthropic Messages API path**; backend is chosen by "key starts with `AIza`", else OpenAI format. Brittle.
- With no key it silently falls back to `heuristic_decision` (find a Mod-11-valid 9-digit number in the DOM). Fine for a demo, misleading if presented as agentic.

## 3. Workflow review
Strengths:
- Registry-first grounding is the right call for accuracy; Mod-11 validation up front is cheap and good.
- Bounded concurrency (`Semaphore`, 5 workers), request/step/time budgets, per-item errors in batch results.
- Provenance ledger, plus detection of 404/captcha/"diagnostic noise" pages in `intelligence/verification.rs`.

Weaknesses:
- **Pipeline is linear.** Gap analysis emits `suggested_followup_query` strings that nothing executes. It is one-shot, not an agent that researches.
- **Web recon failures are swallowed** (`warn!` then `None`), so a dossier can look complete when browsing never ran. Mark each field `verified | registry-only | unverified`.
- No retry/backoff/429 handling for Brreg (10s timeout only); no checkpoint/resume for batches.
- Page readiness uses polling `readyState` + fixed 200ms sleeps, not CDP lifecycle/network-idle events; flaky on SPAs.
- One fresh Chrome per task (`launch_local` in the provider), no pool. The "startup <50ms" claim cannot be Chromium launch; unverified.
- Tests: ~25 small unit tests (security helpers, Mod-11). **Nothing exercises the agent loop, DOM serializer on real pages, or CDP.** No CI seen, no benchmark harness.

## 4. Approach vs 2026 competitors
| Area | nano-research | 2026 field (per sources above) |
|---|---|---|
| Page representation | DOM + DOMSnapshot, indexed pruned elements. README claims "3-tree" but no `Accessibility.getFullAXTree` call was found | Accessibility-tree snapshots (Playwright MCP, Stagehand), DOM+screenshots (Browser Use) |
| Vision | `get_screenshot_base64` exists, **never sent to a model** | Hybrid text+vision common; Anthropic computer use / Skyvern vision-first |
| Models | Gemini + generic OpenAI JSON mode; no tool-calling/schema enforcement | Native tool use/structured outputs, multi-provider |
| Integration | CLI + Rust lib | **MCP server** is now the default way agents consume browsers; none here |
| Infra | local Chrome only | Managed/remote browsers (Browserbase, Steel), session persistence, proxies |
| Reliability | `consecutive_failures` counter only | Action caching / self-healing (Stagehand), deterministic replay |
| Observability | `tracing` logs, event bus | Session replay, traces, per-step screenshots |
| Safety | SSRF/DOM caps (Part 1: bypassable) | Network-layer allowlists, human-in-the-loop at login/payment, secret masking |
| Auth/forms | none seen (login, cookies, 2FA, uploads, iframes/tabs) | Standard |
| Benchmarks | self-reported (96.4% token cut, 2.33s/10 orgs), no public benchmark run | Published WebVoyager-style numbers (task sets differ, not directly comparable) |

Genuine differentiator: **Rust single binary, small footprint, registry-grounded Norwegian-company niche with Mod-11 + provenance.** That niche is defensible. "General autonomous AI browser engine" is not, against Browser Use / Stagehand / Playwright MCP.

## 5. Recommendations (in order)
1. **Be honest about scope**: describe System 2 as deterministic synthesis, or add a real LLM call over registry+recon facts with a strict schema (cite-by-source-id, reject uncited claims). That makes "zero fabrication" testable.
2. **Close the loop**: execute `suggested_followup_query` items within budget and re-synthesize.
3. **Ship an MCP server** (`navigate`, `snapshot`, `click`, `extract`, `research_company(org_nr)`), the biggest compatibility win.
4. Add AX-tree (and optional screenshots) or fix the README to say DOM + DOMSnapshot. Add a real Anthropic provider and an explicit `provider` config instead of key-prefix sniffing.
5. Network-layer security via CDP `Fetch`; drop or guard `Evaluate` (Part 1 #1, #2).
6. Reliability: lifecycle-event waits, browser pool, Brreg retry/429, batch checkpointing.
7. Evidence: CI, an agent-loop test against a local fixture site, a reproducible benchmark script; label README numbers unverified until then.

## Bottom line
A solid small Rust CDP agent skeleton plus a useful Norwegian registry pipeline. Not competitive as a general browser agent against 2026 tools, and its most marketed pieces ("dual-brain", "3-tree perception", "zero fabrication", "Anthropic-aligned") are rule-based templates or not implemented as described. Nothing dangerous beyond Part 1; the main risk is **overclaiming**.

---

# Part 3: Core agent workflow only (ignoring the registry/company APIs)

Scope: `agent/service.rs` loop, `agent/state.rs`, `actions/executor.rs`, `dom/serializer.rs`, `dom/service.rs`, `browser/session.rs`. Code-read only, nothing run.

Loop as built: `check_budget -> get_current_url -> get_dom_state (CDP DOM + DOMSnapshot) -> loop detector -> LLM decision (JSON) -> multi_act (stop on error/done/URL change) -> sleep 200ms`. Clean and Browser-Use-shaped. These are the real defects in it, worst first:

1. **The agent cannot read page content (HIGH).** In `dom/serializer.rs` the branch for non-interactive nodes is `else if !node.children.is_empty() { if ... && node.children.is_empty() {emit text} }`. The inner condition contradicts the outer one, so it can never fire. Paragraphs, headings, tables, and plain text in `div/p/span` are never sent to the model; only text inside buttons/links/inputs is. For a *research* agent this is the central failure: it can click around but can't see the article it landed on. Fix: emit visible text nodes (with a length cap), not only interactive elements.
2. **Action results never reach the model (HIGH).** `format_user_prompt` includes task, URL, step, `memory`, and DOM only. Errors like "Click failed: index 7 not found" or `Evaluate`/`Extract` output go to `consecutive_failures` and logs, not to the next prompt. The model repeats failing actions blind and `Extract` results are lost unless the model happens to copy them into `memory`. Fix: add an `<last_action_results>` block.
3. **Loop detector locks the agent into scrolling (HIGH).** `is_stuck_in_loop` triggers when the URL is unchanged for 4 steps. The recovery branch resets `consecutive_failures` but **not** `consecutive_same_url_count`, and `continue`s before the LLM is called; the next iteration increments the counter again. So on any single-URL task (SPA, form fill, multi-step on one page) after step 4 the agent only scrolls until the step budget ends. Also URL-unchanged is a poor stuck signal; hash the DOM state + last action instead.
4. **UTF-8 panics in the hot path (MEDIUM).** `&v[..60]` (attribute values) and `&text_trimmed[..80]` in the serializer slice by bytes; any non-ASCII page (Telugu, Norwegian, emoji) can panic mid-run. Same bug as `DomGuard::truncate_text`. Use `chars().take(n)`.
5. **Unbounded prompt growth (MEDIUM).** `memory` is a `Vec<String>` that only grows and is sent in full every step; full DOM text is sent with no token cap. `TokenOptimizer`/`TokenBudget` exist but are not used in this loop. Cap memory (summarize or keep last N) and truncate DOM.
6. **Fragile execution details (MEDIUM).**
   - Page readiness is `readyState` polling + fixed sleeps, no network-idle/lifecycle events; SPAs return before content renders.
   - Element indices are rebuilt every step and clicks resolve through a cloned `selector_map`; the multi-action batch (up to 3) acts on the DOM from step start, protected only by "did the URL change" (misses in-page DOM changes such as modals/dropdowns).
   - `attrs.contains_key("cursor")` in `is_inherently_interactive` is not a real signal; `div` with JS click handlers are missed unless CDP marks them clickable.
   - Single tab only; no new-tab/popup-window following, no iframe-aware click routing seen.
7. **LLM layer is minimal (MEDIUM).** One request per step, no retry on invalid JSON (the parse error aborts that decision path), no schema/tool-calling, no system/user separation for Gemini (concatenated), prompt-key-prefix provider sniffing. LLM failure silently switches to the regex heuristic mid-task.
8. **`Done` has no verification step.** `success` is trusted; nothing re-checks that the final answer is grounded in text the agent actually saw. Combined with #1, "zero fabrication" is not enforceable.

## Fix order for the core loop
1. Emit visible page text in the serializer (#1).
2. Feed last action results/errors to the prompt (#2).
3. Fix stuck detector: reset counter after recovery, base it on state hash (#3).
4. Char-safe truncation (#4); cap memory and DOM tokens (#5).
5. Lifecycle/network-idle waits; re-snapshot after each action instead of batching blind (#6).
6. Retry-with-error-feedback on bad JSON, real provider config (#7).
7. Add a fixture-site integration test that runs the loop end to end; items 1-3 would each fail it today.

---

# Part 4: The novelty / core solution, and what to actually build

Ignoring basics, the project's pitch rests on four ideas. I checked each against the code to see whether it is **implemented and wired into the real agent loop**.

| Claimed novelty | In the code | Verdict |
|---|---|---|
| **3-tree fusion + paint-order occlusion pruning (-96.4% tokens)** | `includePaintOrder: true` is requested in `DOMSnapshot` (`cdp/client.rs:201`) but never read; `accessibility_get_full_ax_tree` is defined but not called by `DomService`; no occlusion logic exists. Real pruning = skip script/style tags + viewport bounds filter. The 96.4% figure has no measurement code. | **Not implemented.** Only a basic viewport filter. |
| **MicroAction DSL (`CLICK 12`, `NAV "url"`) for low-token actions** | Parser + enum exist in `token_optimizer.rs`; the live agent (`service.rs`) uses the verbose JSON action schema. `MicroAction` is only used in the canned `dual_brain` demo path (`Click(1)` hard-coded). | **Exists but unused by the real loop.** |
| **System 1 / System 2 single-model switch** | No LLM involved (Part 2). Temperatures 0.0/0.3 in README are not applied anywhere. | **Not implemented as an LLM mechanism.** |
| **Zero-fabrication grounding + provenance** | `verify_grounding` (value must appear verbatim in page text) and `FactProvenance` (source URL, verbatim excerpt, timestamp, confidence) are well-designed, but `verify_grounding` has no caller, and the agent can't even see page text (Part 3 #1). `TokenBudgetTripwire` is also never called. | **Best idea in the repo, but not enforced.** |

So today the novel layer is mostly README + unwired helpers sitting beside an ordinary Browser-Use-style loop. The honest core is: **a Rust agent whose answers are supposed to be provable from page text.** That is the one idea worth finishing, because it is differentiated and testable.

## What to build (the actual solution)
Make **evidence-grounded browsing** the product. Concretely:

1. **Evidence-first perception.** Serialize visible text blocks with stable IDs (`[t14]`) next to interactive elements `[12]`. Every text block carries `source_url + node path + hash`. (Fixes Part 3 #1 and is the prerequisite for everything below.)
2. **Citation-enforced answers.** `Done`/`Extract` must return `{claim, evidence_ids[]}`. The runtime then checks, in code, that each claim's quoted span appears in the stored text of those IDs (`verify_grounding` upgraded to normalized fuzzy match). Uncited or unmatched claims are rejected back to the model as an error, or emitted as `unverified`. This is the real "zero fabrication", and it is cheap, deterministic, and benchmarkable.
3. **Differential perception (a genuinely token-saving idea).** Send the DOM once, then per step send only added/removed/changed nodes keyed by stable IDs (the code already has an `is_new` `*` marker to build on). Measure real tokens per task against full-snapshot baselines; this replaces the unproven 96.4%.
4. **Actually implement occlusion pruning**: use `paintOrder` + bounding boxes from the snapshot it already fetches to drop covered/offscreen/zero-size nodes. Small code, directly delivers the claimed saving, and is verifiable by an A/B token count.
5. **Real two-phase LLM split, if keeping the concept**: cheap fast pass (compact perception + MicroAction DSL, small model) for navigation; one slower pass for synthesis over *only* the evidence store. Make the DSL the live action format (it is the token win over verbose JSON schemas), with a fallback to JSON on parse failure.
6. **Evidence store + replay.** Persist every visited page's extracted text + screenshots hash per run; the final report links claims to evidence. Gives auditability that Browser Use / Stagehand don't ship by default.
7. **Prove it.** Build a small benchmark: ~50 fixture pages with known facts + injected prompt-injection and fake-fact traps. Report: answer accuracy, unsupported-claim rate, tokens/task, steps/task vs the same task on full-DOM prompting. Without numbers, delete the 96.4%/0.0% fabrication badges.

## Positioning
Do not pitch "autonomous AI browser engine" (crowded: Browser Use, Stagehand, Playwright MCP, Skyvern). Pitch: **"a token-frugal browser agent where every claim is cryptographically-traceable to page evidence"**, then implement items 1-4 and 7 first. The Norwegian registry pipeline becomes a demo vertical for that engine, not the engine itself.
