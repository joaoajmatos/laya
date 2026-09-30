# Realistic long-document decision datasets: survey

Status: draft survey, 2026-09-30. Nothing here has been measured on Laya. No dataset was downloaded; no dataset content is reproduced.

Purpose: fill the gap named in [`specs/002-decision-benchmark-baselines/spec.md`](../specs/002-decision-benchmark-baselines/spec.md) (Assumptions: upstream `typed-decisions` is synthetic and short; a realistic long-document set is out of scope for that phase) and in constitution principle VII (length studies should include realistic tasks next to controlled ones). Phase 2 in [`docs/research-plan.md`](research-plan.md) also needs annotated evidence spans, a short evidence-only counterpart per item, and pilot sizes of about 200 short (<512), 100 medium (1K-3K) and 50 long (4K-8K) inputs.

## How to read the confidence labels

- **V-primary**: read on the dataset's own page, paper (via an ar5iv HTML copy) or official repository during this survey.
- **V-mirror**: read only on a third-party mirror or a search-result summary. Treat as a lead.
- **Derived**: my arithmetic from verified numbers.
- **Unverified**: recalled or inferred; do not cite without checking.

Tool caveat: page contents were read through a summarising fetch tool, not raw HTML, and three PDFs (ContractNLI, CUAD, LexGLUE ACL paper) could not be parsed, so those were read through ar5iv. Every number below should be re-checked against the raw source before it goes into a report.

## What Laya is known to have seen (contamination context)

- Upstream's model card evaluates on MASSIVE, XNLI, AG News, DAIR Emotion, Banking77, SST-5 and its own `typed-decisions` set (V-primary, Hugging Face `convaiinnovations/laya`). It does not state the training-data composition or any contamination analysis (V-primary: absence of statement).
- `laya-typed-decisions` was fine-tuned on that benchmark's 1,200-case training split and tested on 400 cases (V-primary, model card). Domains: agent traces, customer service, invoices, security incidents.
- None of the candidates below appear in those lists. That is absence of evidence, not evidence of no overlap: the base encoders' pretraining corpora (ModernBERT, mmBERT) were not audited here, and SEC filings, arXiv papers, court text and Gutenberg fiction are all plausible web-crawl content. Zero-shot contamination risk therefore stays **unknown for all candidates**. For every candidate the same mitigation applies: use official held-out splits and report the short-counterpart result as the solvability check.
- Length note: upstream reports 1,024 tokens for `laya-typed-decisions` and up to 8,192 for the multilingual checkpoint with `max_len=8192` (V-primary, model cards). Inputs above the checkpoint's trained length exercise the windowed path, which is what Phase 2 studies.

## Candidates

Token counts in the sources use each paper's own tokenizer. Re-measure with the Laya tokenizer before binning items.

| # | Dataset | Licence (verified where noted) | Size | Length | Labels and mapping | Evidence span | Fit |
|---|---|---|---|---|---|---|---|
| 1 | [ContractNLI](https://stanfordnlp.github.io/contract-nli/) (Koreeda and Manning 2021) | CC BY 4.0 (V-primary, project page). A Hugging Face mirror (`kiddothe2b/contract-nli`) says CC-BY-NC-SA-4.0: conflict, use the original | 607 NDAs, 17 fixed hypotheses; 423/61/123 docs train/dev/test (V-primary, ar5iv) | mean 2,254 tokens, range 336 to 11,503; 86% exceed 512 (V-primary, ar5iv) | Entailment / Contradiction / NotMentioned per (doc, hypothesis). `choice` (3-way) or `noul` (entailed yes/no) | Yes, sentence or list-item spans, only for Entailment and Contradiction (V-primary) | High |
| 2 | [Qasper](https://huggingface.co/datasets/allenai/qasper) (Dasigi et al. 2021) | CC BY 4.0 (V-primary, HF card) | 1,585 NLP papers, 5,049 questions (V-primary) | Not verified here; full papers, likely several thousand tokens (Unverified) | yes/no 13.9%, unanswerable 10.2%, extractive 51.8%, abstractive 24.2% (V-primary, ar5iv). Filter to yes/no/unanswerable: `choice` (3-way) or `noul` (yes/no) | Yes, paragraph-level evidence, 55.5% of questions need multiple paragraphs (V-primary) | High |
| 3 | [QuALITY](https://github.com/nyu-mll/quality) (Pang et al. 2022) | Sources CC-BY or more permissive per paper (V-primary, ar5iv); per-article licence field in data | 6,737 questions (V-primary) | mean 5,159 tokens, articles 2,000 to 8,000 (V-primary) | 4-option multiple choice: `choice` directly | No span annotation found; has validator answers and "context required" ratings (V-primary) | High for length, low for evidence |
| 4 | [CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa) (Hendrycks et al. 2021) | CC BY 4.0 (V-primary, HF card) | 510 contracts, 41 clause categories, >13,000 annotations (V-primary) | "few pages to over 100 pages" (V-primary); numeric distribution not verified, likely far above 8K tokens (Unverified) | Per (contract, category): clause present or absent, derived from whether answer spans exist (Derived, needs a data check). `noul` | Yes, character offsets (V-primary) | Medium: needs windowing |
| 5 | [LexGLUE ECtHR A/B](https://github.com/coastalcph/lex-glue) | CC-BY-4.0 on the HF card (V-primary); underlying court text terms not checked | ~11k cases, 9k/1k/1k split, 2001-2019 (V-primary, ar5iv) | Cases average 25.2 fact paragraphs, range 5 to 259 (V-primary, rationale paper via ar5iv); token stats only in a figure, not read | Multi-label alleged-article violations. Per article: `noul`; or "which article" as `choice` | Silver paragraph rationales for all cases, gold rationales for 50 test cases by one expert (V-primary) | Medium |
| 6 | LexGLUE SCOTUS | as above | 7.8k opinions; 5,000/1,400/1,400 (V-primary) | "much longer than 512" per figure caption; numbers not read | 14-way issue area: `choice` | None | Low: topic label, no located evidence |
| 7 | [arXiv classification](https://huggingface.co/datasets/ccdv/arxiv-classification) (He et al. 2019, HF packaging) | Not stated on HF card (V-primary: absence). Underlying arXiv terms not checked | ~33k papers, 11 classes; 28.4k/2.5k/2.5k (V-primary) | All >4k tokens; class means 5,630 to 7,439 words (V-mirror) | 11-way subject: `choice` | None; topic is spread across the paper | Low: no "necessary evidence"; length ladder only |
| 8 | OPP-115 / PolicyQA / [PrivacyQA](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP) | OPP-115: research, teaching and scholarship only, in the spirit of CC BY-NC (V-mirror, search summary). PrivacyQA: MIT (V-primary) | OPP-115: 115 privacy policies, 3,792 segments, ~103k annotated spans (V-mirror). PrivacyQA: 1,750 questions (V-primary) | Policy lengths not verified | OPP-115 practice categories per policy: derived `noul` ("does the policy say X"). PrivacyQA: relevant/irrelevant per sentence, sentence-level not document-level | Yes (OPP-115 character offsets, V-mirror; PrivacyQA sentences) | Medium-low: small, NC licence, lengths unknown |
| 9 | [QMSum](https://github.com/Yale-LILY/QMSum) | MIT (V-primary) | 232 meetings, 1,808 query-summary pairs (V-primary) | Not verified | Free-text summaries, no typed label | Yes, relevant spans (V-primary) | Low: would need new labelling |
| 10 | [LongBench v2](https://huggingface.co/datasets/THUDM/LongBench-v2) | Apache 2.0 (V-primary) | 503 four-way MCQ (V-primary) | 8k to 2M words (V-primary) | 4-way `choice` | None | Poor: shortest items already exceed 8,192 tokens |
| 11 | Clinical: EHRNoteQA, LCD, LongHealth (MIMIC-based) | PhysioNet credentialed data use agreement (V-mirror, search results) | EHRNoteQA 962 QA; Level 2 is 3,000 to 7,000 tokens. LCD notes median ~1,687 words (V-mirror) | see left | Multiple choice or mortality label | Not verified | Restricted: cannot be redistributed or run on shared infrastructure without credentials; exclude from the open pipeline |
| 12 | Support tickets: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) and Kaggle ticket sets | CC-BY-NC-4.0 (V-primary, HF card); Kaggle sets CC BY 4.0 / CC0 (V-mirror) | 61.8k rows; 52 queues, 5 priorities, EN and DE (V-primary) | Body at most 2,260 characters (V-primary): short, well under 1K tokens | queue routing `choice`, priority `score` | None | Wrong length; real-vs-synthetic origin unverified |
| 13 | Enron email annotations (Berkeley BEEAP ~1,900 emails; LDC topic set 4,936 emails, 32 topics) | LDC set needs LDC access (V-mirror, catalog snippet); BEEAP terms not checked | see left | Most single emails are short (Unverified); thread-level labelling would be new work | Genre / topic | None | Low |

## What is missing

No public dataset found here covers the deployment classes the research plan names for the long bin: support-ticket or email-thread routing with 1K to 8K tokens, incident reports or transcripts with triage labels. The ticket sets are short and of uncertain provenance; the email corpora are either restricted or lack labels; the clinical sets are credentialed. The candidates below are therefore legal and scientific documents that share the shape (a long real document, a fixed typed question, a located necessary passage) but not the deployment domain. Any report must say so. A private or partner-supplied set (real anonymised tickets, with the owner's licence) remains the only route to the actual routing use case and should be listed as a gap, not silently replaced.

## Ranked shortlist

1. **ContractNLI**: best fit. Verified CC BY 4.0, verified document-level splits, verified sentence-level evidence, mean length 2,254 tokens (fills the 1K-3K bin, tail reaches ~11.5K), fixed typed question with three classes that map cleanly to `choice`. Weaknesses: only 17 hypotheses (memorisation risk for any trained variant); NotMentioned has no evidence span (fits the "missing evidence" family but the span field is empty); legal NDAs, not routing.
2. **Qasper (yes/no and unanswerable subset)**: verified CC BY 4.0, verified paragraph evidence, multi-paragraph questions (55.5%) support the distributed-reasoning family. The usable subset is small: about 700 yes/no and 500 unanswerable questions across all splits (Derived: 13.9% and 10.2% of 5,049), and questions with annotator disagreement need a rule. Paper token lengths need measuring.
3. **QuALITY**: fills the 4K-8K bin with real prose (verified mean 5,159 tokens, range 2K to 8K) and natural 4-way `choice`. Missing: annotated evidence spans. Section "Evidence" below gives a plan; until it is executed, QuALITY items cannot serve the "necessary evidence located" requirement and should be reported as unlocated-evidence items.
4. **CUAD**: verified spans and licence, many contracts, but lengths mostly exceed 8K tokens (Unverified), so each item needs the clause span plus a bounded window, which turns a natural document into a constructed one. Use only if ContractNLI's long tail proves too thin.

Honourable mention: ECtHR (silver rationales for all cases, gold for 50). Rationales are automatically extracted from the judgment text, so they are not verified "necessary" evidence, and the multi-label task needs care to become a single typed question.

Not recommended: arXiv classification and SCOTUS (no evidence), LongBench v2 (too long), MIMIC-derived sets (access restricted), ticket sets (too short).

## Recommendation

Adopt ContractNLI as the primary realistic set, Qasper (yes/no/unanswerable) as a second domain, and QuALITY as a long-bin stress set with evidence to be added by audit. Keep CUAD in reserve. Label the results "realistic legal and scientific documents, not routing" everywhere, and keep them in a separate table from the controlled families.

## Proposed minimal protocol

This extends the Phase 2 plan; it does not replace the controlled families. All numbers are screening sizes, not a power guarantee, matching the plan's own wording.

**Item definition.** One item = (document, typed question, gold label, evidence span or none, source group = document id). ContractNLI item = document plus one of the 17 hypotheses rendered as the question, options Entailment/Contradiction/NotMentioned (`choice`). Qasper item = paper plus question, options yes/no/unanswerable. QuALITY item = article plus question and its four options.

**Splits.** Split by document, never by question, so one document appears in exactly one split (principle VII). Use official splits: ContractNLI 423/61/123 documents; Qasper 1,585 papers across its official train/validation/test; QuALITY official train/dev. Thresholds, temperature and any bin definitions use dev only; the official test documents stay untouched until metrics, margins and comparison procedures are fixed. Zero-shot Laya needs no training split; an adaptation experiment would train on official train only and additionally hold out some hypotheses (ContractNLI) to test more than the 17 fixed ones.

**Sizes (pilot).**
- Short (<512 tokens): evidence-only counterparts of the medium and long items (see below), 200 items, drawn from dev and test in proportion.
- Medium (1K-3K): about 100 ContractNLI items from at least 40 distinct documents, stratified so Entailment, Contradiction and NotMentioned are each at least 25.
- Long (4K-8K): 50 items, mixed from ContractNLI documents above 4K tokens, Qasper long papers and QuALITY articles, at least 15 items per source and no more than 3 items per document.
- Balanced diagnostic sample separate from a sample reflecting natural label frequencies, as the plan requires. Counts by label and by source are reported.
- Documents above 8,192 tokens are excluded or reported as a separate overflow bin, not silently truncated.

**Length binning.** Measure with the Laya tokenizer on the final rendered input (document plus question plus options). Do not reuse the papers' token counts.

**Evidence spans.**
- ContractNLI: use the annotated sentence and list-item spans, mapped to character offsets in the rendered input. NotMentioned items carry an explicit "no evidence" flag.
- Qasper: use the annotated evidence paragraphs. Where annotators disagree on answer or evidence, keep only items with agreeing answers and record the union of agreeing annotators' evidence.
- QuALITY: no native spans. Obtain them in two steps. (a) Candidate passages by lexical or embedding retrieval (not by Laya, to avoid circularity). (b) The researcher audits a sampled subset (target the 50 long items) and marks the minimal passages needed to answer; a necessity check confirms that removing them makes the question unanswerable to a human, and that the short counterpart built from them alone is answerable. Verdicts are the researcher's, consistent with the existing `phase2-audit` workflow. Items without an audited span are reported separately.
- Short evidence-only counterpart: the evidence span or spans plus the question, padded to nothing else, so the native model can be checked on solvability first. Baseline-unsolved items are reported separately as the plan requires.

**Reporting.** Accuracy and macro-F1 per label, per length bin and per source, native versus windowed, with the same predeclared margins as the controlled families. Report contamination status as "unknown", state that no deployment-domain (routing, triage) realistic set was available, and state that legal and scientific text may overlap with pretraining data of the base encoder.

## Effort to convert

| Dataset | Work | Rough effort (Unverified estimate) |
|---|---|---|
| ContractNLI | Load JSON, render 17 hypotheses as questions, map span ids to offsets, build short counterparts | 1 to 2 days |
| Qasper | Filter to yes/no/unanswerable, resolve annotator disagreement, flatten sections, map evidence to offsets | 2 to 3 days |
| QuALITY | Load, render options, retrieval plus researcher audit for spans | 1 day code plus audit time |
| CUAD | Derive presence label, windowing rule, offset mapping | 3 to 4 days |

## Verification log

Verified on a primary or near-primary source (page or paper via ar5iv): ContractNLI licence, size, splits, token stats and span definition; Qasper licence, size, answer-type shares, evidence types; QuALITY size, mean length, licences, format; CUAD licence, contract count and annotation count; ECtHR case counts and rationale annotation; PrivacyQA MIT licence; Laya checkpoint context lengths and evaluation datasets; Tobi-Bueck licence and body length limit; LongBench v2 licence and length range.

Not verified: Laya's training-data composition and any contamination analysis (not published on the pages read); Qasper and CUAD length distributions; LexGLUE and SCOTUS token statistics; OPP-115 licence text from the source itself; per-source licences for QuALITY articles beyond the paper's statement; MIMIC-derived dataset details beyond search snippets; whether the Tobi-Bueck tickets are real or synthetic; whether any candidate is in the base encoders' pretraining corpora. A Hugging Face mirror of ContractNLI lists a different licence from the project page; check the licence at download time.
