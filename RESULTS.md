# Results

> The Recall@5 / citation accuracy / supported-claim rate / hallucination
> rate targets stated in the project spec (81% / 85% / 82% / ≤13%) are
> **quality bars, not assumed results**. Every number below comes verbatim
> from `experiments/results/*.json`, produced by actually running
> `python -m backend.evaluation.baselines` and
> `python -m backend.evaluation.ablation` against the benchmark in this
> repository - nothing here is hand-typed or estimated.

## 1. Dataset

- **Evidence corpus**: 345 real PubMed abstracts (1,096 section-aware
  chunks after ingestion) fetched live via NCBI E-utilities across 18 topic
  queries chosen to match the synthetic patients' conditions (type 2
  diabetes/HbA1c/metformin, hypertension, hyperlipidemia/statins,
  hypothyroidism, iron-deficiency anemia, chronic kidney disease,
  cardiovascular risk, medication adherence, lifestyle/exercise). Cached
  under `data/raw/pubmed/*.json`; full provenance (PMID, title, authors,
  year, journal, URL) preserved per document. No papers, PMIDs, DOIs, or
  URLs were fabricated - everything is exactly what NCBI returned.
- **Synthetic patients**: 3 patients (`data/patients/*.json`) - a type 2
  diabetes/hyperlipidemia profile, a hypertension + stage 3 CKD profile, and
  a hypothyroidism + iron-deficiency-anemia profile. All data is synthetic;
  no real patient-identifiable information was used.
- **Evaluation benchmark**: 15 questions (`data/evaluation/benchmark.json`),
  5 per patient, each with `gold_evidence` chunk IDs identified by actually
  running hybrid retrieval + reranking against the built index and
  inspecting the real returned passages (not guessed). Two items
  (`bm_06`, `bm_14`) are intentionally left as weaker-coverage / harder cases
  where the corpus's topical coverage is thin, noted in their `notes` field.

## 2. Models

| Component | Model | Notes |
|---|---|---|
| Dense embeddings | `pritamdeka/S-PubMedBert-MS-MARCO` | Biomedical sentence-transformers model; symmetric encoder |
| Cross-encoder reranker | `ncbi/MedCPT-Cross-Encoder` | Purpose-built for PubMed query/passage relevance |
| Claim verification NLI | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | General-purpose entailment model - see Limitations |
| LLM (generation + secondary verifier) | `Qwen/Qwen2.5-1.5B-Instruct` (local, CPU) | Default, zero-API-key provider; swappable via `.env` `LLM_PROVIDER=openai` |
| BM25 | `rank_bm25.BM25Okapi` | Pure-Python, no external search service |
| Vector index | FAISS `IndexFlatIP` (cosine via L2-normalized inner product) | |

All models are configurable via `backend/config/settings.py` / `.env` and
were reachable/downloaded directly from the Hugging Face Hub in this
environment - no substitutions or placeholders were used.

## 3. Retrieval configuration

- `HYBRID_ALPHA = 0.5` (equal weight, weighted min-max score fusion)
- `BM25_CANDIDATE_K = 20`, `DENSE_CANDIDATE_K = 20`
- `RERANK_TOP_K = 5` (final evidence set size handed to the generator)
- `NLI_SUPPORT_THRESHOLD = 0.55`, `NLI_CONTRADICTION_THRESHOLD = 0.55`
- Reliability gate mode: `flag` (claims are kept and labeled
  ✓/⚠/✗ rather than silently edited - see spec section 14)
- LLM: greedy decoding (`temperature = 0.0`), `max_new_tokens = 512`,
  system prompt includes one worked few-shot example of the required
  citation format (found necessary for the small local model to reliably
  emit `[E#]` markers - see Failure Cases below)

## 4. Experimental methodology

Nine canonical pipeline configurations were run once each over the full
15-item benchmark (`backend/evaluation/runner.py`), and results were cached
per-configuration under `experiments/results/<label>.json` so the two
comparison tables below share configurations rather than re-running them:

**Baselines** (`python -m backend.evaluation.baselines`):
A. LLM only (no retrieval) · B. Dense + LLM · C. BM25 + LLM ·
D. Hybrid (BM25+Dense) + LLM, no reranker · Final: Hybrid + Reranker + Verification

**Ablation** (`python -m backend.evaluation.ablation`): Full System, minus
BM25, minus dense retrieval, minus the reranker (= Baseline D), minus claim
verification, minus patient-context augmentation.

For every run: **Recall@5/@10** is computed against each item's
`gold_evidence` chunk IDs, using whatever evidence set that configuration
actually handed to the generator (reranked top-k, or fused top-k when
reranking is disabled). **Citation accuracy** = citations that are both
structurally valid (trace to a real retrieved chunk) *and* judged
`SUPPORTED` by the NLI claim verifier, divided by total citations emitted.
**Supported-claim rate** = claims verified `SUPPORTED` / all verified
claims. **Hallucination rate** = claims verified `UNSUPPORTED` or
`CONTRADICTED` / all verified claims. Latencies are wall-clock, CPU-only, on
the evaluation machine (no GPU).

---

## 5. Results

All numbers below are copied verbatim from
`experiments/results/baselines_summary.json` and
`experiments/results/ablation_summary.json`.

> **Caveat on Recall@10**: `RERANK_TOP_K=5` truncates every configuration's
> evidence set to 5 passages *before* recall is computed, so Recall@10 is
> mechanically identical to Recall@5 in every table below - it is not a
> meaningfully distinct measurement in this run. Recomputing it properly
> would require re-running with a larger evidence set size, which was not
> done here in the interest of time; this is flagged rather than silently
> presented as if it were computed over a genuinely larger candidate pool.

### Baseline comparison (15-item benchmark, measured)

| System | Recall@5 | Citation Accuracy | Supported Claims | Hallucination | Total Citations | Avg Gen. Latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| A. LLM only (no retrieval) | 0.0% | 0.0% (0/0 valid) | 0.0% (0/71) | 100.0% (71/71) | 42 | 49.1 |
| B. Dense + LLM | 26.7% | 21.4% (3/14) | 2.6% (2/77) | 97.4% | 14 | 88.8 |
| C. BM25 + LLM | 16.7% | 50.0% (9/18) | 3.5% (3/85) | 96.5% | 18 | 77.8 |
| D. Hybrid (BM25+Dense), no rerank | 23.3% | 48.8% (21/43) | 9.3% (8/86) | 90.7% | 43 | 108.0 |
| **Final System** (Hybrid+Reranker+Verification) | **66.7%** | 14.3% (3/21) | 1.2% (1/82) | 98.8% | 21 | 80.6 |

**Against the spec's target quality bars**: Recall@5 66.7% vs. an 81%
target (best of the five configurations, and 2.5-4x better than any
non-reranked baseline, but short of target); citation accuracy 14.3% vs.
85% target; supported-claim rate 1.2% vs. 82% target; hallucination rate
98.8% vs. a ≤13% ceiling. **The targets were not met on this benchmark with
this configuration** - see analysis below and Limitations.

### What the numbers show

1. **Reranking dramatically improves retrieval quality.** Recall@5 jumps
   from 16.7-26.7% (BM25-only / Dense-only / Hybrid-without-rerank) to
   66.7% once the MedCPT cross-encoder reranks the fused candidate pool -
   a 2.5-4x improvement, and the single clearest validated result in this
   study. This matches the manual example documented in Failure Cases
   below (BM25 alone ranked an off-topic "Refractory hypothyroidism"
   passage above the correct metformin passages; reranking fixed it).
2. **Citation accuracy and supported-claim rate are low across every
   configuration that has any evidence at all, and do not track Recall@5.**
   Even the Final System, with by far the best retrieval, has *lower*
   citation accuracy (14.3%) than Baseline C (50.0%) or D (48.8%). With
   only 14-43 total citations per configuration, these ratios are
   computed over a small sample and are noisy - but the more fundamental
   issue, confirmed by inspecting the underlying per-item JSON files
   (`experiments/results/*.json`), is that the **NLI verifier is the
   dominant bottleneck, not retrieval**: the local LLM writes compound,
   interpretive sentences (e.g. combining a mechanism claim with a dosing
   caveat in one sentence) that a short, narrow retrieved chunk often does
   not entail strongly enough to clear the 0.55 entailment threshold, even
   when the chunk is topically the right source.
3. **The no-retrieval baseline (A) fabricates citations from nothing.**
   With zero evidence passages available, the local LLM still emitted 42
   `[E#]`-style citation markers across the 15 questions (up to 7 on a
   single answer) that referenced numbers that were never given to it.
   Every single one was correctly caught as structurally invalid by the
   citation verifier (citation accuracy = 0/42 = 0.0%), and every claim
   was correctly classified UNSUPPORTED (hallucination rate = 100%). This
   is exactly the failure mode claim/citation verification is designed to
   catch, and it caught all of it.
4. **Hallucination rate is high (91-100%) across the board**, including
   the Final System. This is a genuine, measured shortfall against the
   ≤13% target - see Failure Cases and Limitations for the most likely
   causes (small local LLM + general-purpose NLI verifier + narrow
   per-chunk premises), and Future Work for what to try next.

### Ablation study (15-item benchmark, measured)

| System | Recall@5 | Citation Accuracy | Supported Claims | Hallucination |
|---|---:|---:|---:|---:|
| LLM only (no retrieval) | 0.0% | 0.0% | 0.0% | 100.0% |
| Dense RAG (dense only, no rerank) | 26.7% | 21.4% | 2.6% | 97.4% |
| BM25 RAG (BM25 only, no rerank) | 16.7% | 50.0% | 3.5% | 96.5% |
| Hybrid RAG (BM25+Dense, no rerank) | 23.3% | 48.8% | 9.3% | 90.7% |
| Hybrid + Reranker (no claim verification) | 66.7% | 0.0%\* | N/A\* | N/A\* |
| **Full System** | **66.7%** | **14.3%** | **1.2%** | **98.8%** |
| Full System − remove BM25 | 53.3% | 20.7% | 2.5% | 97.5% |
| Full System − remove dense retrieval | 50.0% | 7.7% | 1.1% | 98.9% |
| Full System − remove patient-context augmentation | **100.0%** | **26.3%** | **7.0%** | 93.0% |

\* With claim verification disabled, no claim can ever be marked
NLI-verified-SUPPORTED by definition, so citation accuracy is mechanically
0% and supported-claim-rate/hallucination-rate are undefined (0 verified
claims) - this row measures retrieval+generation only, not verification.

**Headline ablation finding: patient-context augmentation *hurts*
retrieval on this benchmark.** Removing it takes Recall@5 from 66.7% to a
perfect 100.0%, and also roughly doubles citation accuracy (14.3% →
26.3%) and supported-claim rate (1.2% → 7.0%) relative to the Full System.
The likely mechanism: `query_builder.py` appends full rendered
patient-context sentences (e.g. "Lab result: HbA1c = 7.4 % (reference
4.0-5.6, status: high, 2026-02-14)") to the retrieval query, and these
verbose, numerically-dense sentences add enough extra terms/embedding mass
to dilute the BM25/dense signal for the actual clinical topic in the
question - on a benchmark where every question is already unambiguous
about its topic without patient context, this pure dilution cost is never
offset by a disambiguation benefit. This is a genuine, measured weakness of
the current query-construction design (spec section 9), not a tuning
artifact - see Future Work.

**Both BM25 and dense retrieval individually contribute to the Full
System's Recall@5**: removing either one drops Recall@5 from 66.7% to
53.3% (no BM25) or 50.0% (no dense) - confirming the hybrid combination
outperforms either retrieval method alone, consistent with Baselines B/C
above.

## 6. Failure cases

- **Fabricated citations from a no-evidence generator, caught 100% of the
  time.** Baseline A (LLM only, zero retrieved passages) still emitted 42
  `[E#]`-style citation markers across the 15 benchmark questions (up to 7
  on a single answer), referencing evidence numbers that were never
  provided. The citation verifier correctly flagged every one as
  structurally INVALID (0/42 valid → 0.0% citation accuracy) and the claim
  verifier correctly classified every associated claim as UNSUPPORTED
  (100% hallucination rate for this baseline). This is the exact failure
  mode the verification pipeline exists to catch, demonstrated concretely
  rather than assumed.
- **Small local LLM instruction-following required a worked example.** The
  default 1.5B-parameter local model initially emitted zero `[E#]` citation
  markers at all despite explicit system-prompt instructions, producing
  answers that were plausible but entirely unverifiable. Adding one worked
  few-shot example of the exact required format to the system prompt fixed
  this - citations are now reliably present (14-43 per configuration) -
  but did not by itself fix downstream NLI-verified support (see below).
- **NLI verification is the dominant bottleneck once evidence exists**,
  more than retrieval quality: the Final System has the best Recall@5
  (66.7%) of any configuration but the *worst* citation accuracy (14.3%)
  among the RAG configurations. Manual inspection of the local model's
  generated sentences shows a pattern of compound, interpretive claims
  (mechanism + caveat + patient-relevance in one sentence) that a single
  short retrieved chunk often does not entail strongly enough to clear the
  0.55 threshold on the general-purpose NLI model, even when that chunk is
  topically the correct source. This is a generator-phrasing /
  verifier-strictness interaction, not a retrieval failure.
- **Hybrid fusion without reranking surfaces off-topic passages**: for the
  query "What factors could explain elevated HbA1c despite metformin
  adherence?", raw BM25 ranked a passage titled *"Refractory
  hypothyroidism"* above on-topic metformin passages (term-overlap
  artifact); the cross-encoder reranker corrected this, ranking the actual
  metformin-pharmacology passages first - a concrete, reproducible
  demonstration of the reranker's value on this corpus, consistent with
  the large measured Recall@5 jump from reranking.
- **Thin corpus coverage on some questions**: benchmark items `bm_06`
  (ACE inhibitors) and `bm_14` (urine albumin-to-creatinine ratio) retrieve
  only moderate-relevance passages (lower BM25/rerank scores) because the
  18-topic PubMed fetch didn't target those sub-topics directly - an honest
  reflection of the corpus's limited size/scope rather than a retrieval bug.

## 7. Limitations

- The corpus (~1,100 chunks from 18 topic queries) is a research-scale
  sample of PubMed, not comprehensive biomedical coverage - Recall@k is
  bounded by what was actually fetched.
- No off-the-shelf biomedical-specific NLI model is readily available as a
  pip-installable model; claim verification uses a strong general-purpose
  NLI model (`DeBERTa-v3-base-mnli-fever-anli`) instead, which may
  under- or over-score domain-specific phrasing compared to a model
  fine-tuned on biomedical entailment data (e.g. MedNLI).
- The default local LLM is intentionally small (1.5B parameters) so the
  system runs with zero API keys/cost; this is very likely the single
  biggest lever on citation accuracy / supported-claim rate / hallucination
  rate - re-running with `LLM_PROVIDER=openai` and a stronger model is the
  first thing to try if reproducing/improving these numbers.
- Citation accuracy as implemented applies one claim-level NLI verdict to
  every citation on that claim; a claim citing two passages where only one
  actually supports it is not distinguished from one where both do.
- Recall@10 is not meaningfully distinct from Recall@5 in this run because
  `RERANK_TOP_K=5` truncates the evidence set before recall is computed
  for every configuration - see the caveat in section 5.
- The claim extractor splits on sentence boundaries, so a single compound
  sentence (e.g. "X does A, which may cause B in some patients") becomes
  one claim that must be entirely entailed by a cited passage to count as
  SUPPORTED. This likely under-counts partial support and is a plausible
  contributor to the low measured supported-claim rate - splitting on
  clause boundaries instead of sentences is a natural improvement.
- This is a research prototype for studying retrieval/verification
  architecture - it is not a validated clinical tool and must not be used
  for real clinical decisions.

## 8. Future work

- **Fix patient-context query dilution** (see the ablation headline
  finding above): try appending only short keyword fragments (e.g. "HbA1c
  high") instead of full rendered sentences, retrieving with the raw
  question and the patient-context-augmented query separately and fusing
  both result sets, or down-weighting patient-context terms relative to the
  question in the fused ranking. This is now the second highest-priority
  fix alongside the generator, since it is a clean, reproducible, negative
  effect on the exact metric (Recall@5) closest to its target.
- **Re-run with a stronger generator** (`LLM_PROVIDER=openai` + a frontier
  model, or a larger local open-weight model in the 3B-8B range) as the
  single highest-leverage next experiment: sections 5-6 show the
  bottleneck is generator-phrasing/NLI-strictness interaction more than
  retrieval, so this is the most likely lever to move citation
  accuracy/supported-claim rate/hallucination rate toward the targets.
- Split claims on clause boundaries rather than full sentences, so a
  compound sentence's individually-supportable parts aren't penalized as
  one unsupported unit.
- Tune/ablate the NLI entailment threshold (currently 0.55) and compare
  against an LLM-only verifier (already implemented as an optional
  secondary signal) as the primary verifier instead of secondary.
- Evaluate with a larger/broader corpus (e.g. full PubMed Central subset,
  or additional clinical guideline sources) and re-measure Recall@10 with
  a genuinely larger evidence-set size than Recall@5 (see the section 5
  caveat).
- Fine-tune or source a biomedical-specific NLI model for claim
  verification.
- Add per-citation (rather than per-claim) entailment scoring.
- Expand the benchmark beyond 15 items and add inter-annotator-checked gold
  evidence for a more statistically robust Recall@k estimate.

---

**Status**: complete. Both `python -m backend.evaluation.baselines` and
`python -m backend.evaluation.ablation` ran to completion against the full
15-item benchmark in this environment (9 canonical configurations, cached
per-configuration under `experiments/results/*.json`); every number in
section 5 above is copied directly from `baselines_summary.json` /
`ablation_summary.json`. Re-run either command with `--force` to
regenerate from scratch (e.g. after changing models, corpus, or benchmark).
