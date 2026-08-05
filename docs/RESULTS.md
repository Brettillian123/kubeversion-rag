# Results

Question set: **test** · 1811 answerable questions · 23018 chunks · generated 2026-08-05T16:09:56+00:00

## Ablation

| Configuration | recall@10 | mrr@10 | ndcg@10 | version_correct@1 |
|---|---:|---:|---:|---:|
| BM25 lexical baseline, no version awareness | 0.554 | 0.287 | 0.351 | 0.632 |
| Off-the-shelf bi-encoder, no version awareness | 0.574 | 0.282 | 0.352 | 0.523 |
| Off-the-shelf bi-encoder + metadata version filter | **0.703** | **0.456** | **0.516** | **1.000** |

Each row adds exactly one component to the row above it, so the delta is attributable.

### Reading `version_correct@1` correctly

It asks whether the top-ranked chunk actually applies to the Kubernetes version in the question — the failure this project exists to fix. But it means two different things depending on the row, and conflating them would overstate the result:

- **Rows without the version filter** — a genuine measurement. This is how often an unconstrained retriever surfaces the wrong release's snapshot first.
- **Rows with the version filter** — near-tautological. The filter only returns chunks whose range covers the target, so the metric is ~1.0 by construction and the only way to score below it is to return nothing at all.

So the filter's jump in this column is not a modelling win; it quantifies the size of the problem and confirms the filter is actually wired up. **Among filtered rows, `nDCG@10` is the discriminating metric** — it measures whether the right chunk is ranked first among the ones that *are* version-valid, which is what fine-tuning and reranking are for.

## By question source

### BM25 lexical baseline, no version awareness

| Source | recall@10 | ndcg@10 | version_correct@1 | n |
|---|---:|---:|---:|---:|
| changed-section | 0.547 | 0.346 | 0.639 | — |
| deprecation | 0.865 | 0.525 | 0.324 | — |
| deprecation-boundary | 1.000 | 0.783 | 0.500 | — |

### Off-the-shelf bi-encoder, no version awareness

| Source | recall@10 | ndcg@10 | version_correct@1 | n |
|---|---:|---:|---:|---:|
| changed-section | 0.571 | 0.352 | 0.527 | — |
| deprecation | 0.676 | 0.365 | 0.378 | — |
| deprecation-boundary | 0.750 | 0.533 | 0.250 | — |

### Off-the-shelf bi-encoder + metadata version filter

| Source | recall@10 | ndcg@10 | version_correct@1 | n |
|---|---:|---:|---:|---:|
| changed-section | 0.698 | 0.513 | 1.000 | — |
| deprecation | 0.919 | 0.624 | 1.000 | — |
| deprecation-boundary | 1.000 | 0.839 | 1.000 | — |

## Reading this table honestly

- The question set is split **by document family**, so no section in the test set was seen during training. Splitting by question would leak.
- Generated questions are templated. They isolate version-sensitivity precisely but are narrower than real user phrasing; the hand-written gold set is the check on that.
- Unanswerable questions are excluded from retrieval metrics (there is nothing to retrieve) and are measured as a refusal rate in the serving tests instead.
