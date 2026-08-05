# Results

Question set: **gold** · 11 answerable questions · 23018 chunks · generated 2026-08-05T22:39:45+00:00

## Ablation

| Configuration | recall@10 | mrr@10 | ndcg@10 | version_correct@1 |
|---|---:|---:|---:|---:|
| Off-the-shelf bi-encoder, no version awareness | 0.818 | 0.410 | 0.510 | 0.455 |
| Off-the-shelf bi-encoder + metadata version filter | 0.909 | 0.625 | 0.694 | **1.000** |
| Fine-tuned bi-encoder + version filter | 0.818 | 0.773 | 0.785 | **1.000** |
| Fine-tuned bi-encoder + version filter + fine-tuned cross-encoder | **1.000** | **0.803** | **0.854** | **1.000** |

Each row adds exactly one component to the row above it, so the delta is attributable.

### Reading `version_correct@1` correctly

It asks whether the top-ranked chunk actually applies to the Kubernetes version in the question — the failure this project exists to fix. But it means two different things depending on the row, and conflating them would overstate the result:

- **Rows without the version filter** — a genuine measurement. This is how often an unconstrained retriever surfaces the wrong release's snapshot first.
- **Rows with the version filter** — near-tautological. The filter only returns chunks whose range covers the target, so the metric is ~1.0 by construction and the only way to score below it is to return nothing at all.

So the filter's jump in this column is not a modelling win; it quantifies the size of the problem and confirms the filter is actually wired up. **Among filtered rows, `nDCG@10` is the discriminating metric** — it measures whether the right chunk is ranked first among the ones that *are* version-valid, which is what fine-tuning and reranking are for.

## By question source

### Off-the-shelf bi-encoder, no version awareness

| Source | recall@10 | ndcg@10 | version_correct@1 | n |
|---|---:|---:|---:|---:|
| gold | 0.818 | 0.510 | 0.455 | — |

### Off-the-shelf bi-encoder + metadata version filter

| Source | recall@10 | ndcg@10 | version_correct@1 | n |
|---|---:|---:|---:|---:|
| gold | 0.909 | 0.694 | 1.000 | — |

### Fine-tuned bi-encoder + version filter

| Source | recall@10 | ndcg@10 | version_correct@1 | n |
|---|---:|---:|---:|---:|
| gold | 0.818 | 0.785 | 1.000 | — |

### Fine-tuned bi-encoder + version filter + fine-tuned cross-encoder

| Source | recall@10 | ndcg@10 | version_correct@1 | n |
|---|---:|---:|---:|---:|
| gold | 1.000 | 0.854 | 1.000 | — |

## Reading this table honestly

- The question set is split **by document family**, so no section in the test set was seen during training. Splitting by question would leak.
- Generated questions are templated. They isolate version-sensitivity precisely but are narrower than real user phrasing; the hand-written gold set is the check on that.
- Unanswerable questions are excluded from retrieval metrics (there is nothing to retrieve) and are measured as a refusal rate in the serving tests instead.
