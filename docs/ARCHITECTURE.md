# Architecture — kubeversion-rag

## Goal

Answer Kubernetes questions correctly **for the user's specific cluster version**, with
citations, and prove it beats naive RAG on a held-out benchmark.

The measurable claim: `nDCG@10` and `version-correct@1` on a held-out question set,
reported as an ablation table where every row is one configuration change.

## Why this problem needs modeling, not prompting

`kubernetes/website` keeps the *same document path* on every release branch. Two
snapshots of `content/en/docs/reference/using-api/deprecation-guide.md` from
`release-1.26` and `release-1.34` are ~90% lexically identical but give contradictory
answers about whether `flowcontrol.apiserver.k8s.io/v1beta3` is served.

Off-the-shelf sentence embeddings score both near-identically against a query, so a
naive vector search returns the wrong one roughly half the time. That failure is:

- **real** — serving 1.26 guidance to a 1.34 cluster is actively wrong,
- **measurable** — we know the correct version for every generated question,
- **fixable by modeling** — metadata filtering, a fine-tuned bi-encoder, and a
  cross-encoder reranker each move the number, and we can attribute how much.

The wrong-version snapshot of the *same document* is also a near-perfect hard negative:
maximally similar in embedding space, definitively wrong. Hard negatives are usually the
expensive part of building a retrieval training set; here the corpus generates them for free.

## Data flow

```
kubernetes/website (git, N release branches, sparse: content/en/docs)
        │
        ├─► ingest.fetch      → per-version markdown trees
        ├─► ingest.chunk      → heading-aware chunks, tagged with source version
        │        │
        │        └─► coalesce identical text across adjacent versions
        │                     → Chunk{doc_path, heading_path, text, min_version, max_version}
        │
        └─► ingest.deprecation → structured facts from deprecation-guide.md
                 │                 (api_group, resource, removed_in, replacement, since)
                 ▼
           dataset.build       → (question, target_version, positive_chunk, hard_negatives[])
                 │
        ┌────────┴────────┐
        ▼                 ▼
   train/               eval/
   ├ bi-encoder (MNRL)   ├ recall@k, MRR@10, nDCG@10
   └ cross-encoder (CE)  └ version-correct@1
                          → ablation table (docs/RESULTS.md)
```

At serve time:

```
query ──► retrieval.query.extract_version  (explicit "on 1.28" | implicit | default)
      ──► retrieval.pipeline
            ├ dense recall (bi-encoder, top-50)   [+ optional BM25 union]
            ├ version filter (min_version ≤ v ≤ max_version)
            ├ cross-encoder rerank → top-5
            └ generation (Claude, cite-or-refuse)
```

## Chunk identity and version coalescing

Storing one chunk per (path, heading, release-branch) across 12 branches would multiply
the index ~12× with near-duplicate vectors — which both wastes storage and *degrades*
retrieval, because near-duplicates crowd out diverse results in the top-k.

Instead a chunk is keyed by `(doc_path, heading_path, sha256(text))` and carries the
**contiguous version range** over which that exact text was present. A doc that never
changed between 1.24 and 1.35 becomes a single chunk with `min_version=1.24,
max_version=1.35`. A doc that changed at 1.29 becomes two chunks with adjacent,
non-overlapping ranges.

This makes the version filter a cheap range predicate, and makes "same document, wrong
version" an explicit, enumerable relationship: two chunks sharing a `family_id`
(`doc_path` + `heading_path`) with disjoint version ranges.

## Service topology on Kubernetes

| Component | Workload | Why that primitive |
|---|---|---|
| Qdrant | StatefulSet + PVC | Stateful, needs stable identity and durable storage |
| Embedding service | Deployment + HPA | Stateless, CPU-bound, scales with request rate |
| RAG API | Deployment + HPA | Stateless, latency-sensitive front door |
| Docs ingestion | CronJob | Runs on the K8s release cadence, not per-request |
| Index backfill | Job | One-shot, must run to completion, restartable |
| Eval gate | Job (in CI) | Must pass before a deploy is promoted |

Readiness probes on the RAG API check the Qdrant connection and that the active
collection exists — not merely that the process is alive. A pod that cannot reach its
index should not receive traffic.

## Zero-downtime embedding-model migration

Changing the embedding model invalidates every stored vector. The migration path:

1. `scripts/migrate_embedding_model.py plan` — resolves a new collection name
   (`chunks__<model-slug>__v<n>`).
2. `... backfill` — a Job embeds the corpus into the new collection while the old one
   continues serving.
3. `... evaluate` — runs the eval harness against both collections.
4. `... promote` — flips `ACTIVE_COLLECTION` in a ConfigMap and triggers a rollout,
   only if the new collection wins on the gate metric.
5. `... rollback` — flips back; the old collection was never deleted.

The API reads the collection name from the environment at request time (not import
time) so a ConfigMap change plus a rollout is sufficient — there is no code deploy in
the promote path.

## Evaluation

Two question sources, kept strictly separate:

- **Generated** (`dataset.build`) — templated from parsed deprecation facts. Large
  (thousands), used for training. Split by *document family* so no family appears in
  both train and test; splitting by question would leak.
- **Curated** (`data/gold/*.jsonl`) — hand-written and hand-verified. Small, never
  trained on, and the only set quoted in the headline results.

Metrics, all computed in `eval/metrics.py` with no dependency on the retrieval code:

- `recall@k` — is the correct chunk in the top k
- `MRR@10` — reciprocal rank of the first correct chunk
- `nDCG@10` — graded, discounts correct-but-low results
- `version-correct@1` — does the top-1 chunk's version range contain the query's
  target version (the metric this whole project exists to move)

**A caveat on `version-correct@1` that matters for reading the table.** It measures two
different things depending on the row. Without the version filter it is a genuine
measurement of how often an unconstrained retriever surfaces the wrong release first —
measured at **0.523** for an off-the-shelf bi-encoder, i.e. a coin flip. With the filter
on it is near-tautological, because the filter only returns chunks whose range covers
the target; the only way to score below 1.0 is to return nothing.

So the filter's jump in that column quantifies the size of the problem and confirms the
filter is wired up — it is not a modelling result. Among filtered rows the
discriminating metric is `nDCG@10`, which asks whether the *right* chunk ranks first
among the ones that are version-valid. That is what fine-tuning and reranking have to
move, and it is the number to hold them to.

## Deliberate non-goals

- **Not a general K8s chatbot.** Out-of-scope questions should be refused, and the
  refusal rate on unanswerable questions is a reported metric, not a bug.
- **No LLM-generated training labels.** Positives and hard negatives come from the
  corpus's own structure. An LLM-labelled set would make the eval circular.
- **No GPU required to run the system.** Fine-tuning wants one for an afternoon;
  ingestion, retrieval, eval, and serving are all CPU-only.
