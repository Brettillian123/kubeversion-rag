# kubeversion-rag

Version-aware retrieval over the Kubernetes documentation, with a measured ablation.

**The problem.** `kubernetes/website` keeps the same document path on every release
branch. The deprecation guide from `release-1.26` and from `release-1.34` are ~90%
lexically identical and give contradictory answers about which API versions are
served. Off-the-shelf sentence embeddings rank them almost identically, so a naive
RAG system confidently serves 1.26 guidance to a 1.34 cluster.

**The fix, and the point of the project.** The wrong-version snapshot of the same
section is a near-perfect hard negative — maximally similar in embedding space,
definitively wrong. That makes it possible to mine a labelled retrieval dataset out of
the corpus's own structure, fine-tune a bi-encoder and a cross-encoder reranker on it,
and report an ablation showing what each component actually bought.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and
[`docs/RESULTS.md`](docs/RESULTS.md) for the measured numbers.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[ml,serve,dev]"
```

Ingest the corpus (one sparse, blobless clone; ~10 min on a warm network):

```bash
kvrag ingest --min-version 1.24 --max-version 1.35
```

Build the training set and run the baseline evaluation — no models trained yet:

```bash
kvrag dataset build
kvrag eval run --configs bm25,dense_offtheshelf,dense_filtered
```

Fine-tune and re-evaluate:

```bash
kvrag train biencoder
kvrag train crossencoder
kvrag eval run --all --write-results
```

## Layout

| Path | What lives there |
|---|---|
| `src/kubeversion_rag/ingest/` | Sparse multi-branch fetch, heading-aware chunking, deprecation-guide parsing |
| `src/kubeversion_rag/dataset/` | Question generation, hard-negative mining, leakage-safe splits |
| `src/kubeversion_rag/retrieval/` | BM25, dense retrieval, version filtering, cross-encoder reranking |
| `src/kubeversion_rag/eval/` | Metrics and the ablation runner |
| `src/kubeversion_rag/train/` | Bi-encoder (MNRL) and cross-encoder training |
| `src/kubeversion_rag/serving/` | Embedding service and the RAG API |
| `deploy/` | Dockerfiles, compose, and the Helm chart |
| `scripts/` | Zero-downtime embedding-model migration |

## Running it on Kubernetes

```bash
helm install kvrag deploy/helm/kubeversion-rag \
  --set image.tag=$(git rev-parse --short HEAD) \
  --set anthropic.existingSecret=kvrag-anthropic
```

The API reads its active Qdrant collection from a ConfigMap at request time, which is
what makes `scripts/migrate_embedding_model.py` able to swap embedding models without
a code deploy or downtime. See `docs/ARCHITECTURE.md § Zero-downtime embedding-model
migration`.

## Honest limitations

- **Generated questions are templated.** They exercise version-sensitivity precisely,
  but their phrasing is narrower than real user questions. The hand-written gold set
  exists to keep the headline numbers honest; it is small.
- **Corpus is English docs only.** Localised trees are excluded.
- **The reranker needs a GPU to train** in reasonable time. Everything else — ingest,
  retrieval, eval, serving — is CPU-only.
