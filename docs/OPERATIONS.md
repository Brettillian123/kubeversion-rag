# Operations

## Runbook: swapping the embedding model

The scenario this exists for: `bge-small` is being replaced by `bge-base`, and the
index cannot be rebuilt in place because vectors from two different models are not
comparable — a half-migrated collection returns nonsense, not degraded results.

```bash
export MODEL=BAAI/bge-base-en-v1.5

python scripts/migrate_embedding_model.py plan     --model $MODEL --revision 2
python scripts/migrate_embedding_model.py backfill --model $MODEL --revision 2
python scripts/migrate_embedding_model.py evaluate --model $MODEL --revision 2
python scripts/migrate_embedding_model.py promote  --model $MODEL --revision 2
```

`plan` refuses if the target collection name equals the live one — otherwise the
backfill would rewrite the collection currently serving traffic. Bump `--revision`.

`evaluate` exits non-zero if `version_correct@1` regressed, and `promote` refuses to
run without a passing report. Skipping the gate is possible (`--skip-gate`) but it is
the whole point of the flow, so it should be a deliberate, explained act.

Rollback is one command and takes one rollout, because nothing was deleted:

```bash
python scripts/migrate_embedding_model.py rollback --to chunks__baai_bge_small_en_v1_5__v1
```

**Delete the old collection only after a soak.** It is the rollback target; reclaiming
its disk the same day removes the only cheap way out of a bad promotion.

---

## What to watch

| Signal | Metric | Why it matters |
|---|---|---|
| Answers are silently assuming "latest" | `kvrag_version_source_total{source="defaulted"}` rising as a share | Users are not stating their version. A version-aware system that mostly defaults is a version-blind system with extra steps. |
| The corpus stopped updating | `kvrag_active_collection_points` flat across a release | The ingest CronJob is failing, or `concurrencyPolicy: Forbid` is suppressing runs because one is wedged. |
| Retrieval quality regressed | `kvrag_refusals_total{reason="low_confidence"}` climbing | Either the corpus drifted or a promotion went bad. Check the last migration report. |
| Retrieval quality regressed, *silently* | `kvrag_refusals_total{reason="low_confidence"}` pinned at exactly zero | A gate that never fires reads the same as a gate nothing trips. `KVRAG_MIN_SCORE` is a **cosine** floor because serving does not rerank; setting it on the cross-encoder's scale (a negative number) makes the comparison unsatisfiable. The generator logs an error once at request time if it detects that — grep for `min_score=` in the API logs. |
| The model is inventing sources | `warnings.invalid_citations` non-empty in responses | Citations are validated per response; a sustained non-zero rate means the generation prompt needs work. |
| Backpressure | `kvrag_stage_seconds{stage="embed"}` p99 | The embedding service is the usual bottleneck; it saturates CPU before the API does. |

The two most useful alerts are **defaulted-version share** and **invalid citations**.
Both are silent-wrongness signals: the system keeps returning confident, well-formed
answers while doing the wrong thing, and neither shows up as an error rate.

---

## Debugging a wrong answer

Start by separating the two failure modes, because they have nothing in common:

```bash
curl "$API/debug/retrieve?question=Can+I+use+PodSecurityPolicy&version=1.24"
```

- **The right chunks came back, but the answer is wrong** → generation problem. Check
  `warnings`, then the system prompt.
- **The wrong chunks came back** → retrieval problem. Look at `covers_target` in each
  hit. If everything is `false`, the version filter is not being applied — check that
  `version_low`/`version_high` payload indexes exist on the collection.
- **Nothing came back** → either the collection is empty (readiness should have caught
  that) or the resolved version is outside the ingested window.

`version_source` in every `/ask` response tells you whether the version was stated,
inferred, or defaulted. A wrong answer with `version_source: defaulted` usually is not
a retrieval bug at all — the user did not say what they were running.

---

## Failure modes and what they look like

| Symptom | Likely cause | Fix |
|---|---|---|
| API pods `Ready 0/1` forever after install | No collection backfilled yet | Run the backfill Job. This is the probe working, not a bug. |
| API pods restarting in a loop during a Qdrant incident | Liveness was coupled to Qdrant | It is not, by design — if this happens, someone changed `livenessProbe` to hit `/readyz`. |
| Ingest CronJob never runs | A previous run is wedged and `Forbid` is suppressing new ones | `kubectl delete job` the stuck one; `activeDeadlineSeconds` should have caught it. |
| Embedding pods OOM under load | Batch size × text length exceeded memory | `KVRAG_EMBED_MAX_BATCH` and `KVRAG_EMBED_MAX_CHARS` bound both; lower them or raise the limit. |
| Ablation table shows fine-tuned rows with no gain | The models were never trained and the run fell back to base | The table marks these rows explicitly. Train first. |
| `helm template` fails with "image.tag must be set" | Working as intended | Pass `--set image.tag=$(git rev-parse --short HEAD)`. |

---

## Security notes

- **The API key never enters the chart's values.** It comes from an existing Secret
  referenced by name. A key passed via `--set` ends up in the Helm release object,
  shell history, and CI logs.
- **Egress from the API pod is port-restricted and excludes RFC1918 plus link-local.**
  NetworkPolicy has no DNS-name selector, so "only api.anthropic.com" is not
  expressible; the link-local exclusion at least blocks the cloud metadata endpoint,
  which is the usual next hop after an SSRF. Tightening further needs an egress gateway.
- **Service account tokens are not mounted.** Nothing here calls the Kubernetes API, so
  the token is pure attack surface.
- **User questions are untrusted input that reaches a model prompt.** The mitigations
  are structural rather than prompt-based: retrieved context comes only from the
  indexed corpus, citations are validated against the blocks actually supplied, and the
  answer path has no tools and no side effects. A successful injection can make the
  answer wrong; it cannot make the system *do* anything.
- **Qdrant has no authentication in the default chart.** It is protected by
  NetworkPolicy alone. If your cluster does not enforce NetworkPolicy — some CNIs
  silently ignore it — set `KVRAG_QDRANT_API_KEY` and enable Qdrant's own auth.
