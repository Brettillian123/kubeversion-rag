# Results

Hand-written analysis. The generated tables live in [`docs/results/`](results/) and are
rewritten by `kvrag eval run --write-results`; this file is not.

| Generated table | Question set |
|---|---|
| [`results/ablation__test.md`](results/ablation__test.md) | 1,811 templated questions, families disjoint from training |
| [`results/ablation__gold.md`](results/ablation__gold.md) | 11 hand-written questions in human phrasing |

## The headline

Held-out test split, 1,811 questions, split **by document family** so no section seen in
training appears in test.

| Configuration | recall@1 | recall@10 | MRR@10 | nDCG@10 | version-correct@1 |
|---|---:|---:|---:|---:|---:|
| BM25, no version awareness | 0.168 | 0.554 | 0.287 | 0.351 | 0.632 |
| Off-the-shelf bi-encoder, no version awareness | 0.155 | 0.585 | 0.290 | 0.361 | **0.518** |
| Off-the-shelf bi-encoder + version filter | 0.351 | 0.704 | 0.465 | 0.523 | 1.000 |
| **Fine-tuned bi-encoder + version filter** | **0.593** | **0.964** | **0.714** | **0.774** | 1.000 |
| … + fine-tuned cross-encoder | 0.572 | 0.937 | 0.684 | 0.744 | 1.000 |
| … + BM25 union | 0.572 | 0.935 | 0.682 | 0.743 | 1.000 |

Two things to take from it.

**The premise held.** An off-the-shelf dense retriever puts the wrong release's snapshot
first 48% of the time — worse than BM25, which at least reacts to the words that differ.
Dense embeddings collapse two near-identical snapshots into neighbouring points and then
choose between them arbitrarily.

**Fine-tuning is where the gain is.** nDCG@10 0.523 → 0.774, recall@10 0.704 → 0.964, on
families never seen in training. The hard negatives that made it possible were mined from
the corpus's own structure — the same section at a wrong version — with no LLM labelling
anywhere, which matters because an LLM-labelled test set would make the evaluation
circular.

**The reranker is the row worth stopping on, because it is the one that failed.**

## What the reranker cost

The architectural argument for a cross-encoder is sound. A bi-encoder compresses a passage
into one vector *before* it sees the query; two snapshots of the same section differ by a
handful of tokens, and that difference rarely survives the compression. A cross-encoder
reads both together, so one contradicting sentence can dominate.

It still made retrieval worse, three times, and each attempt failed for a different reason.
The architecture was never the problem. **Where the negatives came from was.**

| Reranker trained on | nDCG@10 | vs. no reranker |
|---|---:|---:|
| *(no reranker — fine-tuned bi-encoder alone)* | **0.774** | — |
| Same-family wrong-version negatives only | 0.233 | **−0.541** |
| … + uniformly sampled from the corpus | 0.664 | −0.110 |
| … + mined from the fine-tuned retriever's own output | 0.744 | −0.030 |

Every one of those models trained cleanly. Loss fell, dev loss fell, and the margin between
positives and the negatives it was shown was excellent. **None of that can see this class of
bug**, because all of it is computed on the same skewed distribution the model was trained
on. The only thing that detected it was end-to-end retrieval on held-out questions.

### Round 1 — hard negatives only, and the 19-point margin that meant nothing

Negatives were the same section at a wrong version. The reasoning: positives and negatives
from the same section force the model to learn the version signal rather than topic
similarity. It learned exactly that.

| population | mean | p50 | p90 |
|---|---:|---:|---:|
| positive | +8.88 | +9.23 | +10.24 |
| same-family, wrong version | −9.95 | −10.70 | −7.58 |
| random unrelated chunk | −4.68 | −10.82 | **+8.62** |

A 19-point separation on exactly the distinction the project is about — and nDCG@10 fell to
0.233, a third of no reranker at all.

The third row is the bug. The model had never scored a passage from another document, so
its scores there were *arbitrary rather than low*: median −10.8, but a p90 of +8.62,
overlapping the positives. It beat a random chunk 92.5% of the time, which sounds strong
until you notice there are 49 of them in a top-50: 0.925⁴⁹ ≈ 2%.

### Round 2 — the right target, still the wrong population

Adding uniformly-sampled negatives fixed the calibration it was diagnosed to fix. The
unrelated-chunk p90 went from +8.62 to −11.33, and P(positive > random) from 92.5% to
100.0%. nDCG@10 recovered 0.233 → 0.664.

Still below 0.774. So rather than theorise about the candidate distribution a third time, I
measured it — what the fine-tuned retriever *actually* returns for 200 held-out questions:

| what a real top-50 contains | share |
|---|---:|
| version-variants of the correct chunk | 2.0% |
| everything else | 98.0% |

| rank of the correct chunk | mean | @1 | @3 |
|---|---:|---:|---:|
| dense | 2.83 | 58.3% | 80.9% |
| after reranking | 3.84 | 48.2% | 72.9% |

**When the reranker demoted the correct chunk, the winner was a different section of the
same document 45.7% of the time** — scoring +9.67 against the positive's +9.34.

That population is absent from uniform sampling: a random draw from 23,018 chunks
essentially never lands in the same document. And it is exactly the population an *improved*
retriever concentrates. The bi-encoder fine-tune made the system good enough to find the
right document, which handed the reranker the one discrimination it had no training signal
for. **Fixing the retriever moved the reranker's serving distribution out from under it.**

### Round 3 — stop guessing the distribution

`kvrag dataset mine-negatives` retrieves with the fine-tuned bi-encoder under the serving
version filter and keeps what comes back. The training candidates become the serving
candidates by construction.

The measurement that says it worked: **50.6% of mined negatives come from the same document
as the positive**, against ~0% under uniform sampling. Training loss went *up* — 0.193 →
0.734 — because the negatives are no longer trivially separable, which is the point.

nDCG@10 0.664 → 0.744. Promoted 42 queries, demoted 43, mean rank 2.86 against dense's 2.83.

**Still −0.030 against no reranker.** Three rounds of increasingly correct negative sampling
converge toward the bi-encoder's own ordering without beating it.

### The conclusion, which is negative

**On this corpus, after the bi-encoder fine-tune, the cross-encoder does not pay for
itself.** It is roughly neutral on ranking quality and costs a forward pass over 50 passages
per query. The honest recommendation is to serve without it.

That is not the result I expected, and the reason it is worth reporting is that the
bi-encoder fine-tune is what made it true. At recall@10 0.964 and recall@1 0.593 there is
very little headroom left in the top-50 for a 6-layer MiniLM to find. A reranker earns its
cost when the retriever in front of it is mediocre. This one is not, any more.

Deliberately not claimed: that cross-encoders don't help at version-sensitive retrieval in
general. A larger reranker, or one trained with a listwise ranking loss rather than pointwise
BCE, may well clear 0.774. What is established is narrower and better supported — that *this*
reranker, trained three ways, does not.

## The gold set, and an inversion worth flagging

The generated questions are templated. The 14 hand-written questions (11 answerable) exist
to check the fine-tune learned version-sensitivity rather than a phrasing pattern.

| Configuration | recall@10 | MRR@10 | nDCG@10 | version-correct@1 |
|---|---:|---:|---:|---:|
| Off-the-shelf bi-encoder, no version awareness | 0.818 | 0.410 | 0.510 | 0.455 |
| Off-the-shelf bi-encoder + version filter | 0.909 | 0.625 | 0.694 | 1.000 |
| Fine-tuned bi-encoder + version filter | 0.818 | 0.773 | **0.785** | 1.000 |
| … + fine-tuned cross-encoder | **1.000** | **0.803** | **0.854** | 1.000 |

**The fine-tune generalizes.** nDCG@10 0.694 → 0.785 on phrasing it never trained on. The
gain is smaller than on the templated set (+0.09 against +0.25), which is what you would
expect and is the more honest of the two numbers.

**And the reranker helps here — the opposite sign from the test set.** 0.785 → 0.854, with
recall@10 reaching 1.000.

**This is 11 questions.** One question moving is nine points of nDCG. It is a signal, not a
result, and it is reported because suppressing an inconvenient direction would be worse than
reporting a noisy one.

The mechanism it hints at is at least coherent: the bi-encoder was fine-tuned on templated
phrasing and is near-ceiling there (recall@10 0.964), leaving nothing for a reranker to
recover. On human phrasing it is weaker (recall@10 0.818) and there is room. If that holds,
the reranker's value is concentrated exactly where the bi-encoder is out of distribution —
an argument for keeping it in production, where phrasing is human, even though the large
benchmark says it is not worth it. **Confirming that needs a gold set an order of magnitude
larger, and it has not been done.**

## A place the reranker would have earned its keep, and still doesn't

Ranking and *refusing* are different jobs, and the gold set separates them cleanly.

Best score for each gold question, by scale:

| | answerable (n=11) | expected refusal (n=3) | separable? |
|---|---|---|---|
| dense cosine | 0.563 – 0.818 | 0.532 – 0.592 | **no — they overlap** |
| cross-encoder logit | −9.61 – +6.03 | −10.10 – −9.87 | yes, by 0.26 |

**Cosine similarity cannot detect unanswerability on this corpus.** With 23,018 chunks
something is always topically close, and the nearest-neighbour score to a question about a
feature gate that does not exist looks like the score to a real but awkwardly-phrased one.

The cross-encoder can, because its score is a calibrated relevance logit rather than a
similarity — the same property that made it useless for reordering an already-good list.
**It is still not adopted**: 0.26 logits of margin measured on 14 questions is a threshold
that would not survive a fifteenth, and buying it costs a torch process in the API pod.

So `KVRAG_MIN_SCORE` is set *below* the lowest answerable question (0.35 against 0.563)
rather than tuned to separate the populations. It catches genuinely far-off queries only,
and real unanswerability is left to the generator's cite-or-refuse contract, which reads
the passages instead of scoring them.

This surfaced as a live bug rather than an open question: the constant was **−4.0**, a
sensible cross-encoder floor left behind from an earlier design in which serving reranked.
Serving does not rerank, so `best < −4.0` was never true and the low-confidence refusal was
unreachable. A gate that never fires is indistinguishable from a gate nothing trips, which
is why the fix is a runtime scale check that logs loudly rather than a corrected constant.

## Reading `version-correct@1` correctly

It asks whether the top-ranked chunk applies to the Kubernetes version in the question — the
failure this project exists to fix. It means two different things depending on the row:

- **Rows without the version filter** — a genuine measurement, and the number that sizes the
  problem: 0.518 means an unconstrained dense retriever is at a coin flip.
- **Rows with the filter** — near-tautological. The filter only returns chunks whose range
  covers the target, so ~1.0 is by construction and the only way to score lower is to return
  nothing at all.

The filter's jump in this column is not a modelling win. **Among filtered rows the
discriminating metric is nDCG@10.**

## Reading the rest of it honestly

- The split is **by document family**, so no section in the test set was seen in training.
  Splitting by question would leak: the same section generates several questions.
- Generated questions are templated, and the gold set that checks them is small.
- Unanswerable questions are excluded from retrieval metrics — there is nothing to retrieve
  — and are measured as a refusal rate in the serving tests instead.
- `recall@1` is reported alongside `recall@10` because they moved in opposite directions for
  the reranker at one point, and reporting only one would have hidden it.
