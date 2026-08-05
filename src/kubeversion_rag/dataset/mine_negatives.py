"""Mine cross-encoder negatives from the retriever that will actually feed it.

**Why this step exists, in the order the lessons arrived.**

A reranker is trained on (query, passage) pairs and deployed on whatever the retriever
hands it. If those two populations differ, training metrics stay excellent and end-to-end
retrieval degrades, with nothing in the logs to say why. This project hit that twice,
and each time the assumption about the serving distribution was wrong in a different way.

*First attempt* -- negatives were same-family, wrong-version chunks only. The reasoning
was that positives and negatives from the same section force the model to learn the
version signal rather than topic similarity. It did exactly that, separating them by 19
points. End-to-end nDCG@10 then fell from 0.774 to 0.233. The model had never scored a
passage from another document, so its scores there were arbitrary rather than low, and
one high score among 49 competitors is enough to lose.

*Second attempt* -- added uniformly-sampled negatives, on the theory that the top-50 is
mostly unrelated chunks. That recovered most of the loss (0.233 -> 0.664) but still sat
below no reranker at all. Measuring the real top-50 instead of assuming it showed why:
it is 98% other families, so the sampling target was right, but when the reranker
demoted the correct chunk the winner was *a different section of the same document* 46%
of the time. A uniform draw from 23k chunks essentially never lands in the same document,
so that population -- the one the improved retriever concentrates -- was absent from
training.

The general form of the mistake is assuming the candidate distribution. The general fix
is to stop assuming: retrieve with the fine-tuned bi-encoder under the same version
filter used at serving time, and take what it returns. That is what this module does.

Same-family version negatives are still mined separately and kept, because they are only
~2% of a retrieved top-50 -- too rare for the model to learn the version distinction from
the retriever's output alone. The two sources are complementary: retrieved negatives
supply the distribution, family negatives supply the signal.

Ordering matters and is enforced by the CLI: ``train biencoder`` -> ``dataset
mine-negatives`` -> ``train crossencoder``. Mining against the base bi-encoder would
reintroduce the same class of mismatch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from ..models import Corpus, Example

log = logging.getLogger(__name__)

BATCH = 256


def mine_retriever_negatives(
    corpus: Corpus,
    examples: Sequence[Example],
    pipeline,  # ResolvedPipeline; untyped to keep torch out of this module's imports
    top_k: int = 30,
    keep: int = 8,
) -> tuple[list[Example], dict[str, int]]:
    """Attach each example's hardest *retrieved* negatives.

    Anything sharing the positive's family is dropped rather than kept as a negative:
    those chunks are the same section at another version, and several of them legitimately
    answer the question at their own version. They are handled by the family-negative
    miner, which knows which version each one is correct for. Labelling them irrelevant
    here would teach the model that the right section is wrong.
    """
    counts = {"examples": 0, "negatives": 0, "no_positive": 0, "empty": 0}
    stats = {"same_document": 0, "other_document": 0}

    answerable = [example for example in examples if not example.unanswerable]
    mined: dict[int, tuple[str, ...]] = {}

    for start in range(0, len(answerable), BATCH):
        batch = answerable[start : start + BATCH]
        results = pipeline.retrieve_many(
            [example.question for example in batch],
            [example.target_version for example in batch],
            top_k=top_k,
        )
        for example, hits in zip(batch, results, strict=True):
            positive = corpus.get(example.positive_chunk_id)
            if positive is None:
                counts["no_positive"] += 1
                continue
            negatives = [hit.chunk for hit in hits if hit.chunk.family_id != positive.family_id][
                :keep
            ]
            if not negatives:
                counts["empty"] += 1
            for chunk in negatives:
                key = "same_document" if chunk.doc_path == positive.doc_path else "other_document"
                stats[key] += 1
            mined[id(example)] = tuple(chunk.chunk_id for chunk in negatives)
            counts["negatives"] += len(negatives)
            counts["examples"] += 1

        log.info("mined %d/%d", min(start + BATCH, len(answerable)), len(answerable))

    updated = [
        replace(example, retriever_negative_ids=mined.get(id(example), ())) for example in examples
    ]

    total = stats["same_document"] + stats["other_document"]
    if total:
        # The number this step exists for. Uniform sampling produces essentially 0% here;
        # if this is also near zero the mining is not doing its job and the reranker will
        # keep confusing sibling sections.
        log.info(
            "mined negatives: %.1f%% from the same document as the positive, %.1f%% elsewhere",
            100 * stats["same_document"] / total,
            100 * stats["other_document"] / total,
        )
    return updated, counts
