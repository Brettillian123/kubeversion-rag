"""Guards on retriever-in-the-loop negative mining.

The bug this exists to prevent has already happened twice in this project, in two
different forms, and both times it was invisible: the reranker's training negatives came
from a population it does not meet at inference, training loss looked excellent, and
end-to-end retrieval got worse. These assert the properties that keep the mined negatives
matched to what the retriever actually returns.
"""

from __future__ import annotations

import pytest

from kubeversion_rag.dataset.mine_negatives import mine_retriever_negatives
from kubeversion_rag.models import Chunk, Corpus, Example, RetrievedChunk
from kubeversion_rag.train.train_crossencoder import _build_pairs
from kubeversion_rag.versions import MinorVersion


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        chunks=[
            # Two versions of the same section: one family, one legitimate answer each.
            Chunk("api.md", ("Deprecations",), "served until 1.27", v("1.24"), v("1.27")),
            Chunk("api.md", ("Deprecations",), "removed in 1.28", v("1.28"), v("1.31")),
            # A sibling section of the same document -- the population that broke the
            # second attempt and that uniform sampling never produces.
            Chunk("api.md", ("Migration",), "how to migrate", v("1.24"), v("1.31"), part=1),
            Chunk("other.md", ("Storage",), "unrelated topic", v("1.24"), v("1.31")),
        ]
    )


class FakePipeline:
    """Returns a fixed candidate list, so mining is tested without a model."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.calls: list[int] = []

    def retrieve_many(self, queries, versions, top_k=None):
        self.calls.append(len(queries))
        ordered = list(self.corpus)
        return [
            [RetrievedChunk(chunk=chunk, score=1.0) for chunk in ordered[: top_k or len(ordered)]]
            for _ in queries
        ]


def example_for(corpus: Corpus) -> Example:
    positive = list(corpus)[0]
    return Example(
        question="Is the API served on 1.26?",
        target_version=v("1.26"),
        positive_chunk_id=positive.chunk_id,
        family_id=positive.family_id,
    )


class TestMining:
    def test_same_family_chunks_are_never_mined_as_negatives(self, corpus):
        # They are the same section at another version. Several of them are the correct
        # answer at their own version, so labelling them irrelevant teaches the model
        # that the right section is wrong.
        example = example_for(corpus)
        positive = corpus.get(example.positive_chunk_id)
        updated, _ = mine_retriever_negatives(corpus, [example], FakePipeline(corpus))

        mined = [corpus.get(cid) for cid in updated[0].retriever_negative_ids]
        assert mined, "mining produced nothing"
        assert all(chunk.family_id != positive.family_id for chunk in mined)

    def test_sibling_sections_of_the_same_document_are_mined(self, corpus):
        # The whole point of the step. When the reranker demoted the correct chunk, the
        # winner was a different section of the same document 46% of the time, and that
        # population was absent from training.
        example = example_for(corpus)
        positive = corpus.get(example.positive_chunk_id)
        updated, _ = mine_retriever_negatives(corpus, [example], FakePipeline(corpus))

        mined = [corpus.get(cid) for cid in updated[0].retriever_negative_ids]
        assert any(chunk.doc_path == positive.doc_path for chunk in mined)

    def test_unanswerable_examples_are_left_alone(self, corpus):
        # There is no positive, so there is nothing for a negative to be relative to.
        refusal = Example(
            question="What is the airspeed velocity of a swallow?",
            target_version=v("1.30"),
            positive_chunk_id="",
            unanswerable=True,
        )
        pipeline = FakePipeline(corpus)
        updated, _ = mine_retriever_negatives(corpus, [refusal], pipeline)
        assert updated[0].retriever_negative_ids == ()
        assert pipeline.calls == [], "unanswerable questions should not be retrieved for"

    def test_keep_bounds_the_negatives_per_example(self, corpus):
        example = example_for(corpus)
        updated, counts = mine_retriever_negatives(corpus, [example], FakePipeline(corpus), keep=1)
        assert len(updated[0].retriever_negative_ids) == 1
        assert counts["negatives"] == 1

    def test_mined_ids_round_trip_through_serialisation(self, corpus):
        # They live in the split files between `dataset mine-negatives` and
        # `train crossencoder`. Dropping them on write would silently return the
        # reranker to the distribution that measured 0.664 against 0.774.
        example = example_for(corpus)
        updated, _ = mine_retriever_negatives(corpus, [example], FakePipeline(corpus))
        restored = Example.from_dict(updated[0].to_dict())
        assert restored.retriever_negative_ids == updated[0].retriever_negative_ids


class TestPairBuilding:
    def test_retriever_negatives_reach_the_training_pairs(self, corpus):
        import random

        example = example_for(corpus)
        updated, _ = mine_retriever_negatives(corpus, [example], FakePipeline(corpus))
        rows, counts = _build_pairs(
            corpus,
            updated,
            negatives_per_positive=4,
            rng=random.Random(0),
            random_negatives_per_positive=0,
            retriever_negatives_per_positive=6,
        )
        assert counts["retriever_negatives"] > 0
        assert counts["missing_retriever_negatives"] == 0
        assert any(row["label"] == 1.0 for row in rows)

    def test_examples_without_mined_negatives_are_counted_not_silently_dropped(self, corpus):
        # Skipping `dataset mine-negatives` produces exactly this state. It has to be
        # loud, because the resulting model trains cleanly and serves badly.
        import random

        _, counts = _build_pairs(
            corpus,
            [example_for(corpus)],
            negatives_per_positive=4,
            rng=random.Random(0),
            random_negatives_per_positive=0,
            retriever_negatives_per_positive=6,
        )
        assert counts["missing_retriever_negatives"] == 1
