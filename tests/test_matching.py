from __future__ import annotations

import numpy as np

from busreid.matching import MatchingEngine
from busreid.types import EmbeddingRecord, MatchStatus


def record(event_id: int, time: float, vector: list[float], passenger: str | None = None) -> EmbeddingRecord:
    return EmbeddingRecord(event_id, f"track{event_id}", time, np.asarray(vector, dtype=np.float32), passenger)


def test_fixture_same_person_matches_unverified() -> None:
    decisions = MatchingEngine(0.8, 0.05).match_batch(
        [record(1, 1, [1, 0], "P001")], [record(2, 5, [0.99, 0.01])]
    )
    assert decisions[0].in_event_id == 1
    assert decisions[0].status == MatchStatus.UNVERIFIED


def test_non_match_rejected_as_unknown() -> None:
    decision = MatchingEngine(0.8).match_batch([record(1, 1, [1, 0])], [record(2, 5, [0, 1])])[0]
    assert decision.status == MatchStatus.UNKNOWN
    assert decision.in_event_id is None


def test_out_before_in_does_not_match() -> None:
    decision = MatchingEngine(0.1).match_batch([record(1, 10, [1, 0])], [record(2, 5, [1, 0])])[0]
    assert decision.status == MatchStatus.UNKNOWN
    assert "IN trước OUT" in decision.reason


def test_one_to_one_assignment() -> None:
    decisions = MatchingEngine(0.7, 0.01).match_batch(
        [record(1, 1, [1, 0]), record(2, 2, [0, 1])],
        [record(3, 4, [0.99, 0.01]), record(4, 5, [0.01, 0.99])],
    )
    assert {item.in_event_id for item in decisions} == {1, 2}


def test_similar_clothing_ambiguous_returns_unknown() -> None:
    decision = MatchingEngine(0.7, 0.08).match_batch(
        [record(1, 1, [1, 0]), record(2, 2, [0.99, 0.02])], [record(3, 5, [1, 0.01])]
    )[0]
    assert decision.status == MatchStatus.UNKNOWN
    assert "tương đồng" in decision.reason


def test_paired_gt_can_be_confirmed() -> None:
    decision = MatchingEngine(0.8).match_batch(
        [record(1, 1, [1, 0])], [record(2, 2, [1, 0])], paired_and_verified=True
    )[0]
    assert decision.status == MatchStatus.CONFIRMED

