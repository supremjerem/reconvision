"""Temporal voting is what turns 200 flickering frames into one usable event."""

from __future__ import annotations

import pytest

from reconvision.domain.models import MatchResult
from reconvision.domain.smoothing import TrackVote, VotePolicy


def matched(identity_id: str, similarity: float = 0.7) -> MatchResult:
    return MatchResult(identity_id=identity_id, similarity=similarity, runner_up_similarity=0.1)


def unmatched() -> MatchResult:
    return MatchResult(identity_id=None, similarity=0.2, runner_up_similarity=0.1)


def test_a_fresh_track_has_decided_nothing() -> None:
    verdict = TrackVote().verdict()

    assert not verdict.is_conclusive
    assert verdict.observations == 0


def test_one_frame_is_never_enough() -> None:
    """A lone confident frame is exactly the false positive this stage exists to stop."""
    vote = TrackVote()
    vote.observe(matched("jeremie"), weight=1.0)

    assert not vote.verdict().is_conclusive


def test_a_consistently_recognised_track_resolves_to_that_person() -> None:
    vote = TrackVote()
    for _ in range(5):
        vote.observe(matched("jeremie"), weight=0.9)

    verdict = vote.verdict()

    assert verdict.is_known_person
    assert verdict.identity_id == "jeremie"
    assert verdict.confidence == pytest.approx(1.0)


def test_a_consistently_unrecognised_track_resolves_to_a_stranger() -> None:
    vote = TrackVote()
    for _ in range(5):
        vote.observe(unmatched(), weight=0.9)

    verdict = vote.verdict()

    assert verdict.is_conclusive
    assert verdict.identity_id is None
    assert not verdict.is_known_person


def test_one_stray_frame_does_not_overturn_a_track() -> None:
    """The core promise: a person recognised in most frames stays recognised."""
    vote = TrackVote()
    for _ in range(9):
        vote.observe(matched("jeremie"), weight=0.9)
    vote.observe(matched("sibling"), weight=0.9)

    assert vote.verdict().identity_id == "jeremie"


def test_a_genuinely_split_track_is_reported_inconclusive() -> None:
    """Half the frames say one sibling, half say the other. Reporting either would
    be inventing an answer, so the track declines to decide."""
    vote = TrackVote()
    for _ in range(4):
        vote.observe(matched("twin_a"), weight=1.0)
        vote.observe(matched("twin_b"), weight=1.0)

    verdict = vote.verdict()

    assert not verdict.is_conclusive
    assert verdict.identity_id is None


def test_good_frames_outweigh_bad_ones() -> None:
    """Two sharp close-up frames beat six distant blurry ones that disagree, which
    is what quality weighting is for."""
    vote = TrackVote(policy=VotePolicy(min_observations=3, min_weight_share=0.6))
    for _ in range(2):
        vote.observe(matched("jeremie"), weight=1.0)
    for _ in range(6):
        vote.observe(unmatched(), weight=0.05)

    assert vote.verdict().identity_id == "jeremie"


def test_a_track_of_unusable_faces_decides_nothing() -> None:
    """Every frame carried zero weight, so there is no evidence to weigh."""
    vote = TrackVote()
    for _ in range(10):
        vote.observe(unmatched(), weight=0.0)

    verdict = vote.verdict()

    assert not verdict.is_conclusive
    assert verdict.observations == 10


def test_a_track_counts_the_frames_it_has_seen() -> None:
    """The pipeline polls this to know when a track has enough evidence to emit."""
    vote = TrackVote()
    for _ in range(7):
        vote.observe(unmatched(), weight=0.5)

    assert vote.observations == 7


def test_the_best_similarity_seen_is_kept_for_later_calibration() -> None:
    vote = TrackVote()
    vote.observe(matched("jeremie", similarity=0.55), weight=1.0)
    vote.observe(matched("jeremie", similarity=0.81), weight=1.0)
    vote.observe(matched("jeremie", similarity=0.62), weight=1.0)

    assert vote.best_similarity == pytest.approx(0.81)


def test_evidence_requirements_are_configurable() -> None:
    lenient = TrackVote(policy=VotePolicy(min_observations=2, min_weight_share=0.51))
    lenient.observe(matched("jeremie"), weight=1.0)
    lenient.observe(matched("jeremie"), weight=1.0)

    assert lenient.verdict().is_known_person


def test_a_negative_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TrackVote().observe(matched("jeremie"), weight=-1.0)


@pytest.mark.parametrize(
    ("observations", "share"),
    [(0, 0.6), (3, 0.0), (3, 1.5)],
)
def test_an_impossible_vote_policy_is_refused(observations: int, share: float) -> None:
    with pytest.raises(ValueError):
        VotePolicy(min_observations=observations, min_weight_share=share)
