"""Matching is what decides whether the system says 'that is you'."""

from __future__ import annotations

import numpy as np
import pytest

from reconvision.domain.matching import (
    GalleryMatcher,
    ThresholdPolicy,
    cosine_similarity,
    normalise,
)
from reconvision.domain.models import GalleryEntry

from .conftest import nearby_embedding, random_embedding

PERMISSIVE = ThresholdPolicy(match_threshold=0.4, min_margin=0.0)


def test_normalise_produces_a_unit_vector(rng: np.random.Generator) -> None:
    scaled = random_embedding(rng) * 17.0

    assert float(np.linalg.norm(normalise(scaled))) == pytest.approx(1.0)


def test_normalise_rejects_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        normalise(np.zeros(512, dtype=np.float32))


def test_a_vector_is_perfectly_similar_to_itself(rng: np.random.Generator) -> None:
    embedding = random_embedding(rng)

    assert cosine_similarity(embedding, embedding) == pytest.approx(1.0, abs=1e-6)


def test_similarity_ignores_magnitude(rng: np.random.Generator) -> None:
    """Only direction carries identity, so an unnormalised input must not change it."""
    embedding = random_embedding(rng)

    assert cosine_similarity(embedding, embedding * 42.0) == pytest.approx(1.0, abs=1e-6)


def test_nearby_embedding_helper_hits_the_requested_similarity(
    rng: np.random.Generator,
) -> None:
    """Guards the fixture itself: the rest of the suite trusts its geometry."""
    base = random_embedding(rng)

    assert cosine_similarity(base, nearby_embedding(base, rng, 0.73)) == pytest.approx(
        0.73, abs=1e-5
    )


def test_an_empty_gallery_matches_nobody(rng: np.random.Generator) -> None:
    """The state the system ships in: nothing enrolled, so nothing is recognised."""
    matcher = GalleryMatcher(entries=[], policy=PERMISSIVE)

    result = matcher.match(random_embedding(rng))

    assert matcher.is_empty
    assert not result.is_match


def test_another_capture_of_an_enrolled_face_matches(rng: np.random.Generator) -> None:
    enrolled = random_embedding(rng)
    matcher = GalleryMatcher([GalleryEntry("jeremie", enrolled)], PERMISSIVE)

    result = matcher.match(nearby_embedding(enrolled, rng, similarity=0.7))

    assert result.identity_id == "jeremie"
    assert result.similarity == pytest.approx(0.7, abs=1e-5)


def test_a_stranger_does_not_match(rng: np.random.Generator) -> None:
    matcher = GalleryMatcher([GalleryEntry("jeremie", random_embedding(rng))], PERMISSIVE)

    result = matcher.match(random_embedding(rng))

    assert not result.is_match
    assert result.identity_id is None


def test_an_identity_scores_by_its_closest_photo_not_its_average(
    rng: np.random.Generator,
) -> None:
    """Enrolment spans lighting and angles. Averaging those vectors would describe
    a face that was never photographed; the nearest single shot is the real signal."""
    daylight = random_embedding(rng)
    entries = [
        GalleryEntry("jeremie", daylight),
        GalleryEntry("jeremie", random_embedding(rng)),
        GalleryEntry("jeremie", random_embedding(rng)),
    ]
    matcher = GalleryMatcher(entries, PERMISSIVE)

    result = matcher.match(nearby_embedding(daylight, rng, similarity=0.75))

    assert result.identity_id == "jeremie"
    assert result.similarity == pytest.approx(0.75, abs=1e-5)


def test_the_closer_of_two_enrolled_people_wins(rng: np.random.Generator) -> None:
    jeremie = random_embedding(rng)
    matcher = GalleryMatcher(
        [GalleryEntry("jeremie", jeremie), GalleryEntry("sibling", random_embedding(rng))],
        PERMISSIVE,
    )

    result = matcher.match(nearby_embedding(jeremie, rng, similarity=0.8))

    assert result.identity_id == "jeremie"
    assert result.runner_up_identity_id == "sibling"
    assert result.margin > 0


def test_a_score_below_the_threshold_is_unknown(rng: np.random.Generator) -> None:
    enrolled = random_embedding(rng)
    strict = ThresholdPolicy(match_threshold=0.75, min_margin=0.0)
    matcher = GalleryMatcher([GalleryEntry("jeremie", enrolled)], strict)

    result = matcher.match(nearby_embedding(enrolled, rng, similarity=0.6))

    assert not result.is_match
    assert result.similarity == pytest.approx(0.6, abs=1e-5)


def test_two_people_scoring_almost_alike_is_reported_unknown(
    rng: np.random.Generator,
) -> None:
    """The case a threshold alone gets wrong. Both siblings clear 0.4, and the
    winner leads by 0.02 - an answer too close to call, so it must not be called."""
    shared = random_embedding(rng)
    probe = nearby_embedding(shared, rng, similarity=0.9)
    matcher = GalleryMatcher(
        [
            GalleryEntry("twin_a", nearby_embedding(probe, rng, similarity=0.62)),
            GalleryEntry("twin_b", nearby_embedding(probe, rng, similarity=0.60)),
        ],
        ThresholdPolicy(match_threshold=0.4, min_margin=0.05),
    )

    result = matcher.match(probe)

    assert not result.is_match
    assert result.margin < 0.05


def test_a_thin_margin_cannot_veto_the_only_enrolled_person(
    rng: np.random.Generator,
) -> None:
    """With one identity there is no runner-up, so the margin rule has nothing to
    compare and must stay out of the way."""
    enrolled = random_embedding(rng)
    matcher = GalleryMatcher(
        [GalleryEntry("jeremie", enrolled)],
        ThresholdPolicy(match_threshold=0.4, min_margin=0.5),
    )

    result = matcher.match(nearby_embedding(enrolled, rng, similarity=0.7))

    assert result.identity_id == "jeremie"


def test_gallery_reports_who_is_enrolled(rng: np.random.Generator) -> None:
    matcher = GalleryMatcher(
        [
            GalleryEntry("jeremie", random_embedding(rng)),
            GalleryEntry("jeremie", random_embedding(rng)),
            GalleryEntry("guest", random_embedding(rng)),
        ],
        PERMISSIVE,
    )

    assert matcher.known_identity_ids == ("jeremie", "guest")
    assert matcher.entry_count == 3


@pytest.mark.parametrize("threshold", [-1.5, 1.5])
def test_a_threshold_outside_cosine_range_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError, match="cosine similarity"):
        ThresholdPolicy(match_threshold=threshold)


def test_a_negative_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ThresholdPolicy(match_threshold=0.4, min_margin=-0.1)
