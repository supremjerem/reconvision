"""Calibration turns a guess about the threshold into a measurement.

Synthetic embeddings with controlled similarities, so the arithmetic of the
operating points is checked exactly rather than approximately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reconvision.application.evaluation import (
    LabelledEmbedding,
    evaluate,
    format_distribution,
    load_embeddings,
    save_embeddings,
    score_pairs,
)
from tests.unit.conftest import nearby_embedding, random_embedding


def population(
    rng: np.random.Generator,
    people: int = 20,
    photos: int = 4,
    within_person_similarity: float = 0.75,
) -> list[LabelledEmbedding]:
    """Several people, each photographed several times at a known similarity."""
    labelled: list[LabelledEmbedding] = []
    for index in range(people):
        base = random_embedding(rng)
        labelled.append(LabelledEmbedding(f"person_{index}", base))
        labelled += [
            LabelledEmbedding(
                f"person_{index}", nearby_embedding(base, rng, within_person_similarity)
            )
            for _ in range(photos - 1)
        ]
    return labelled


def test_pairs_are_split_into_same_and_different_person(
    rng: np.random.Generator,
) -> None:
    genuine, impostor = score_pairs(population(rng, people=3, photos=2))

    # Three people with two photos each: three same-person pairs, twelve others.
    assert genuine.size == 3
    assert impostor.size == 12


def test_a_face_is_never_compared_with_itself(rng: np.random.Generator) -> None:
    """Self-comparisons score 1.0 and would flatter the genuine distribution into
    looking far more separable than it is."""
    genuine, _ = score_pairs(population(rng, people=2, photos=3))

    assert np.all(genuine < 0.999)


def test_same_person_pairs_score_higher_than_different_person_pairs(
    rng: np.random.Generator,
) -> None:
    """The property everything else depends on. Without it no threshold exists."""
    report = evaluate(population(rng))

    assert report.genuine_scores.mean() > report.impostor_scores.mean()
    assert report.separation > 0.5


def test_operating_points_trade_recognition_against_wrong_identifications(
    rng: np.random.Generator,
) -> None:
    """Demanding fewer wrong names means a higher bar, which recognises fewer
    genuine faces. That trade-off is the whole content of the report."""
    report = evaluate(population(rng, people=40))
    points = {point.false_accept_rate: point for point in report.operating_points}

    assert points[1e-2].threshold < points[1e-4].threshold
    assert points[1e-2].true_accept_rate >= points[1e-4].true_accept_rate


def test_a_threshold_is_recommended_for_one_wrong_name_in_a_thousand(
    rng: np.random.Generator,
) -> None:
    recommended = evaluate(population(rng, people=40)).recommended()

    assert recommended is not None
    assert recommended.false_accept_rate == pytest.approx(1e-3)
    assert -1.0 <= recommended.threshold <= 1.0


def test_the_recommended_threshold_holds_wrong_identifications_to_its_promise(
    rng: np.random.Generator,
) -> None:
    """The number printed for the user to paste into .env, checked against the
    data it was derived from."""
    report = evaluate(population(rng, people=60))
    recommended = report.recommended()
    assert recommended is not None

    observed_false_accepts = float(np.mean(report.impostor_scores >= recommended.threshold))

    assert observed_false_accepts <= 1e-3 + 1e-9


def test_well_separated_identities_have_a_low_equal_error_rate(
    rng: np.random.Generator,
) -> None:
    assert evaluate(population(rng, people=30)).equal_error_rate < 0.05


def test_indistinguishable_identities_are_reported_as_such(
    rng: np.random.Generator,
) -> None:
    """A gallery built from photographs that all look alike is not a calibration
    problem, and no threshold will fix it. The report has to say so."""
    base = random_embedding(rng)
    indistinguishable = [
        LabelledEmbedding(f"person_{index}", nearby_embedding(base, rng, 0.99))
        for index in range(10)
        for _ in range(3)
    ]

    report = evaluate(indistinguishable)

    assert report.separation < 0.05
    assert report.equal_error_rate > 0.2


def test_a_single_photograph_yields_no_measurement(rng: np.random.Generator) -> None:
    """One photo per person gives no same-person pair, so half the measurement is
    missing and the command must say so rather than print a number."""
    single = [LabelledEmbedding(f"person_{index}", random_embedding(rng)) for index in range(5)]

    report = evaluate(single)

    assert report.genuine_pairs == 0
    assert report.recommended() is None


def test_an_empty_population_does_not_crash() -> None:
    report = evaluate([])

    assert report.genuine_pairs == 0
    assert report.separation == 0.0
    assert report.equal_error_rate == 1.0


def test_the_distribution_is_rendered_as_a_readable_histogram(
    rng: np.random.Generator,
) -> None:
    """The shape is what tells you whether a threshold is meaningful at all: two
    separated humps mean any sensible cut works, one broad hump means none does."""
    report = evaluate(population(rng))

    lines = format_distribution(report.genuine_scores)

    assert len(lines) == 20
    assert any("#" in line for line in lines)


def test_an_empty_distribution_renders_nothing() -> None:
    assert format_distribution(np.array([])) == []


@pytest.mark.parametrize("false_accept_rate", [1e-2, 1e-3, 1e-4])
def test_every_operating_point_keeps_the_rate_it_names(
    rng: np.random.Generator, false_accept_rate: float
) -> None:
    """Each printed line is a promise about how often a stranger is wrongly named.
    Interpolating a quantile between two samples quietly breaks that promise, so
    the bound is asserted directly against the data it came from."""
    report = evaluate(population(rng, people=80))
    point = next(p for p in report.operating_points if p.false_accept_rate == false_accept_rate)

    observed = float(np.mean(report.impostor_scores >= point.threshold))

    assert observed <= false_accept_rate


def test_a_rate_too_strict_for_the_sample_admits_nobody(rng: np.random.Generator) -> None:
    """With only a few hundred impostor pairs, one in ten thousand cannot be
    observed. The honest answer is a threshold above every impostor seen, not a
    number implying a precision the data does not support."""
    report = evaluate(population(rng, people=6, photos=3))
    strictest = min(report.operating_points, key=lambda point: point.false_accept_rate)

    assert float(np.mean(report.impostor_scores >= strictest.threshold)) == 0.0


def test_the_equal_error_rate_scales_to_a_realistic_measurement(
    rng: np.random.Generator,
) -> None:
    """A real run produces millions of impostor pairs. Treating every observed
    score as a candidate threshold is quadratic and simply never returns, so this
    pins the size the implementation has to cope with."""
    import time

    report = evaluate(population(rng, people=120, photos=8))
    assert report.impostor_pairs > 400_000

    started = time.perf_counter()
    rate = report.equal_error_rate
    elapsed = time.perf_counter() - started

    assert 0.0 <= rate <= 1.0
    assert elapsed < 5.0


def test_cached_descriptors_round_trip(tmp_path: Path, rng: np.random.Generator) -> None:
    """Encoding the public dataset takes minutes; choosing a threshold is naturally
    iterative. The cache is what makes that loop bearable."""
    original = population(rng, people=4, photos=3)
    path = tmp_path / "cache.npz"

    save_embeddings(path, original)
    restored = load_embeddings(path)

    assert restored is not None
    assert [item.identity_id for item in restored] == [item.identity_id for item in original]
    assert np.allclose(restored[0].embedding, original[0].embedding)


def test_a_missing_cache_is_not_an_error(tmp_path: Path) -> None:
    assert load_embeddings(tmp_path / "absent.npz") is None


def test_a_corrupt_cache_is_discarded_rather_than_repaired(tmp_path: Path) -> None:
    """A cache written by a different model version would silently corrupt the one
    measurement the system's accuracy is read from."""
    path = tmp_path / "cache.npz"
    path.write_bytes(b"not an npz archive")

    assert load_embeddings(path) is None


def test_an_empty_population_can_be_cached(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"

    save_embeddings(path, [])

    assert load_embeddings(path) == []
