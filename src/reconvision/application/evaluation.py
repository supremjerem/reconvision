"""Choosing the match threshold from measurement rather than from a hunch.

The threshold is the single number that decides whether the system recognises you
or accuses you of being a stranger, and there is no value that is correct in the
abstract: it depends on the model, the cameras and the people enrolled. This
module measures the two distributions that matter and reports the operating point.

The vocabulary is the field's own:

- a *genuine* pair is two photographs of the same person,
- an *impostor* pair is two photographs of different people,
- **TAR@FAR** is the share of genuine pairs accepted, at a threshold chosen so
  that only a given share of impostor pairs is accepted.

FAR is the number to fix first. It is the rate at which a stranger is greeted by
your name, and in a house that is the error that matters; TAR is then simply how
often the system recognises you, given that constraint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from reconvision.domain.matching import normalise
from reconvision.domain.models import Embedding

#: False-accept rates to report. 1e-3 is the headline figure: one stranger in a
#: thousand comparisons wrongly named.
REPORTED_FALSE_ACCEPT_RATES = (1e-2, 1e-3, 1e-4)


@dataclass(frozen=True, slots=True)
class LabelledEmbedding:
    """One face descriptor and the person it belongs to."""

    identity_id: str
    embedding: Embedding


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """What a given threshold would do in practice."""

    false_accept_rate: float
    threshold: float
    true_accept_rate: float

    def describe(self) -> str:
        return (
            f"FAR {self.false_accept_rate:>7.1%}  ->  threshold {self.threshold:.3f}  "
            f"recognises {self.true_accept_rate:.1%} of genuine faces"
        )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Score distributions and the thresholds they imply."""

    genuine_scores: np.ndarray
    impostor_scores: np.ndarray
    operating_points: Sequence[OperatingPoint]
    identities: int

    @property
    def genuine_pairs(self) -> int:
        return int(self.genuine_scores.size)

    @property
    def impostor_pairs(self) -> int:
        return int(self.impostor_scores.size)

    @property
    def separation(self) -> float:
        """Gap between the average genuine and average impostor score.

        A quick sanity figure. If this is near zero the embeddings carry no usable
        identity signal and no threshold will rescue the system.
        """
        if not self.genuine_pairs or not self.impostor_pairs:
            return 0.0
        return float(self.genuine_scores.mean() - self.impostor_scores.mean())

    @property
    def equal_error_rate(self) -> float:
        """The rate where wrongly accepting and wrongly rejecting are equally common.

        A single number for comparing configurations. Not a good operating point
        for a home: it treats greeting a stranger by your name and failing to
        recognise you as equally bad, and they are not.
        """
        if not self.genuine_pairs or not self.impostor_pairs:
            return 1.0

        thresholds = np.unique(np.concatenate([self.genuine_scores, self.impostor_scores]))
        false_accepts = np.array([np.mean(self.impostor_scores >= t) for t in thresholds])
        false_rejects = np.array([np.mean(self.genuine_scores < t) for t in thresholds])
        crossing = int(np.argmin(np.abs(false_accepts - false_rejects)))
        return float((false_accepts[crossing] + false_rejects[crossing]) / 2)

    def recommended(self) -> OperatingPoint | None:
        """The operating point to put in `.env`: one wrong name per thousand."""
        for point in self.operating_points:
            if point.false_accept_rate == 1e-3:
                return point
        return self.operating_points[0] if self.operating_points else None


def score_pairs(
    labelled: Sequence[LabelledEmbedding],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute every genuine and impostor similarity in one pass.

    All pairs rather than a sample: at household scale the full matrix is a few
    million floats, and sampling would put noise into the one measurement the
    whole system's accuracy is read from.
    """
    if len(labelled) < 2:
        return np.array([]), np.array([])

    matrix = np.stack([normalise(item.embedding) for item in labelled])
    similarities = matrix @ matrix.T

    identities = np.array([item.identity_id for item in labelled])
    same_person = identities[:, None] == identities[None, :]
    # Only the upper triangle: a pair is one comparison, and a face compared with
    # itself scores 1.0 and would flatter every genuine distribution.
    upper = np.triu(np.ones_like(similarities, dtype=bool), k=1)

    return similarities[upper & same_person], similarities[upper & ~same_person]


def evaluate(labelled: Sequence[LabelledEmbedding]) -> EvaluationReport:
    """Measure how separable the enrolled identities are."""
    genuine, impostor = score_pairs(labelled)

    return EvaluationReport(
        genuine_scores=genuine,
        impostor_scores=impostor,
        operating_points=[
            point
            for rate in REPORTED_FALSE_ACCEPT_RATES
            if (point := _operating_point(genuine, impostor, rate)) is not None
        ],
        identities=len({item.identity_id for item in labelled}),
    )


def _operating_point(
    genuine: np.ndarray, impostor: np.ndarray, false_accept_rate: float
) -> OperatingPoint | None:
    """The threshold admitting at most this share of impostor pairs."""
    if genuine.size == 0 or impostor.size == 0:
        return None

    # Counted directly rather than taken from np.quantile, whose interpolation
    # returns a value between two samples and lets slightly more impostors through
    # than the rate names. The figure is printed to the user as a promise about how
    # often a stranger is wrongly named, so it has to hold exactly.
    ranked = np.sort(impostor)
    total = ranked.size
    admissible = int(np.floor(false_accept_rate * total))

    if admissible == 0:
        # No impostor may be accepted, so the threshold sits just above the
        # highest of them.
        threshold = float(np.nextafter(ranked[-1], np.inf))
    else:
        threshold = float(ranked[total - admissible])

    return OperatingPoint(
        false_accept_rate=false_accept_rate,
        threshold=threshold,
        true_accept_rate=float(np.mean(genuine >= threshold)),
    )


def format_distribution(scores: np.ndarray, buckets: int = 20) -> list[str]:
    """A text histogram, so `eval` shows the shape rather than only numbers.

    The shape is what tells you whether a threshold is even meaningful: two
    well-separated humps mean any reasonable cut works, one broad hump means no
    threshold will.
    """
    if scores.size == 0:
        return []

    counts, edges = np.histogram(scores, bins=buckets, range=(-0.2, 1.0))
    peak = max(1, int(counts.max()))
    return [
        f"{edges[index]:+.2f}  {'#' * round(40 * count / peak):<40} {count}"
        for index, count in enumerate(counts)
    ]
