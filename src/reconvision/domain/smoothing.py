"""Turning many per-frame guesses into one decision per tracked path.

Deciding identity frame by frame is both noisy and wasteful: a person crossing a
room yields hundreds of frames, and the answer flickers between them. Accumulating
the evidence along a track and deciding once produces a single event, and produces
a better answer than any individual frame, because the frames where the face was
large and sharp outweigh the ones where it was not.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from reconvision.domain.models import MatchResult, TrackVerdict

#: Key under which evidence for "matched nobody" accumulates. Not an identity, so
#: it cannot collide with one: identity ids are non-empty strings.
_UNKNOWN = ""


@dataclass(frozen=True, slots=True)
class VotePolicy:
    """How much agreement a track needs before its verdict is trusted."""

    #: A single frame is never enough. Two agreeing frames are a coincidence away
    #: from wrong; three is the point where a stray match stops deciding alone.
    min_observations: int = 3
    #: Share of the total weight the winner must hold. At 0.6 a track where the
    #: face was recognised in most good frames wins, while a genuinely split track
    #: is reported inconclusive instead of being resolved by a hair.
    min_weight_share: float = 0.6

    def __post_init__(self) -> None:
        if self.min_observations < 1:
            message = f"A track needs at least one observation, got {self.min_observations}"
            raise ValueError(message)
        if not 0.0 < self.min_weight_share <= 1.0:
            message = f"Weight share must be in (0, 1], got {self.min_weight_share}"
            raise ValueError(message)


@dataclass(slots=True)
class TrackVote:
    """Accumulates recognition evidence for one tracked subject.

    This is the only mutable object in the domain, and deliberately so: a track is
    an accumulator that exists for the lifetime of a person crossing the frame.
    """

    policy: VotePolicy = field(default_factory=VotePolicy)
    _weight_by_identity: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _observations: int = 0
    _best_similarity: float = -1.0

    def observe(self, match: MatchResult, weight: float) -> None:
        """Record one frame's outcome, weighted by the face's quality.

        A weight of zero still counts as an observation but sways nothing, which
        is what an unusable face should do.
        """
        if weight < 0.0:
            message = f"Observation weight must be non-negative, got {weight}"
            raise ValueError(message)

        key = match.identity_id if match.identity_id is not None else _UNKNOWN
        self._weight_by_identity[key] += weight
        self._observations += 1
        if match.is_match:
            self._best_similarity = max(self._best_similarity, match.similarity)

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def best_similarity(self) -> float:
        """Highest similarity seen on this track, for reporting on the event."""
        return self._best_similarity

    def verdict(self) -> TrackVerdict:
        """The current decision for this track.

        Safe to call at any time: a track that has not yet seen enough frames is
        reported as inconclusive rather than as an unknown person, so the pipeline
        can wait instead of raising a false alarm.
        """
        total_weight = sum(self._weight_by_identity.values())
        if self._observations < self.policy.min_observations or total_weight <= 0.0:
            return TrackVerdict(
                identity_id=None,
                confidence=0.0,
                observations=self._observations,
                is_conclusive=False,
            )

        winner, winning_weight = max(self._weight_by_identity.items(), key=lambda item: item[1])
        share = winning_weight / total_weight

        if share < self.policy.min_weight_share:
            return TrackVerdict(
                identity_id=None,
                confidence=share,
                observations=self._observations,
                is_conclusive=False,
            )

        return TrackVerdict(
            identity_id=winner if winner != _UNKNOWN else None,
            confidence=share,
            observations=self._observations,
            is_conclusive=True,
        )
