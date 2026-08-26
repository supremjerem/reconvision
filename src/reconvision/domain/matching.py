"""Comparing a face descriptor against the enrolled gallery.

At household scale the gallery holds a few thousand vectors at most, so matching
is an exact brute-force dot product against a single matrix. An approximate index
would be both slower and less accurate here; see ADR 0003.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from reconvision.domain.models import Embedding, GalleryEntry, MatchResult

#: Below this norm a vector carries no direction and cannot be normalised.
_NEGLIGIBLE_NORM = 1e-8


def normalise(embedding: Embedding) -> Embedding:
    """Scale an embedding to unit length so dot products become cosine similarities."""
    norm = float(np.linalg.norm(embedding))
    if norm < _NEGLIGIBLE_NORM:
        message = "Cannot normalise a zero-length embedding"
        raise ValueError(message)
    return np.asarray(embedding / norm, dtype=np.float32)


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    """Cosine similarity in [-1, 1]. Inputs need not be normalised."""
    return float(np.dot(normalise(left), normalise(right)))


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """The rule turning similarity scores into an accept or reject decision.

    Two conditions, not one. The threshold answers "is this close enough to
    someone we know"; the margin answers "is it clearly closer to this person
    than to the next one". Two household members scoring 0.44 and 0.43 clears any
    sensible threshold while being a coin flip, and the margin is what catches it.
    """

    match_threshold: float
    min_margin: float = 0.05

    def __post_init__(self) -> None:
        if not -1.0 <= self.match_threshold <= 1.0:
            message = (
                f"Threshold must be a cosine similarity in [-1, 1], got {self.match_threshold}"
            )
            raise ValueError(message)
        if self.min_margin < 0.0:
            message = f"Margin must be non-negative, got {self.min_margin}"
            raise ValueError(message)

    def decide(
        self,
        best_identity_id: str,
        best_similarity: float,
        runner_up_identity_id: str | None = None,
        runner_up_similarity: float = -1.0,
    ) -> MatchResult:
        """Accept the best candidate, or return an unknown result."""
        clears_threshold = best_similarity >= self.match_threshold
        # A single enrolled identity has no runner-up, so the margin cannot
        # discriminate and must not be allowed to veto the match.
        has_rival = runner_up_identity_id is not None
        clears_margin = not has_rival or (best_similarity - runner_up_similarity) >= self.min_margin

        return MatchResult(
            identity_id=best_identity_id if clears_threshold and clears_margin else None,
            similarity=best_similarity,
            runner_up_similarity=runner_up_similarity,
            runner_up_identity_id=runner_up_identity_id,
        )


class GalleryMatcher:
    """Matches embeddings against the enrolled gallery.

    An identity's score is the *best* of its enrolled embeddings rather than their
    mean. People are enrolled across varied lighting and angles, so averaging
    those vectors blurs them into a descriptor that resembles no actual capture;
    the nearest single enrolled shot is the meaningful comparison.
    """

    def __init__(self, entries: Iterable[GalleryEntry], policy: ThresholdPolicy) -> None:
        self._policy = policy
        entry_list = list(entries)
        self._identity_ids: list[str] = [entry.identity_id for entry in entry_list]
        self._known_identity_ids: tuple[str, ...] = tuple(dict.fromkeys(self._identity_ids))

        if entry_list:
            stacked = np.stack([normalise(entry.embedding) for entry in entry_list])
            self._matrix = np.asarray(stacked, dtype=np.float32)
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)

    @property
    def is_empty(self) -> bool:
        """True when nobody has been enrolled yet, so every face is unknown."""
        return not self._identity_ids

    @property
    def known_identity_ids(self) -> tuple[str, ...]:
        return self._known_identity_ids

    @property
    def entry_count(self) -> int:
        return len(self._identity_ids)

    def match(self, embedding: Embedding) -> MatchResult:
        """Compare one embedding against every enrolled entry."""
        if self.is_empty:
            return MatchResult(identity_id=None, similarity=-1.0, runner_up_similarity=-1.0)

        similarities = self._matrix @ normalise(embedding)
        ranked = self._rank_identities(similarities)

        best_identity_id, best_similarity = ranked[0]
        runner_up_identity_id, runner_up_similarity = ranked[1] if len(ranked) > 1 else (None, -1.0)

        return self._policy.decide(
            best_identity_id=best_identity_id,
            best_similarity=best_similarity,
            runner_up_identity_id=runner_up_identity_id,
            runner_up_similarity=runner_up_similarity,
        )

    def _rank_identities(self, similarities: Embedding) -> Sequence[tuple[str, float]]:
        """Reduce per-entry scores to one score per identity, best first."""
        best_per_identity: dict[str, float] = {}
        for identity_id, similarity in zip(self._identity_ids, similarities, strict=True):
            score = float(similarity)
            if score > best_per_identity.get(identity_id, -np.inf):
                best_per_identity[identity_id] = score

        return sorted(best_per_identity.items(), key=lambda item: item[1], reverse=True)
