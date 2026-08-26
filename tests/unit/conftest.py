"""Synthetic embeddings for the domain suite.

Real ArcFace vectors are not needed to test matching rules, and depending on them
would drag a 350 MB model into a suite that must stay instant. What matters is the
geometry: vectors that are near each other stand for the same person, vectors that
are far apart stand for different people.
"""

from __future__ import annotations

import numpy as np
import pytest

from reconvision.domain.matching import normalise
from reconvision.domain.models import Embedding

EMBEDDING_DIMENSIONS = 512


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded so a failure is reproducible rather than a once-a-week mystery."""
    return np.random.default_rng(seed=20260826)


def random_embedding(rng: np.random.Generator) -> Embedding:
    """A unit vector standing in for one person's face descriptor."""
    return normalise(rng.standard_normal(EMBEDDING_DIMENSIONS).astype(np.float32))


def nearby_embedding(
    base: Embedding, rng: np.random.Generator, similarity: float = 0.8
) -> Embedding:
    """Another capture of the same face, at a controlled cosine similarity.

    Built by mixing the base with an orthogonal direction, so the resulting
    similarity is the requested one rather than an approximation.
    """
    noise = random_embedding(rng)
    orthogonal = normalise(noise - float(np.dot(noise, base)) * base)
    mixed = similarity * base + np.sqrt(1.0 - similarity**2) * orthogonal
    return normalise(mixed.astype(np.float32))
