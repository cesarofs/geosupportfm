"""Block-level aggregation for geospatial foundation-model embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BlockEmbeddingAggregation:
    block_indices: np.ndarray
    centers: np.ndarray
    embeddings: np.ndarray
    counts: np.ndarray
    weight_sums: np.ndarray


def aggregate_embeddings_to_blocks(
    coords: np.ndarray,
    embeddings: np.ndarray,
    block_size: float | Sequence[float],
    origin: Sequence[float] | None = None,
    sample_weights: np.ndarray | None = None,
    unit_normalize: bool = False,
    min_count: int = 1,
) -> BlockEmbeddingAggregation:
    """Aggregate embedding vectors into aligned rectangular blocks.

    Weighted arithmetic means implement areal support integration. Set
    ``unit_normalize=True`` for directional embeddings such as AlphaEarth when
    downstream analysis expects points on the unit hypersphere.
    """
    xy = np.asarray(coords, dtype=float)
    z = np.asarray(embeddings, dtype=float)
    if xy.ndim != 2 or len(xy) == 0 or z.ndim != 2 or len(z) != len(xy):
        raise ValueError("coords and embeddings must be non-empty aligned 2D arrays")
    if np.any(~np.isfinite(xy)) or np.any(~np.isfinite(z)):
        raise ValueError("coords and embeddings must be finite")
    dimensions = xy.shape[1]
    size = np.asarray(block_size if np.ndim(block_size) else [block_size] * dimensions, dtype=float)
    if size.shape != (dimensions,) or np.any(size <= 0):
        raise ValueError("block_size must be positive and match coordinate dimensions")
    anchor = np.min(xy, axis=0) if origin is None else np.asarray(tuple(origin), dtype=float)
    if anchor.shape != (dimensions,):
        raise ValueError("origin must match coordinate dimensions")
    weights = np.ones(len(xy)) if sample_weights is None else np.asarray(sample_weights, dtype=float).reshape(-1)
    if len(weights) != len(xy) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weights must be finite, non-negative, and match coords")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    indices = np.floor((xy - anchor) / size).astype(int)
    unique, inverse = np.unique(indices, axis=0, return_inverse=True)
    rows: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    for block_id, block_index in enumerate(unique):
        mask = inverse == block_id
        count = int(mask.sum())
        total = float(np.sum(weights[mask]))
        if count < min_count or total <= 0:
            continue
        mean = np.average(z[mask], axis=0, weights=weights[mask])
        if unit_normalize:
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                mean = mean / norm
        rows.append((block_index, mean, count, total))
    if not rows:
        return BlockEmbeddingAggregation(
            np.empty((0, dimensions), dtype=int),
            np.empty((0, dimensions)),
            np.empty((0, z.shape[1])),
            np.empty(0, dtype=int),
            np.empty(0),
        )
    block_indices = np.stack([row[0] for row in rows])
    centers = anchor + (block_indices + 0.5) * size
    return BlockEmbeddingAggregation(
        block_indices=block_indices,
        centers=centers,
        embeddings=np.stack([row[1] for row in rows]),
        counts=np.asarray([row[2] for row in rows], dtype=int),
        weight_sums=np.asarray([row[3] for row in rows], dtype=float),
    )

