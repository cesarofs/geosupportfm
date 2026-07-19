"""Support-aware spatial utilities."""

from geosupportfm.support.raster_aggregation import aggregate_array, aggregate_raster_stack
from geosupportfm.support.embedding_aggregation import BlockEmbeddingAggregation, aggregate_embeddings_to_blocks
from geosupportfm.support.regularization import (
    SpatialSupport,
    SupportRegularizationResult,
    regularize_lmc,
    support_covariance_matrix,
)
from geosupportfm.support.spatial_support import (
    aggregate_by_group,
    assign_spatial_blocks,
    grouped_train_test_split,
)

__all__ = [
    "aggregate_array",
    "aggregate_raster_stack",
    "BlockEmbeddingAggregation",
    "SpatialSupport",
    "SupportRegularizationResult",
    "aggregate_by_group",
    "aggregate_embeddings_to_blocks",
    "assign_spatial_blocks",
    "grouped_train_test_split",
    "regularize_lmc",
    "support_covariance_matrix",
]
