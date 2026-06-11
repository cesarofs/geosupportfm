"""Support-aware spatial utilities."""

from geosupportfm.support.raster_aggregation import aggregate_array, aggregate_raster_stack
from geosupportfm.support.spatial_support import (
    aggregate_by_group,
    assign_spatial_blocks,
    grouped_train_test_split,
)

__all__ = [
    "aggregate_array",
    "aggregate_raster_stack",
    "aggregate_by_group",
    "assign_spatial_blocks",
    "grouped_train_test_split",
]
