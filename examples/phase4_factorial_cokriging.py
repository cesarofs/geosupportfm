"""End-to-end Phase 4 example: LMC, block regularization, and scale filtering."""
from __future__ import annotations

import numpy as np

from geosupportfm.coregionalization import LMCStructure, LinearModelOfCoregionalization
from geosupportfm.kriging import factorial_cokriging
from geosupportfm.support import SpatialSupport, regularize_lmc


def main() -> None:
    rng = np.random.default_rng(42)
    observations = rng.uniform(0.0, 100.0, size=(40, 2))
    measurements = rng.normal(0.0, 5.0, size=(40, 2))
    lmc = LinearModelOfCoregionalization(
        (
            LMCStructure("exponential", 10.0, np.array([[2.0, 0.5], [0.5, 1.5]])),
            LMCStructure("exponential", 50.0, np.array([[4.0, 2.5], [2.5, 3.0]])),
        )
    )
    grid_x, grid_y = np.mgrid[0:100:10j, 0:100:10j]
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    pixel = SpatialSupport.rectangle((10.0, 10.0), discretization=5)
    regularization = regularize_lmc(lmc, pixel, np.linspace(0.0, 100.0, 21))
    short = factorial_cokriging(observations, measurements, targets, lmc, 0, target_support=pixel)
    long = factorial_cokriging(observations, measurements, targets, lmc, 1, target_support=pixel)
    print("Block/point variance ratios:", regularization.variance_ratio)
    print("Short-range block components:", short.estimates.shape)
    print("Long-range block components:", long.estimates.shape)


if __name__ == "__main__":
    main()

