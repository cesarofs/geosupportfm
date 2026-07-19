# GeoSupportFM

> **Geostatistical Support-Aware Learning for Geospatial Foundation Models**
>
> *Foundation Models learn representations. Geostatistics learns space. GeoSupportFM studies their intersection.*

---

## Overview

GeoSupportFM is an open-source research initiative exploring the integration of **Geospatial Foundation Models**, **Geostatistics**, and **Scientific Artificial Intelligence**.

The project investigates how geospatial foundation models can be combined with rigorous spatial statistics to improve prediction, uncertainty quantification, and scientific inference under:

* Sparse ground-truth observations
* Multi-resolution datasets
* Heterogeneous spatial supports
* Environmental and agricultural applications
* Earth Observation data fusion

The central hypothesis behind GeoSupportFM is:

> Foundation Models provide powerful representations of the Earth, but geostatistics provides the mathematical framework required for robust spatial inference under uncertainty.

---

## Motivation

Recent geospatial foundation models have demonstrated remarkable capabilities in learning universal representations from satellite imagery and multimodal Earth Observation data.

However, many scientific and agricultural applications continue to face fundamental challenges:

* Limited field observations
* Change-of-support effects
* Spatial autocorrelation
* Scale incompatibilities
* Uncertainty quantification
* Transferability across regions

GeoSupportFM aims to bridge these worlds by developing methods that combine:

* Geospatial Foundation Models
* Variography
* Co-regionalization
* Kriging and Cokriging
* Spatial Uncertainty Analysis
* Scientific Machine Learning

---

## Research Questions

### 1. Support-Aware Learning

How should geospatial embeddings be aggregated when moving between different spatial supports?

Examples:

* Point observations → satellite pixels
* Satellite pixels → management zones
* Plot measurements → landscape analysis

---

### 2. Embedding Variography

Do foundation model embeddings exhibit meaningful spatial structures?

Research questions:

* What is the spatial range of embedding dimensions?
* Do embeddings preserve anisotropy?
* How stable are variograms across years and regions?
* Can variogram properties improve downstream predictions?

---

### 3. Geostatistical Feature Selection

Current approaches typically select embedding dimensions using:

* Correlation
* Random Forest importance
* Mutual information

GeoSupportFM investigates:

* Variogram-aware feature selection
* Cross-variogram analysis
* Spatially informed embedding ranking

---

### 4. Sparse Ground Truth Learning

Many environmental variables are measured at only a handful of locations.

Examples:

* Crop yield
* Soil organic carbon
* Soil moisture
* Available water capacity
* Biomass
* Nutrient availability

GeoSupportFM investigates how foundation model embeddings can improve predictions under extreme data scarcity.

---

### 5. Kriging in Latent Space

A fundamental research question:

> Should we interpolate target variables directly, or interpolate the latent representation itself?

GeoSupportFM explores spatial interpolation directly within embedding spaces.

---

### 6. Uncertainty-Aware Earth Intelligence

Scientific applications require more than predictions.

They require:

* Confidence intervals
* Prediction intervals
* Spatial uncertainty maps
* Risk assessment

GeoSupportFM develops uncertainty-aware workflows combining foundation models and geostatistical inference.

---

## Initial Applications

### Coffee Yield Mapping

Combining:

* Field observations
* Sentinel-2 NDVI
* Satellite Embedding V1

using:

* Variography
* Linear Models of Coregionalization
* Block Cokriging

---

### Soil Organic Carbon

Support-aware prediction of SOC using sparse field samples and Earth Observation embeddings.

---

### Water Availability

Integration with the **agriwater** ecosystem for spatial estimation of water-related variables.

---

### Tropical Agriculture

Development of methods specifically designed for agricultural systems in tropical environments.

---

## Project Structure

```text
geosupportfm/
│
├── docs/
├── notebooks/
├── examples/
│
├── src/
│   └── geosupportfm/
│       ├── support/
│       ├── variograms/
│       ├── coregionalization/
│       ├── kriging/
│       ├── embeddings/
│       ├── uncertainty/
│       └── visualization/
│
├── tests/
├── papers/
├── datasets/
├── benchmarks/
│
├── README.md
├── LICENSE
└── pyproject.toml
```

---

## Roadmap

### Phase 1 — Foundations

* [x] Repository architecture
* [x] AlphaEarth data loaders
* [x] Sentinel-2 data loaders
* [x] Landsat data loaders
* [x] Spatial support transformation utilities
* [x] Raster aggregation tools

#### Phase 1 quick start

The Phase 1 API loads harmonized Earth Observation products from Earth Engine
and provides NumPy utilities for transforming raster support:

```python
import ee

from geosupportfm.foundations import (
    load_alphaearth_embeddings,
    load_landsat_sr,
    load_sentinel2_sr,
)
from geosupportfm.support import aggregate_array

ee.Initialize(project="your-google-cloud-project")
region = ee.Geometry.Rectangle([-47.1, -22.9, -46.9, -22.7])

embeddings = load_alphaearth_embeddings(2024, region=region)
sentinel2 = load_sentinel2_sr("2024-01-01", "2025-01-01", region=region)
landsat = load_landsat_sr(
    "2024-01-01",
    "2025-01-01",
    region=region,
    sensors=("landsat8", "landsat9"),
)

# Aggregate a local 10 m raster band to 30 m mean support.
band_30m = aggregate_array(band_10m, factor=3, reducer="mean")
```

Use `sample_alphaearth_embeddings` to extract latent vectors at an Earth
Engine `FeatureCollection`. `aggregate_raster_stack`, `aggregate_by_group`, and
the spatial block utilities support local multi-band and tabular workflows.

---

### Phase 2 — Reproducible Science

* [x] Coffee yield case study
* [x] Reproducible notebooks
* [x] Benchmark datasets
* [x] Open evaluation framework

#### Phase 2 quick start

The Phase 2 benchmark generates a reproducible spatial dataset, creates a
grouped spatial split, and compares Random Forest, ordinary kriging, and a
hybrid Random Forest plus kriged-residual model:

```python
from pathlib import Path

from geosupportfm.benchmarks import (
    SyntheticSpatialCaseStudyConfig,
    run_case_study,
)

config = SyntheticSpatialCaseStudyConfig(
    output_dir=Path("outputs/synthetic_case_study"),
    random_state=42,
    n_points=600,
    n_blocks=8,
    test_size=0.25,
)
result = run_case_study(config)

print(result["comparison"])
print(result["saved_paths"])
```

The workflow writes model comparisons, predictions, feature importance, and
diagnostic figures to the configured output directory. Run
`python -m geosupportfm.benchmarks.synthetic_spatial_case_study` for the
default experiment.

---

### Phase 3 — Embedding Variography

* [x] Variogram analysis of latent dimensions
* [x] Cross-variogram exploration
* [x] Spatial structure diagnostics

#### Phase 3 quick start

The Phase 3 API measures spatial continuity across latent dimensions, ranks
their spatial structure, and relates embeddings to an observed target through
cross-semivariograms:

```python
from geosupportfm.variograms import (
    empirical_cross_variogram,
    spatial_structure_diagnostics,
)

diagnostics, report = spatial_structure_diagnostics(
    coords=sample_coords,
    embeddings=embedding_matrix,
    band_names=embedding_names,
    n_lags=12,
    pair_min=5,
)

top_dimension = int(report.iloc[0]["dimension"])
cross_variogram = empirical_cross_variogram(
    sample_coords,
    embedding_matrix[:, top_dimension],
    target_values,
    n_lags=12,
    pair_min=5,
)

print(diagnostics)
print(report.head(10))
```

`latent_variography_report` returns the per-dimension ranking directly, while
`empirical_variogram` and `variogram_summary` expose the underlying univariate
analysis for custom research workflows.

---

### Phase 4 — Support-Aware Foundation Models

* [x] Change-of-support correction
* [x] Spatial support regularization
* [x] Block-level embedding aggregation
* [x] Permissible Linear Models of Coregionalization
* [x] Multivariate factorial point/block cokriging

#### Phase 4 quick start

The Phase 4 API represents each spatial scale with a positive-semidefinite
coregionalization matrix, integrates covariance over discretized supports, and
uses component-specific cross-covariances to filter multivariate fields:

```python
from geosupportfm.coregionalization import LMCStructure, LinearModelOfCoregionalization
from geosupportfm.kriging import factorial_cokriging
from geosupportfm.support import SpatialSupport

lmc = LinearModelOfCoregionalization((short_structure, long_structure))
pixel = SpatialSupport.rectangle((10.0, 10.0), discretization=5)
regional = factorial_cokriging(
    sample_coords,
    sample_values,
    target_coords,
    lmc,
    component=1,
    target_support=pixel,
)
```

Use `empirical_variogram_matrix` and `fit_lmc` to calculate an LMC from direct
and cross-semivariograms. `regularize_lmc` reports point-to-block variance
correction factors and the regularized block variogram. See
`examples/phase4_factorial_cokriging.py` for a complete synthetic workflow.

---

### Phase 5 — Latent Space Geostatistics

* [ ] Kriging in embedding spaces
* [ ] Spatial interpolation of latent representations
* [ ] Spatially aware foundation model adaptation

---

## Planned Publications

### Paper 1

**Support-Aware Geostatistical Fusion of Satellite Embeddings and Agricultural Field Observations**

---

### Paper 2

**Spatial Structure of Geospatial Foundation Model Embeddings**

---

### Paper 3

**Variogram-Aware Feature Selection for Geospatial Foundation Models**

---

### Paper 4

**Kriging in Geospatial Foundation Model Latent Spaces**

---

### Paper 5

**Scientific Inference under Sparse Ground Truth using Geospatial Foundation Models**

---

## Scientific Principles

GeoSupportFM follows five core principles:

1. Reproducibility First
2. Open Science
3. Explainability
4. Uncertainty Quantification
5. Scientific Utility

---

## Why GeoSupportFM?

The geospatial AI community has made enormous progress in representation learning.

The geostatistical community has developed decades of theory for spatial inference.

GeoSupportFM seeks to connect these two traditions.

Rather than treating foundation model embeddings as black-box features, GeoSupportFM investigates:

* Their spatial behavior
* Their support dependence
* Their uncertainty characteristics
* Their role in rigorous scientific inference

---

## Installation

```bash
git clone https://github.com/cesarofs/geosupportfm.git

cd geosupportfm

pip install -e .
```

---

## Citation

```bibtex
@software{silva2026geosupportfm,
  title={GeoSupportFM: Geostatistical Support-Aware Learning for Geospatial Foundation Models},
  author={Silva, Cesar de Oliveira Ferreira},
  year={2026},
  url={https://github.com/cesarofs/geosupportfm}
}
```

---

## Author

**Cesar de Oliveira Ferreira Silva, PhD**

Agricultural Engineer | GeoAI Researcher | Geostatistics | Spatial Modeling | Earth Observation | Scientific Machine Learning

* University of Campinas (UNICAMP)
* Embrapa Digital Agriculture

Website: https://cesarofs.github.io

---

## Acknowledgements

GeoSupportFM is inspired by advances in:

* Geostatistics
* Precision Agriculture
* Earth Observation
* Scientific Machine Learning
* Geospatial Foundation Models

and aims to contribute open, reproducible methods at their intersection.

---

### Project Status

🚧 Early Research Stage

The repository currently serves as an open research laboratory for developing the next generation of support-aware geospatial AI methods.
