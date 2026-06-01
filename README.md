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

* [ ] Repository architecture
* [ ] AlphaEarth data loaders
* [ ] Sentinel-2 data loaders
* [ ] Landsat data loaders
* [ ] Spatial support transformation utilities
* [ ] Raster aggregation tools

---

### Phase 2 — Reproducible Science

* [ ] Coffee yield case study
* [ ] Reproducible notebooks
* [ ] Benchmark datasets
* [ ] Open evaluation framework

---

### Phase 3 — Embedding Variography

* [ ] Variogram analysis of latent dimensions
* [ ] Cross-variogram exploration
* [ ] Spatial structure diagnostics

---

### Phase 4 — Support-Aware Foundation Models

* [ ] Change-of-support correction
* [ ] Spatial support regularization
* [ ] Block-level embedding aggregation

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
