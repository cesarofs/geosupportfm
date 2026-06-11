"""Earth Engine sampling helpers for MapBiomas and Satellite Embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence

from .config import GeoSupportEmbeddingConfig


def initialize_earth_engine(project_id: str) -> Any:
    """Authenticate and initialize Earth Engine lazily."""
    import ee

    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    return ee


def brazil_geometry(ee_module: Any) -> Any:
    """Return the Brazil geometry from the LSIB dataset."""
    brazil = (
        ee_module.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee_module.Filter.eq("country_na", "Brazil"))
    )
    return brazil.geometry()


def load_lulc(ee_module: Any, year: int, config: GeoSupportEmbeddingConfig) -> Any:
    """Load MapBiomas LULC for one year and mask nodata."""
    image_collection = (
        ee_module.ImageCollection(config.lulc_asset)
        .filter(ee_module.Filter.eq("collection_id", 10.0))
        .filter(ee_module.Filter.eq("version", "v1"))
        .filter(ee_module.Filter.eq("year", year))
    )
    image = ee_module.Image(image_collection.first()).select([0], [config.lulc_column]).toInt16()
    return image.updateMask(image.neq(0))


def load_embeddings(ee_module: Any, year: int, config: GeoSupportEmbeddingConfig, region: Any) -> Any:
    """Load annual satellite embeddings and mosaic tiles over the target region."""
    return (
        ee_module.ImageCollection(config.emb_collection)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
        .mosaic()
        .select(list(config.emb_bands))
    )


def sample_year(
    ee_module: Any,
    year: int,
    config: GeoSupportEmbeddingConfig,
    region: Any,
) -> Any:
    """Build a class-balanced sample for one year."""
    stack = load_embeddings(ee_module, year, config, region).addBands(load_lulc(ee_module, year, config)).clip(region)
    samples = stack.stratifiedSample(
        numPoints=config.per_class,
        classBand=config.lulc_column,
        region=region,
        scale=config.sample_scale,
        seed=config.seed,
        geometries=False,
        tileScale=16,
        dropNulls=True,
    )
    return samples.map(lambda feature: feature.set({config.year_column: year}))


def build_samples(
    ee_module: Any,
    config: GeoSupportEmbeddingConfig,
    years: Sequence[int] | None = None,
) -> Any:
    """Merge yearly samples into one FeatureCollection."""
    region = brazil_geometry(ee_module)
    years_iter = list(config.years if years is None else years)
    samples = ee_module.FeatureCollection([])
    for year in years_iter:
        samples = samples.merge(sample_year(ee_module, year, config, region))
    return samples.select(list(config.emb_bands) + [config.lulc_column, config.year_column])


def export_yearly_samples_to_drive(
    ee_module: Any,
    config: GeoSupportEmbeddingConfig,
    years: Sequence[int] | None = None,
    folder: str = "gee_lulc_embedding",
) -> list[Any]:
    """Export yearly samples to Drive as CSV files."""
    region = brazil_geometry(ee_module)
    tasks: list[Any] = []
    for year in list(config.years if years is None else years):
        fc = sample_year(ee_module, year, config, region).select(list(config.emb_bands) + [config.lulc_column, config.year_column])
        task = ee_module.batch.Export.table.toDrive(
            collection=fc,
            description=f"mapbiomas_embedding_{year}",
            folder=folder,
            fileNamePrefix=f"sample_{year}",
            fileFormat="CSV",
        )
        task.start()
        tasks.append(task)
    return tasks


def read_local_sample_csvs(directory: str | Path) -> list[Any]:
    """Return sorted local CSV paths ready for pandas ingestion."""
    directory_path = Path(directory)
    return sorted(directory_path.glob("sample_*.csv"))
