from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep

@dataclass(frozen=True)
class GridSpec:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    width: int
    height: int
    crs: str = "EPSG:4326"

    @property
    def dx(self) -> float:
        return (self.lon_max - self.lon_min) / (self.width - 1)

    @property
    def dy(self) -> float:
        return (self.lat_max - self.lat_min) / (self.height - 1)

    def lonlat_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        lons = self.lons
        lats = self.lats
        lon2d, lat2d = np.meshgrid(lons, lats)
        return lon2d, lat2d

    @property
    def lons(self) -> np.ndarray:
        return np.linspace(self.lon_min, self.lon_max, self.width, dtype=np.float32)

    @property
    def lats(self) -> np.ndarray:
        return np.linspace(self.lat_max, self.lat_min, self.height, dtype=np.float32)


def _merged_geom_from_geojson(aoi_geojson: dict):
    feats = aoi_geojson.get("features") or []
    geoms = []
    for feat in feats:
        geom = feat.get("geometry") if isinstance(feat, dict) else None
        if geom:
            try:
                geoms.append(shape(geom))
            except Exception:
                pass
    if not geoms:
        raise ValueError("AOI GeoJSON has no valid geometries")
    return unary_union(geoms)


def _pad_degenerate_bounds(minx: float, miny: float, maxx: float, maxy: float) -> Tuple[float,float,float,float]:
    # Protect the UI and raster export against point/line-like AOIs or bad inputs.
    span_x = maxx - minx
    span_y = maxy - miny
    eps_x = 0.25 if span_x == 0 else max(span_x * 0.005, 1e-4)
    eps_y = 0.25 if span_y == 0 else max(span_y * 0.005, 1e-4)
    if span_x == 0:
        minx -= eps_x
        maxx += eps_x
    if span_y == 0:
        miny -= eps_y
        maxy += eps_y
    return float(minx), float(miny), float(maxx), float(maxy)

def bbox_from_geojson(aoi_geojson: dict) -> Tuple[float,float,float,float]:
    geom = _merged_geom_from_geojson(aoi_geojson)
    minx, miny, maxx, maxy = geom.bounds
    return _pad_degenerate_bounds(minx, miny, maxx, maxy)

def mask_from_geojson(aoi_geojson: dict, grid: GridSpec) -> np.ndarray:
    """
    Returns uint8 mask (1 inside AOI, 0 outside) with shape (H, W), aligned with grid lon/lat mesh.
    """
    geom = _merged_geom_from_geojson(aoi_geojson)
    pg = prep(geom)

    lon2d, lat2d = grid.lonlat_mesh()
    H, W = lon2d.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    # vectorized point-in-polygon is nontrivial; do a fast loop (H*W is small in demo)
    for i in range(H):
        for j in range(W):
            pt = Point(float(lon2d[i, j]), float(lat2d[i, j]))
            if pg.covers(pt):
                mask[i, j] = 1
    return mask
