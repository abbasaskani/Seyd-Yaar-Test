from __future__ import annotations

from typing import Dict, Any, Tuple
import math
import numpy as np
from scipy.ndimage import map_coordinates


def uv_to_speed_dir(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    spd = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2).astype(np.float32)
    # Heading toward which the vector points, 0=N, 90=E
    direc = (np.degrees(np.arctan2(u.astype(np.float32), v.astype(np.float32))) + 360.0) % 360.0
    return spd, direc.astype(np.float32)


def wind_proxy_from_waves(
    waves_hs_m: np.ndarray,
    wave_period_s: np.ndarray | None,
    wave_dir_deg: np.ndarray | None,
    stokes_u_m_s: np.ndarray | None = None,
    stokes_v_m_s: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Operational wind proxy when true atmospheric wind is not available.

    This is intentionally labelled a proxy. Direction follows the available wave direction
    field (best operational cue available in the current stack). Speed is a bounded index
    derived from sea-state intensity and period, slightly adjusted by Stokes magnitude.
    """
    hs = np.asarray(waves_hs_m, dtype=np.float32)
    if wave_period_s is None:
        Tp = np.full_like(hs, 7.0, dtype=np.float32)
    else:
        Tp = np.maximum(np.asarray(wave_period_s, dtype=np.float32), 1.0)
    if wave_dir_deg is None:
        wdir = np.zeros_like(hs, dtype=np.float32)
    else:
        wdir = np.asarray(wave_dir_deg, dtype=np.float32) % 360.0

    st_mag = np.zeros_like(hs, dtype=np.float32)
    if stokes_u_m_s is not None and stokes_v_m_s is not None:
        st_mag = np.sqrt(np.asarray(stokes_u_m_s, dtype=np.float64) ** 2 + np.asarray(stokes_v_m_s, dtype=np.float64) ** 2).astype(np.float32)

    # Heuristic wind proxy magnitude for operational ranking only.
    wspd = 2.0 + 2.6 * np.clip(hs, 0.0, 5.0) + 0.15 * np.clip(10.0 - Tp, 0.0, 8.0) + 8.0 * np.clip(st_mag, 0.0, 0.6)
    wspd = np.clip(wspd, 0.0, 22.0).astype(np.float32)
    # Convert heading to vector toward which the proxy points.
    u = (wspd * np.sin(np.deg2rad(wdir))).astype(np.float32)
    v = (wspd * np.cos(np.deg2rad(wdir))).astype(np.float32)
    return {
        "wind_u_m_s": u,
        "wind_v_m_s": v,
        "wind_speed_m_s": wspd,
        "wind_dir_deg": wdir,
    }


def _sample2(arr: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return map_coordinates(np.asarray(arr, dtype=np.float32), [y, x], order=1, mode="nearest").astype(np.float32)


def compute_setline_surfaces(
    prob01: np.ndarray,
    wind_dir_deg: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    mask_u8: np.ndarray | None = None,
    ops01: np.ndarray | None = None,
    total_length_nm: float = 10.0,
    high_prob_threshold: float = 0.70,
) -> Dict[str, np.ndarray]:
    """Compute local setline heading/score and a single best recommended setline.

    Operational assumption requested by the user:
    setline orientation is perpendicular to local wind direction.
    """
    p = np.asarray(prob01, dtype=np.float32)
    wdir = np.asarray(wind_dir_deg, dtype=np.float32) % 360.0
    H, W = p.shape
    if ops01 is None:
        ops = np.ones_like(p, dtype=np.float32)
    else:
        ops = np.asarray(ops01, dtype=np.float32)
    if mask_u8 is None:
        mask = np.ones_like(p, dtype=np.uint8)
    else:
        mask = np.asarray(mask_u8, dtype=np.uint8)

    set_heading = ((wdir + 90.0) % 180.0).astype(np.float32)
    score = np.full_like(p, np.nan, dtype=np.float32)
    cover = np.full_like(p, np.nan, dtype=np.float32)

    lat_line = lat2d[:, 0].astype(np.float32)
    lon_line = lon2d[0, :].astype(np.float32)
    dlat = float(np.nanmean(np.abs(np.diff(lat_line)))) if H > 1 else 0.1
    dlon = float(np.nanmean(np.abs(np.diff(lon_line)))) if W > 1 else 0.1
    half_m = float(total_length_nm) * 1852.0 / 2.0
    n_samp = max(17, int(math.ceil(float(total_length_nm) * 2.0)) * 2 + 1)

    yy0, xx0 = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    flat_y = yy0.ravel()
    flat_x = xx0.ravel()
    cand = np.where((mask.ravel() > 0) & np.isfinite(p.ravel()))[0]
    if cand.size == 0:
        z = np.zeros_like(p, dtype=np.float32)
        return {
            "net_set_heading_deg": set_heading,
            "net_set_score": z,
            "net_set_cover": z,
            "best": {
                "available": False,
                "reason": "no valid masked cells",
                "total_length_nm": float(total_length_nm),
            },
        }

    best_idx = None
    best_val = -1.0
    best_meta: Dict[str, Any] = {
        "available": False,
        "reason": "no candidate exceeded threshold",
        "total_length_nm": float(total_length_nm),
    }

    for k in cand.tolist():
        y0 = float(flat_y[k])
        x0 = float(flat_x[k])
        lat0 = float(lat2d[int(round(y0)), int(round(x0))])
        head = float(set_heading[int(round(y0)), int(round(x0))])
        along_u = math.sin(math.radians(head))
        along_v = math.cos(math.radians(head))
        m_per_deg_lat = 111320.0
        m_per_deg_lon = max(111320.0 * math.cos(math.radians(lat0)), 1e-3)
        dlat_half = (half_m * along_v) / m_per_deg_lat
        dlon_half = (half_m * along_u) / m_per_deg_lon
        # endpoints in geographic coords
        lat_a = lat0 - dlat_half
        lon_a = float(lon2d[int(round(y0)), int(round(x0))]) - dlon_half
        lat_b = lat0 + dlat_half
        lon_b = float(lon2d[int(round(y0)), int(round(x0))]) + dlon_half
        ys = np.linspace(y0 - dlat_half / max(dlat, 1e-6), y0 + dlat_half / max(dlat, 1e-6), n_samp).astype(np.float32)
        xs = np.linspace(x0 - dlon_half / max(dlon, 1e-6), x0 + dlon_half / max(dlon, 1e-6), n_samp).astype(np.float32)
        pp = _sample2(p, ys, xs)
        oo = _sample2(ops, ys, xs)
        valid = np.isfinite(pp) & np.isfinite(oo)
        if not np.any(valid):
            continue
        pp = pp[valid]
        oo = oo[valid]
        mean_p = float(np.mean(pp))
        p90 = float(np.percentile(pp, 90))
        high_cov = float(np.mean(pp >= float(high_prob_threshold)))
        mean_ops = float(np.mean(oo))
        val = 0.50 * mean_p + 0.20 * p90 + 0.20 * high_cov + 0.10 * mean_ops
        r = int(round(y0)); c = int(round(x0))
        score[r, c] = val
        cover[r, c] = high_cov
        if val > best_val:
            best_val = val
            best_idx = (r, c)
            best_meta = {
                "available": True,
                "center_lat": lat0,
                "center_lon": float(lon2d[r, c]),
                "heading_deg": head,
                "perpendicular_to_wind_deg": float((wdir[r, c] + 90.0) % 180.0),
                "wind_dir_deg": float(wdir[r, c]),
                "line_start": {"lat": float(lat_a), "lon": float(lon_a)},
                "line_end": {"lat": float(lat_b), "lon": float(lon_b)},
                "total_length_nm": float(total_length_nm),
                "score": float(val),
                "mean_prob": mean_p,
                "p90_prob": p90,
                "high_prob_cover_fraction": high_cov,
                "mean_ops": mean_ops,
            }

    score = np.where(np.isfinite(score), score, np.nanmedian(score[np.isfinite(score)]) if np.any(np.isfinite(score)) else 0.0).astype(np.float32)
    cover = np.where(np.isfinite(cover), cover, 0.0).astype(np.float32)
    return {
        "net_set_heading_deg": set_heading.astype(np.float32),
        "net_set_score": np.clip(score, 0.0, 1.0).astype(np.float32),
        "net_set_cover": np.clip(cover, 0.0, 1.0).astype(np.float32),
        "best": best_meta,
    }
