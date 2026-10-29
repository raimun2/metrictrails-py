"""Internal helpers shared by several functions (port of R/utils.R)."""

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from pyproj import Transformer

from . import _rstats as rs

NAMING_HINT = ". See help(uniform_stream) for the naming convention."


def utm_epsg(lon, lat):
    """EPSG code of the WGS 84 / UTM zone containing the centre of the stream."""
    lon = rs.na_omit(lon)
    lat = rs.na_omit(lat)
    lon0 = (lon.min() + lon.max()) / 2
    lat0 = (lat.min() + lat.max()) / 2
    zone = math.floor((lon0 + 180) / 6) + 1
    zone = min(max(zone, 1), 60)
    return 32700 + zone if lat0 < 0 else 32600 + zone


@lru_cache(maxsize=64)
def _transformer(src, dst):
    return Transformer.from_crs(src, dst, always_xy=True)


def _project(a, b, src, dst):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full((a.size, 2), np.nan)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.any():
        x, y = _transformer(src, dst).transform(a[ok], b[ok])
        out[ok, 0] = x
        out[ok, 1] = y
    return out


def to_metric(lon, lat, epsg):
    """Project lon/lat (degrees) to EPSG:epsg; rows with NA stay NA."""
    return _project(lon, lat, "EPSG:4326", f"EPSG:{epsg}")


def to_lonlat(x, y, epsg):
    return _project(x, y, f"EPSG:{epsg}", "EPSG:4326")


def segment_id(data):
    """Integer segment id (1-based, in order of appearance) for every row."""
    if "segment" not in data.columns:
        return np.ones(len(data), dtype=int)
    codes, _ = pd.factorize(data["segment"], sort=False, use_na_sentinel=False)
    return codes.astype(int) + 1


def check_columns(data, columns, fun):
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"{fun}() needs column(s): {', '.join(missing)}{NAMING_HINT}")
    return True


def diff_by_segment(x, seg, first=0.0):
    """Differences inside each segment; the first row of every segment gets `first`."""
    x = np.asarray(x, dtype=float)
    d = np.empty(x.size)
    if x.size == 0:
        return d
    d[0] = first
    d[1:] = np.diff(x)
    starts = np.concatenate([[True], seg[1:] != seg[:-1]])
    d[starts] = first
    return d


def is_first_of_segment(seg):
    seg = np.asarray(seg)
    return np.concatenate([[True], seg[1:] != seg[:-1]])


def nearest_neighbour(query, target):
    """Nearest row of `target` for every row of `query` (metric coordinates).

    Returns 1-based indices into the complete rows of `target`, as in R, and
    the distances. In 2D the search reproduces sf::st_nearest_feature (GEOS
    STRtree), including how it breaks ties; in 3D it is a brute-force search
    that keeps the first of equidistant rows, as max.col(ties = "first").
    """
    query = np.asarray(query, dtype=float)
    target = np.asarray(target, dtype=float)
    target = target[~np.isnan(target).any(axis=1)]
    ok = ~np.isnan(query).any(axis=1)
    index = np.full(query.shape[0], -1, dtype=int)
    dist = np.full(query.shape[0], np.nan)
    q = query[ok]
    if q.shape[1] == 2:
        from ._strtree import PointIndex
        tree = PointIndex(target[:, 0], target[:, 1])
        found = np.array([tree.nearest(x, y) for x, y in q], dtype=int)
    else:
        found = np.empty(q.shape[0], dtype=int)
        block = max(1, int(1e6 // max(1, target.shape[0])))
        for start in range(0, q.shape[0], block):
            rows = slice(start, min(q.shape[0], start + block))
            d2 = np.zeros((q[rows].shape[0], target.shape[0]))
            for j in range(q.shape[1]):
                d2 += (q[rows, j][:, None] - target[None, :, j]) ** 2
            found[rows] = np.argmin(d2, axis=1)
    index[ok] = found + 1
    dist[ok] = np.sqrt(((q - target[found]) ** 2).sum(axis=1))
    return index, dist


def as_frame(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def ensure_float(v):
    return np.asarray(pd.to_numeric(pd.Series(v), errors="coerce"), dtype=float)


def datetime_to_seconds(values):
    """Seconds since the epoch (float, NaN for NaT) of a datetime-like column."""
    s = pd.Series(values)
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    out = ((s - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    return out


def seconds_to_datetime(secs):
    """UTC datetimes (nanosecond resolution) from seconds since the epoch."""
    secs = np.asarray(secs, dtype=float)
    ns = np.where(np.isnan(secs), np.nan, np.round(secs * 1e9))
    return pd.to_datetime(pd.Series(ns, dtype="float64"), unit="ns", utc=True).astype("datetime64[ns, UTC]")
