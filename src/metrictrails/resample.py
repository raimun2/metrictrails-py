"""Spatial resolution and resampling (port of R/resample_stream.R)."""

import math

import numpy as np
import pandas as pd

from . import _rstats as rs
from ._utils import as_frame, datetime_to_seconds, seconds_to_datetime, check_columns, is_first_of_segment, segment_id, to_lonlat, to_metric, utm_epsg


def resolution_stream(data, q=0.9, idle=0.1):
    """Distribution of the distance between consecutive readings and the Nyquist resolution.

    `data` can be a DataFrame or a list (or dict) of DataFrames; in that case
    one row per stream is returned, with a `stream` column first.
    """
    if isinstance(data, (list, tuple, dict)):
        items = data.items() if isinstance(data, dict) else enumerate(data, start=1)
        rows = []
        for key, d in items:
            row = resolution_stream(d, q=q, idle=idle)
            row.insert(0, "stream", key)
            rows.append(row)
        return pd.concat(rows, ignore_index=True)

    check_columns(data, ["lon", "lat"], "resolution_stream")
    epsg = utm_epsg(data["lon"], data["lat"])
    xy = to_metric(data["lon"], data["lat"], epsg)
    seg = segment_id(data)
    is_first = is_first_of_segment(seg)
    delta_d = np.sqrt(np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2)
    delta_d = delta_d[~is_first[1:]]
    with np.errstate(invalid="ignore"):
        moving = delta_d[(delta_d >= idle) & ~np.isnan(delta_d)]
    if moving.size == 0:
        raise ValueError("The stream does not move.")
    qs = rs.quantile7(moving, [0.05, 0.25, 0.5, 0.75, 0.95])
    valid = delta_d[~np.isnan(delta_d)]
    return pd.DataFrame({
        "n": [len(data)],
        "pct_idle": [100 * rs.r_mean(valid < idle)],
        "mean": [rs.r_mean(moving)],
        "sd": [rs.r_sd(moving)],
        "q05": [qs[0]], "q25": [qs[1]], "q50": [qs[2]], "q75": [qs[3]], "q95": [qs[4]],
        "max": [moving.max()],
        "q": [float(q)],
        "res": [rs.quantile7(moving, q)],
    })


def _point_segment_distance(px, py, ax, ay, bx, by):
    """GEOS Distance::pointToSegment."""
    if ax == bx and ay == by:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    len2 = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
    r = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / len2
    if r <= 0.0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    if r >= 1.0:
        return math.sqrt((px - bx) ** 2 + (py - by) ** 2)
    s = ((ay - py) * (bx - ax) - (ax - px) * (by - ay)) / len2
    return abs(s) * math.sqrt(len2)


def douglas_peucker(x, y, tolerance):
    """Indices (0-based) kept by GEOS DouglasPeuckerLineSimplifier."""
    n = len(x)
    use = np.ones(n, dtype=bool)
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if i + 1 >= j:
            continue
        ax, ay, bx, by = x[i], y[i], x[j], y[j]
        # vectorised distance to the segment, same formulas as GEOS
        px = x[i + 1:j]
        py = y[i + 1:j]
        if ax == bx and ay == by:
            d = np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        else:
            len2 = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
            r = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / len2
            s = ((ay - py) * (bx - ax) - (ax - px) * (by - ay)) / len2
            d = np.abs(s) * math.sqrt(len2)
            d = np.where(r <= 0.0, np.sqrt((px - ax) ** 2 + (py - ay) ** 2), d)
            d = np.where(r >= 1.0, np.sqrt((px - bx) ** 2 + (py - by) ** 2), d)
        k = int(np.argmax(d))
        if d[k] <= tolerance:
            use[i + 1:j] = False
        else:
            m = i + 1 + k
            stack.append((m, j))
            stack.append((i, m))
    return np.where(use)[0]


def resample_stream(data, res="nyquist", q=0.9, method="arclength", idle=0.1):
    """Resample each segment of a stream to a uniform spatial resolution.

    The resolution used is stored in `result.attrs["res"]`.
    """
    if method not in ("arclength", "simplify"):
        raise ValueError("'method' must be 'arclength' or 'simplify'")
    check_columns(data, ["lon", "lat"], "resample_stream")
    data = as_frame(data)
    if isinstance(res, str) and res == "nyquist":
        res = float(resolution_stream(data, q=q, idle=idle)["res"].iloc[0])
    if isinstance(res, bool) or not isinstance(res, (int, float, np.integer, np.floating)) or res <= 0:
        raise ValueError("'res' must be a positive number or \"nyquist\".")
    res = float(res)

    epsg = utm_epsg(data["lon"], data["lat"])
    xy = to_metric(data["lon"], data["lat"], epsg)
    seg = segment_id(data)

    def is_numeric(col):
        s = data[col]
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

    numeric_cols = [c for c in data.columns if is_numeric(c) and c not in ("lon", "lat", "segment")]
    time_cols = [c for c in data.columns if pd.api.types.is_datetime64_any_dtype(data[c])]
    other_cols = [c for c in data.columns if c not in ("lon", "lat") and c not in numeric_cols
                  and c not in time_cols]

    pieces = []
    for s in pd.unique(seg):
        rows = np.where(seg == s)[0]
        x = xy[rows, 0]
        y = xy[rows, 1]
        s_orig = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))])
        total = s_orig[-1]
        if rows.size < 2 or total < res:
            continue
        if method == "arclength":
            n_step = math.floor(total / res)
            s_new = rs.seq_length(0.0, total, n_step + 1)
        else:
            s_dense = rs.seq_length(0.0, total, math.ceil(total / 0.2) + 1)
            x_dense = rs.approx(s_orig, x, s_dense, ties="mean")
            y_dense = rs.approx(s_orig, y, s_dense, ties="mean")
            kept = douglas_peucker(x_dense, y_dense, res)
            # as in R: find the kept vertices in order by their coordinates
            idx = np.empty(kept.size, dtype=int)
            pointer = 0
            for k, kk in enumerate(kept):
                while pointer < x_dense.size - 1 and (x_dense[pointer] != x_dense[kk]
                                                      or y_dense[pointer] != y_dense[kk]):
                    pointer += 1
                idx[k] = pointer
            s_new = s_dense[idx]

        piece = {"x": rs.approx(s_orig, x, s_new, ties="mean"),
                 "y": rs.approx(s_orig, y, s_new, ties="mean")}
        for col in numeric_cols:
            v = np.asarray(data[col].iloc[rows], dtype=float)
            ok = ~np.isnan(v)
            if ok.sum() >= 2:
                piece[col] = rs.approx(s_orig[ok], v[ok], s_new, ties="mean", rule=2)
            else:
                piece[col] = np.full(s_new.size, np.nan)
        for col in time_cols:
            v = datetime_to_seconds(data[col].iloc[rows])
            ok = ~np.isnan(v)
            if ok.sum() >= 2:
                value = rs.approx(s_orig[ok], v[ok], s_new, ties="mean", rule=2)
            else:
                value = np.full(s_new.size, np.nan)
            piece[col] = seconds_to_datetime(value).to_numpy()
        before = np.maximum(rs.find_interval(s_new, s_orig, rightmost_closed=True), 1) - 1
        for col in other_cols:
            piece[col] = data[col].iloc[rows].to_numpy()[before]
        if "segment" in data.columns:
            piece["segment"] = np.repeat(data["segment"].iloc[rows[0]], s_new.size)
        pieces.append(pd.DataFrame(piece))
    if not pieces:
        raise ValueError("No segment is longer than 'res'.")
    out = pd.concat(pieces, ignore_index=True)

    ll = to_lonlat(out["x"], out["y"], epsg)
    rest = [c for c in out.columns if c not in ("x", "y")]
    out = pd.concat([pd.DataFrame({"lon": ll[:, 0], "lat": ll[:, 1]}), out[rest]], axis=1)
    first = [c for c in ("lon", "lat", "ele", "timestamp", "time") if c in out.columns]
    out = out[first + [c for c in out.columns if c not in first]]
    out.attrs["res"] = res
    return out
