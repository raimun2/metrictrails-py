"""Fractal dimension by the divider method (port of R/fractal_stream.R)."""

import math

import numpy as np
import pandas as pd

from . import _rstats as rs
from ._utils import check_columns, segment_id, to_metric, utm_epsg


def fractal_stream(data, scales=None, n_scales=10):
    """Divider (Richardson) estimate of the fractal dimension of the 2D path.

    Returns a dict with `dimension`, `r_squared` and `scaling` (a DataFrame
    with the divider size `r`, the number of steps `n` and the measured
    `length` at every scale).
    """
    check_columns(data, ["lon", "lat"], "fractal_stream")
    lon = np.asarray(data["lon"], dtype=float)
    lat = np.asarray(data["lat"], dtype=float)
    ok = ~np.isnan(lon) & ~np.isnan(lat)
    data = data.loc[ok].reset_index(drop=True)
    epsg = utm_epsg(data["lon"], data["lat"])
    xy = to_metric(data["lon"], data["lat"], epsg)
    seg = segment_id(data)

    if scales is None:
        step = np.sqrt(np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2)
        step = step[seg[1:] == seg[:-1]]
        total = rs.r_sum(step)
        moving = step[step >= 0.1]
        if moving.size < 3:
            raise ValueError("The stream does not move.")
        r_min = 2 * rs.r_median(moving)
        r_max = total / 10
        if r_max <= r_min:
            raise ValueError("The path is too short for its resolution to estimate a fractal dimension.")
        scales = np.exp(rs.seq_length(math.log(r_min), math.log(r_max), n_scales))
    scales = np.asarray(scales, dtype=float)

    n_steps = np.zeros(scales.size)
    lengths = np.zeros(scales.size)
    for k, r in enumerate(scales):
        for s in pd.unique(seg):
            rows = np.where(seg == s)[0]
            if rows.size < 2:
                continue
            n, length = divider_walk(xy[rows, 0], xy[rows, 1], r)
            n_steps[k] += n
            lengths[k] += length

    lx = np.log(scales)
    ly = np.log(lengths)
    beta, _ = rs.wls(np.column_stack([np.ones(lx.size), lx]), ly)
    return {
        "dimension": 1 - beta[1],
        "r_squared": rs.r_cor(lx, ly) ** 2,
        "scaling": pd.DataFrame({"r": scales, "n": n_steps, "length": lengths}),
    }


def divider_walk(x, y, r):
    """Walk a divider of size r along the polyline; (steps, measured length)."""
    n_pts = len(x)
    cx = x[0]
    cy = y[0]
    i = 0
    u0 = 0.0
    n = 0
    r2 = r * r * (1 - 1e-9)
    while True:
        k = i
        while k < n_pts - 1 and (x[k + 1] - cx) ** 2 + (y[k + 1] - cy) ** 2 < r2:
            k += 1
        if k >= n_pts - 1:
            return n, n * r + math.sqrt((x[n_pts - 1] - cx) ** 2 + (y[n_pts - 1] - cy) ** 2)
        dx = x[k + 1] - x[k]
        dy = y[k + 1] - y[k]
        fx = x[k] - cx
        fy = y[k] - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        u = (-b + math.sqrt(max(b * b - 4 * a * c, 0))) / (2 * a)
        lower = u0 if k == i else 0.0
        u = min(max(u, lower), 1.0)
        cx = x[k] + u * dx
        cy = y[k] + u * dy
        i = k
        u0 = u
        n += 1
