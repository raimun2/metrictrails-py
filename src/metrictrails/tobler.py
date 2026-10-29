"""Tobler hiking function (port of R/tobler_stream.R)."""

import math
import warnings

import numpy as np
import pandas as pd

from . import _rstats as rs
from ._utils import check_columns

PARAM_NAMES = ("v1", "m1", "c_plus", "c_minus")


def _params(params):
    """Tobler parameters from a DataFrame row, a dict, a Series or a sequence."""
    if isinstance(params, pd.DataFrame):
        params = params.iloc[0]
    if isinstance(params, (dict, pd.Series)):
        try:
            p = [float(params[k]) for k in PARAM_NAMES]
            if not any(math.isnan(v) for v in p):
                return p
        except (KeyError, TypeError, ValueError):
            pass
        params = list(params.values() if isinstance(params, dict) else params.to_numpy())
    return [float(v) for v in list(params)[:4]]


def tobler_speed(slope, params):
    """Speed predicted by the Tobler function at `slope` (rise over run)."""
    p = _params(params)
    slope = np.asarray(slope, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(slope >= p[1], p[0] * np.exp(-p[2] * (slope - p[1])),
                        p[0] * np.exp(-p[3] * (slope - p[1])))


def tobler_stream(data, tau=0.9, method="pinball", slope_range=(-0.6, 0.6), min_speed=0.1,
                  bin_width=0.02, min_bin=10):
    """Fit the Tobler hiking function as the tau-quantile frontier of speed against slope."""
    from .measures import differential_stream

    if method not in ("pinball", "binned"):
        raise ValueError("'method' must be 'pinball' or 'binned'")
    check_columns(data, ["lon", "lat", "ele", "time"], "tobler_stream")
    d = differential_stream(data)
    speed = d["speed"].to_numpy(dtype=float)
    slope = d["slope"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        keep = (np.isfinite(speed) & np.isfinite(slope) & (speed > min_speed)
                & (slope >= slope_range[0]) & (slope <= slope_range[1]))
    v = speed[keep]
    m = slope[keep]
    if v.size < 30:
        raise ValueError("Fewer than 30 usable readings to fit the Tobler function.")

    breaks = rs.seq_by(math.floor(m.min() / bin_width) * bin_width,
                       math.ceil(m.max() / bin_width) * bin_width + bin_width, bin_width)
    code = rs.bincode(m, breaks)
    bm, bv, bn = [], [], []
    for b in range(1, len(breaks)):
        in_bin = np.where(code == b)[0]
        if in_bin.size >= min_bin:
            bm.append(rs.r_median(m[in_bin]))
            bv.append(rs.quantile7(v[in_bin], tau))
            bn.append(in_bin.size)
    if len(bm) < 4:
        raise ValueError("Not enough populated slope bins; lower 'min_bin' or widen 'bin_width'.")
    bm = np.array(bm)
    bv = np.array(bv)
    w = np.array(bn, dtype=float)
    logv = np.log(bv)

    def fit_at(m1):
        X = np.column_stack([np.ones(bm.size), np.maximum(bm - m1, 0), np.minimum(bm - m1, 0)])
        return rs.wls(X, logv, w)

    def rss(m1):
        return rs.r_sum(fit_at(m1)[1] ** 2)

    inner = rs.quantile7(bm, [0.1, 0.9])
    m1 = rs.optimize_brent(rss, inner[0], inner[1])
    coef = fit_at(m1)[0]
    theta = np.array([math.exp(coef[0]), m1, -coef[1], -coef[2]])

    def pinball(r):
        return rs.r_sum(r * (tau - (r < 0)))

    if method == "pinball":
        v_cap = 2 * v.max()

        def loss(p):
            if p[0] <= 0 or p[0] > v_cap or p[1] < slope_range[0] or p[1] > slope_range[1]:
                return math.inf
            return pinball(v - tobler_speed(m, p))

        starts = [theta, theta * np.array([1.1, 1, 1.2, 1.2]), theta * np.array([0.9, 1, 0.8, 0.8]),
                  theta + np.array([0, 0.03, 0, 0]), theta - np.array([0, 0.03, 0, 0])]
        best = None
        for st in starts:
            par, value, _ = rs.nelder_mead(loss, st, maxit=4000, reltol=1e-10)
            if best is None or value < best[1]:
                best = (par, value)
        theta = best[0]

    if theta[0] > 10:
        warnings.warn(f"v1 = {round(theta[0], 1)} m/s is implausible for human locomotion; "
                      "check the time of the stream.")
    r1 = 1 - pinball(v - tobler_speed(m, theta)) / pinball(v - rs.quantile7(v, tau))
    return pd.DataFrame({"v1": [theta[0]], "m1": [theta[1]], "c_plus": [theta[2]],
                         "c_minus": [theta[3]], "tau": [tau], "n": [v.size], "r1": [r1],
                         "method": [method]})
