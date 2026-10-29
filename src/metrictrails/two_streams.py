"""Comparing streams, linear referencing and map matching.

Port of R/compare_stream.R, R/detect_stream.R, R/project_stream.R and
R/match_stream.R.
"""

import math

import numpy as np
import pandas as pd

from . import _rstats as rs
from ._utils import as_frame, check_columns, nearest_neighbour, to_lonlat, to_metric, utm_epsg
from ._strtree import SegmentIndex
from .tobler import tobler_speed


# ---- compare_stream -------------------------------------------------------------

def compare_stream(a, b, threshold=1, dims=2):
    """Hausdorff distance, median separation and metric score between two streams."""
    cols = ["lon", "lat", "ele"] if dims == 3 else ["lon", "lat"]
    check_columns(a, cols, "compare_stream")
    check_columns(b, cols, "compare_stream")
    epsg = utm_epsg(a["lon"], a["lat"])
    pa = to_metric(a["lon"], a["lat"], epsg)
    pb = to_metric(b["lon"], b["lat"], epsg)
    if dims == 3:
        pa = np.column_stack([pa, np.asarray(a["ele"], dtype=float)])
        pb = np.column_stack([pb, np.asarray(b["ele"], dtype=float)])
    pa = pa[~np.isnan(pa).any(axis=1)]
    pb = pb[~np.isnan(pb).any(axis=1)]
    _, d_ab = nearest_neighbour(pa, pb)
    _, d_ba = nearest_neighbour(pb, pa)
    return pd.DataFrame({
        "dims": [dims],
        "hausdorff": [max(d_ab.max(), d_ba.max())],
        "hausdorff_ab": [d_ab.max()],
        "hausdorff_ba": [d_ba.max()],
        "separation_ab": [rs.r_median(d_ab)],
        "separation_ba": [rs.r_median(d_ba)],
        "metric_score_ab": [100 * rs.r_mean(d_ab <= threshold)],
        "metric_score_ba": [100 * rs.r_mean(d_ba <= threshold)],
    })


# ---- detect_stream --------------------------------------------------------------

def _empty_passes(diagnosis):
    out = pd.DataFrame({"pass": pd.Series(dtype=int), "start_row": pd.Series(dtype=int),
                        "end_row": pd.Series(dtype=int), "direction": pd.Series(dtype=object),
                        "coverage": pd.Series(dtype=float), "status": pd.Series(dtype=object)})
    out.attrs["diagnosis"] = diagnosis
    return out


def detect_stream(data, route, tolerance=50, min_completion=1, min_coverage=0.97, min_partial=0.25):
    """Detect the passes of a stream along a route (rows are 1-based, as in R)."""
    check_columns(data, ["lon", "lat"], "detect_stream")
    check_columns(route, ["lon", "lat"], "detect_stream")
    epsg = utm_epsg(route["lon"], route["lat"])
    rxy = to_metric(route["lon"], route["lat"], epsg)
    dxy = to_metric(data["lon"], data["lat"], epsg)
    rxy = rxy[~np.isnan(rxy).any(axis=1)]
    n = dxy.shape[0]
    m = rxy.shape[0]

    nearest, d_data = nearest_neighbour(dxy, rxy)
    _, d_route = nearest_neighbour(rxy, dxy)
    if np.nanmin(d_data) > tolerance:
        return _empty_passes("streams do not intersect")
    with np.errstate(invalid="ignore"):
        completion = rs.r_mean(d_route <= tolerance)
    if completion < min_completion:
        return _empty_passes("streams intersect but the stream covers only "
                             f"{_r_round(100 * completion, 1):g}% of the route")

    route_length = rs.r_sum(np.sqrt(np.diff(rxy[:, 0]) ** 2 + np.diff(rxy[:, 1]) ** 2))
    step = np.concatenate([[0.0], np.sqrt(np.diff(dxy[:, 0]) ** 2 + np.diff(dxy[:, 1]) ** 2)])
    step[np.isnan(step)] = 0
    cum = np.cumsum(step)

    ends = [rxy[0], rxy[m - 1]]
    near_end = []
    for e in range(2):
        with np.errstate(invalid="ignore"):
            close = np.sqrt((dxy[:, 0] - ends[e][0]) ** 2 + (dxy[:, 1] - ends[e][1]) ** 2) <= tolerance
        near_end.append(close)

    passes = []
    idx = np.arange(1, n + 1)
    for e in range(2):
        close = near_end[e]
        prev = np.concatenate([[False], close[:-1]])
        starts = np.where(close & ~prev)[0] + 1
        target = near_end[1 - e]
        for start in starts:
            arrival = np.where(target & (idx > start) & (cum - cum[start - 1] >= 0.5 * route_length))[0]
            if arrival.size > 0:
                end = arrival[0] + 1
            else:
                end = int(np.max(np.where(cum <= cum[start - 1] + route_length + tolerance)[0])) + 1
            if end - start < 2:
                continue
            sub = dxy[start - 1:end]
            _, dcov = nearest_neighbour(rxy, sub)
            covered = dcov <= tolerance
            rho = rs.cor_spearman(np.arange(1, end - start + 2), nearest[start - 1:end])
            passes.append({"start_row": int(start), "end_row": int(end),
                           "direction": "reverse" if (not math.isnan(rho) and rho < 0) else "forward",
                           "coverage": rs.r_mean(covered)})
    if passes:
        p = pd.DataFrame(passes)
        p = p[p["coverage"] >= min_partial]
        p = p.iloc[np.lexsort((p["start_row"].to_numpy(), -p["coverage"].to_numpy()))].reset_index(drop=True)
        keep = np.ones(len(p), dtype=bool)
        for i in range(len(p)):
            if not keep[i]:
                continue
            overlap = ((p["start_row"] <= p["end_row"].iloc[i]) & (p["end_row"] >= p["start_row"].iloc[i])).to_numpy().copy()
            overlap[i] = False
            keep[overlap & (np.arange(len(p)) > i)] = False
        p = p[keep]
        p = p.sort_values("start_row", kind="mergesort").reset_index(drop=True)
        passes = p if len(p) else None
    else:
        passes = None
    if passes is None:
        return _empty_passes("the stream never passes through the start or end of the route")
    passes["status"] = np.where(passes["coverage"] >= min_coverage, "match", "partial")
    passes.insert(0, "pass", np.arange(1, len(passes) + 1))
    passes.attrs["diagnosis"] = "match" if (passes["status"] == "match").any() else "partial match only"
    return passes


def _r_round(x, digits):
    """round() in R (IEC 60559, half to even on the decimal representation)."""
    return float(np.round(x, digits))


# ---- project / locate -------------------------------------------------------------

def project_stream(data, epsg=None):
    """Add metric coordinates `x`, `y` (UTM zone of the stream by default)."""
    check_columns(data, ["lon", "lat"], "project_stream")
    data = as_frame(data)
    if epsg is None:
        epsg = utm_epsg(data["lon"], data["lat"])
    xy = to_metric(data["lon"], data["lat"], epsg)
    data["x"] = xy[:, 0]
    data["y"] = xy[:, 1]
    data.attrs["epsg"] = int(epsg)
    return data


def _point_segment_distance(px, py, ax, ay, bx, by):
    """GEOS Distance::pointToSegment, vectorised (same operations, same rounding)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        dxs = bx - ax
        dys = by - ay
        len2 = dxs * dxs + dys * dys
        r = ((px - ax) * dxs + (py - ay) * dys) / len2
        s = ((ay - py) * dxs - (ax - px) * dys) / len2
        d = np.abs(s) * np.sqrt(len2)
        da = np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        db = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
        d = np.where(r <= 0.0, da, d)
        d = np.where(r >= 1.0, db, d)
        d = np.where(len2 == 0, da, d)
    return d


def project_on_line(px, py, rx, ry, rs_):
    """Orthogonal projection of points on a polyline: (s, signed cross distance).

    The nearest segment is found as sf::st_nearest_feature does (GEOS
    STRtree), so that exact ties are resolved as in R.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    ax, ay = rx[:-1], ry[:-1]
    bx, by = rx[1:], ry[1:]
    sdx = bx - ax
    sdy = by - ay
    slen2 = sdx * sdx + sdy * sdy
    index = SegmentIndex(rx, ry)
    a = np.array([index.nearest(x, y) for x, y in zip(px, py)], dtype=int)
    dx = sdx[a]
    dy = sdy[a]
    len2 = slen2[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(len2 > 0, ((px - ax[a]) * dx + (py - ay[a]) * dy) / len2, 0)
    u = np.minimum(np.maximum(u, 0), 1)
    qx = ax[a] + u * dx
    qy = ay[a] + u * dy
    d = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)
    side = np.sign(dx * (py - qy) - dy * (px - qx))
    return rs_[a] + u * np.sqrt(len2), side * d


def locate_stream(data, reference):
    """Linear referencing of every reading on the polyline of `reference`."""
    check_columns(data, ["lon", "lat"], "locate_stream")
    check_columns(reference, ["lon", "lat"], "locate_stream")
    data = as_frame(data)
    epsg = utm_epsg(reference["lon"], reference["lat"])
    rxy = to_metric(reference["lon"], reference["lat"], epsg)
    if np.isnan(rxy).any():
        raise ValueError("The reference has missing coordinates.")
    if rxy.shape[0] < 2:
        raise ValueError("The reference needs at least 2 readings.")
    rs_ = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(rxy[:, 0]) ** 2 + np.diff(rxy[:, 1]) ** 2))])
    oxy = to_metric(data["lon"], data["lat"], epsg)
    ok = ~np.isnan(oxy).any(axis=1)
    s = np.full(len(data), np.nan)
    cross = np.full(len(data), np.nan)
    ps, pc = project_on_line(oxy[ok, 0], oxy[ok, 1], rxy[:, 0], rxy[:, 1], rs_)
    s[ok] = ps
    cross[ok] = pc
    px = rs.approx(rs_, rxy[:, 0], s, ties="ordered")
    py = rs.approx(rs_, rxy[:, 1], s, ties="ordered")
    ll = to_lonlat(px, py, epsg)
    data["s_ref"] = s
    data["offset_cross"] = cross
    data["lon_ref"] = ll[:, 0]
    data["lat_ref"] = ll[:, 1]
    return data


# ---- match_stream -----------------------------------------------------------------

def match_stream(data, reference, params, direction="auto", by=None, dims=2, lambda_=1,
                 tol_factor=1.1, slope_clip=0.6, max_cells=5e7):
    """Align a stream along a reference trail by dynamic programming (Viterbi).

    `lambda_` is the `lambda` argument of the R function (a reserved word in
    Python). The report is stored in `result.attrs["match_report"]`.
    """
    if direction not in ("auto", "forward", "reverse"):
        raise ValueError("'direction' must be 'auto', 'forward' or 'reverse'")
    need = ["lon", "lat", "time"] + (["ele"] if dims == 3 else []) + ([by] if by else [])
    check_columns(data, need, "match_stream")
    check_columns(reference, ["lon", "lat"], "match_stream")
    data = as_frame(data)

    epsg = utm_epsg(reference["lon"], reference["lat"])
    rxy = to_metric(reference["lon"], reference["lat"], epsg)
    rz = (np.asarray(reference["ele"], dtype=float) if "ele" in reference.columns
          else np.zeros(len(reference)))
    if np.isnan(rxy).any() or np.isnan(rz).any():
        raise ValueError("The reference has missing coordinates or elevations.")
    rs_ = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(rxy[:, 0]) ** 2 + np.diff(rxy[:, 1]) ** 2))])
    m = rxy.shape[0]
    ref_lon = np.asarray(reference["lon"], dtype=float)
    ref_lat = np.asarray(reference["lat"], dtype=float)

    if by is None:
        group = np.ones(len(data), dtype=int)
    else:
        codes, _ = pd.factorize(data[by], sort=False)
        group = codes + 1

    out = data.copy()
    out["lon_obs"] = data["lon"]
    out["lat_obs"] = data["lat"]
    if "ele" in data.columns:
        out["ele_obs"] = data["ele"]
    new_cols = ["ref_index", "s_ref", "offset_d", "offset_cross", "offset_along",
                "speed_h", "speed_v", "speed_total", "slope_ref"]
    for col in new_cols:
        out[col] = np.nan
    out["ele"] = np.nan
    lon_new = np.asarray(out["lon"], dtype=float).copy()
    lat_new = np.asarray(out["lat"], dtype=float).copy()
    cols = {c: np.full(len(data), np.nan) for c in new_cols + ["ele"]}
    reports = []

    for g in pd.unique(group):
        rows = np.where(group == g)[0]
        oxy = to_metric(data["lon"].iloc[rows], data["lat"].iloc[rows], epsg)
        t = np.asarray(data["time"].iloc[rows], dtype=float)
        n = rows.size
        if n * m > max_cells:
            raise ValueError(f"Round with {n} readings and a reference of {m} vertices exceeds "
                             "'max_cells'; split the stream into rounds or use a coarser reference.")
        proj_s, _ = project_on_line(oxy[:, 0], oxy[:, 1], rxy[:, 0], rxy[:, 1], rs_)

        dir_g = direction
        if dir_g == "auto":
            lo = math.ceil(0.1 * n)
            hi = max(lo, math.floor(0.9 * n))
            inner = np.arange(lo, hi + 1)
            rho = rs.cor_spearman(np.arange(1, inner.size + 1), proj_s[inner - 1])
            dir_g = "reverse" if (not math.isnan(rho) and rho < 0) else "forward"

        ord_ = np.arange(m) if dir_g == "forward" else np.arange(m - 1, -1, -1)
        gx = rxy[ord_, 0]
        gy = rxy[ord_, 1]
        gz = rz[ord_]
        gs = rs_ if dir_g == "forward" else rs_[m - 1] - rs_[ord_]
        oz = np.asarray(data["ele"].iloc[rows], dtype=float) if dims == 3 else None

        j, objective = viterbi_match(oxy[:, 0], oxy[:, 1], oz, t, gx, gy, gz, gs, params,
                                     lambda_, tol_factor, slope_clip)
        j_ref = ord_[j]

        lon_new[rows] = ref_lon[j_ref]
        lat_new[rows] = ref_lat[j_ref]
        cols["ele"][rows] = rz[j_ref]
        cols["ref_index"][rows] = j_ref + 1
        cols["s_ref"][rows] = rs_[j_ref]
        cols["offset_d"][rows] = np.sqrt((oxy[:, 0] - rxy[j_ref, 0]) ** 2 + (oxy[:, 1] - rxy[j_ref, 1]) ** 2)
        prev_j = np.maximum(j - 1, 0)
        next_j = np.minimum(j + 1, m - 1)
        tx = gx[next_j] - gx[prev_j]
        ty = gy[next_j] - gy[prev_j]
        norm = np.sqrt(tx ** 2 + ty ** 2)
        norm[norm == 0] = np.nan
        tx = tx / norm
        ty = ty / norm
        ex = oxy[:, 0] - gx[j]
        ey = oxy[:, 1] - gy[j]
        cols["offset_along"][rows] = ex * tx + ey * ty
        cols["offset_cross"][rows] = tx * ey - ty * ex

        ds = np.concatenate([[np.nan], np.diff(gs[j])])
        dz = np.concatenate([[np.nan], np.diff(gz[j])])
        dt = np.concatenate([[np.nan], np.diff(t)])
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = ~np.isnan(dt) & (dt > 0)
            cols["speed_h"][rows] = np.where(ok, ds / dt, np.nan)
            cols["speed_v"][rows] = np.where(ok, np.abs(dz) / dt, np.nan)
            cols["speed_total"][rows] = np.where(ok, np.sqrt(ds ** 2 + dz ** 2) / dt, np.nan)
            pos = ~np.isnan(ds) & (ds > 0)
            cols["slope_ref"][rows] = np.where(pos, dz / ds, np.nan)

            load = np.bincount(j, minlength=m)
            cap = max(2, math.ceil(n / m))
            slope_c = np.minimum(np.maximum(np.where(pos, dz / ds, 0), -slope_clip), slope_clip)
            ceiling_h = tobler_speed(slope_c, params) * tol_factor
            at_ceiling = ok & (ds > 0) & ((ds / dt) >= 0.98 * ceiling_h)
        reports.append({
            "group": None if by is None else data[by].iloc[rows[0]],
            "direction": dir_g,
            "n": n,
            "vertices_used": int((load > 0).sum()),
            "max_per_vertex": int(load.max()),
            "vertices_over_cap": int((load > cap).sum()),
            "pct_at_ceiling": 100 * rs.r_mean(at_ceiling[ok]),
            "objective": objective,
            "offset_median": rs.r_median(cols["offset_d"][rows]),
            "offset_q95": rs.quantile7(cols["offset_d"][rows], 0.95),
        })

    out["lon"] = lon_new
    out["lat"] = lat_new
    for c, v in cols.items():
        out[c] = v
    out.attrs["match_report"] = pd.DataFrame(reports)
    return out


def viterbi_match(ox, oy, oz, t, gx, gy, gz, gs, params, lam, tol_factor, slope_clip):
    """Exact minimisation of the matching objective; returns (0-based path, objective)."""
    n = ox.size
    m = gx.size
    slopes = rs.seq_by(-slope_clip, slope_clip, 0.01)
    v_top = np.max(tobler_speed(slopes, params) * tol_factor * np.sqrt(1 + slopes ** 2))

    def unary(i):
        u = (ox[i] - gx) ** 2 + (oy[i] - gy) ** 2
        if oz is not None:
            u = u + (oz[i] - gz) ** 2
        return u

    D = unary(0)
    back = np.zeros((n, m), dtype=np.int64)
    idx = np.arange(m)
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        best = D.copy()
        arg = idx.copy()
        if not np.isnan(dt) and dt > 0:
            reach = rs.find_interval(gs + v_top * dt, gs)
            k_max = int(np.max(reach - (idx + 1)))
            for k in range(1, k_max + 1):
                frm = np.arange(0, m - k)
                to = frm + k
                ds = gs[to] - gs[frm]
                dz = gz[to] - gz[frm]
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    slope = np.where(ds > 0, dz / ds, 0)
                    slope_c = np.minimum(np.maximum(slope, -slope_clip), slope_clip)
                    vh_c = tobler_speed(slope_c, params) * tol_factor
                    allowed = (ds == 0) | (np.sqrt(ds ** 2 + dz ** 2) / dt <= vh_c * np.sqrt(1 + slope_c ** 2) + 1e-9)
                    vh = tobler_speed(slope, params) * tol_factor
                    excess = np.maximum(ds / dt - vh, 0)
                    pen = excess ** 2 * (1 + slope ** 2 + (1 + slope ** 2)) / lam ** 2
                pen[ds == 0] = 0
                cand = D[frm] + pen
                cand[~allowed] = np.inf
                update = cand < best[to]
                best[to[update]] = cand[update]
                arg[to[update]] = frm[update]
        D = best + unary(i)
        back[i] = arg
    j = np.empty(n, dtype=int)
    j[n - 1] = int(np.argmin(D))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return j, float(np.min(D))
