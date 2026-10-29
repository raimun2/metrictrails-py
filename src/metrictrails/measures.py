"""Derived measures (port of R/differential_stream.R and R/summarise_stream.R)."""

import math

import numpy as np
import pandas as pd

from . import _rstats as rs
from ._utils import as_frame, check_columns, diff_by_segment, is_first_of_segment, segment_id, to_metric, utm_epsg


def differential_stream(data, distance="utm", idle=0.1):
    """Distance, heading change, elevation, speed and acceleration between readings."""
    if distance not in ("utm", "haversine"):
        raise ValueError("'distance' must be 'utm' or 'haversine'")
    check_columns(data, ["lon", "lat"], "differential_stream")
    data = as_frame(data)
    seg = segment_id(data)
    n = len(data)
    is_first = is_first_of_segment(seg)

    epsg = utm_epsg(data["lon"], data["lat"])
    xy = to_metric(data["lon"], data["lat"], epsg)
    dx = diff_by_segment(xy[:, 0], seg)
    dy = diff_by_segment(xy[:, 1], seg)

    with np.errstate(invalid="ignore", divide="ignore"):
        if distance == "utm":
            delta_d = np.sqrt(dx ** 2 + dy ** 2)
        else:
            rad = math.pi / 180
            lat = np.asarray(data["lat"], dtype=float)
            lon = np.asarray(data["lon"], dtype=float)
            lat1 = np.concatenate([[np.nan], lat[:-1]]) * rad
            lat2 = lat * rad
            dlat = lat2 - lat1
            dlon = (lon - np.concatenate([[np.nan], lon[:-1]])) * rad
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            delta_d = 2 * 6371008.8 * np.arcsin(np.minimum(1, np.sqrt(a)))
            delta_d[is_first] = 0
        data["delta_d"] = delta_d
        data["distance"] = np.cumsum(delta_d)

        moving = delta_d >= idle
        heading = np.where(moving & ~is_first, np.arctan2(dy, dx), np.nan)
        prev = np.concatenate([[np.nan], heading[:-1]])
        turn = np.mod(heading - prev + math.pi, 2 * math.pi) - math.pi
        turn[is_first] = np.nan
        data["turn_angle"] = np.abs(turn) * 180 / math.pi

        has_ele = "ele" in data.columns
        if has_ele:
            delta_ele = diff_by_segment(data["ele"], seg)
            data["delta_ele"] = delta_ele
            data["dplus"] = np.maximum(delta_ele, 0)
            data["dminus"] = np.maximum(-delta_ele, 0)
            data["delta_d3d"] = np.sqrt(delta_d ** 2 + delta_ele ** 2)
            data["slope"] = np.where(moving & ~is_first, delta_ele / delta_d, np.nan)

        if "time" in data.columns:
            delta_time = diff_by_segment(data["time"], seg)
            data["delta_time"] = delta_time
            ok = ~is_first & (delta_time > 0)
            speed = np.where(ok, delta_d / delta_time, np.nan)
            data["speed"] = speed
            data["accel"] = np.where(ok, diff_by_segment(speed, seg, first=np.nan) / delta_time, np.nan)
            data["pace"] = np.where(speed > 0, 1000 / 60 / speed, np.nan)
            if has_ele:
                vert = np.where(ok, delta_ele / delta_time, np.nan)
                data["vert_speed"] = vert
                data["vert_accel"] = np.where(ok, diff_by_segment(vert, seg, first=np.nan) / delta_time,
                                              np.nan)
                data["speed_3d"] = np.where(ok, data["delta_d3d"].to_numpy() / delta_time, np.nan)
    return data


def _r_mod(a, b):
    return np.mod(a, b)


def summarise_stream(data, by=None, idle=0.1, fractal=False):
    """One row of summary measures per stream (or per group of `by` columns)."""
    if isinstance(by, str):
        by = [by]
    check_columns(data, ["lon", "lat"] + list(by or []), "summarise_stream")
    data = as_frame(data)
    if by is None:
        group = np.ones(len(data), dtype=int)
    else:
        key = data[by].astype(str).agg("\r".join, axis=1)
        codes, _ = pd.factorize(key, sort=False)
        group = codes + 1

    rows = []
    for g in pd.unique(group):
        sub = data.loc[group == g].reset_index(drop=True)
        d = differential_stream(sub, idle=idle)
        dd = d["delta_d"].to_numpy()
        is_idle = dd < idle
        is_idle[0] = False
        xy = to_metric(sub["lon"], sub["lat"], utm_epsg(sub["lon"], sub["lat"]))
        straight = math.sqrt((xy[-1, 0] - xy[0, 0]) ** 2 + (xy[-1, 1] - xy[0, 1]) ** 2)
        dist2 = rs.r_sum(dd)
        row = {
            "n": len(sub),
            "distance_2d": dist2,
            "resolution": rs.r_mean(dd[1:]),
            "sinuosity": dist2 / straight if straight != 0 else (math.inf if dist2 > 0 else math.nan),
            "mean_turn": rs.r_mean(rs.na_omit(d["turn_angle"])),
            "pct_idle": 100 * rs.r_mean(is_idle[1:]),
        }
        if "ele" in sub.columns:
            ele = np.asarray(sub["ele"], dtype=float)
            row["distance_3d"] = rs.r_sum(rs.na_omit(d["delta_d3d"]))
            row["elevation_gain"] = rs.r_sum(rs.na_omit(d["dplus"]))
            row["elevation_loss"] = rs.r_sum(rs.na_omit(d["dminus"]))
            row["ele_min"] = np.nanmin(ele)
            row["ele_max"] = np.nanmax(ele)
            row["mean_slope"] = 100 * rs.r_mean(rs.na_omit(d["slope"]))
            row["iqr_slope"] = 100 * rs.r_iqr(d["slope"])
            row["km_effort"] = row["distance_2d"] / 1000 + row["elevation_gain"] / 100
            row["idle_gain"] = rs.r_sum(rs.na_omit(d["dplus"].to_numpy()[is_idle]))
            row["idle_loss"] = rs.r_sum(rs.na_omit(d["dminus"].to_numpy()[is_idle]))
        if "time" in sub.columns:
            t = rs.na_omit(sub["time"])
            row["duration"] = t.max() - t.min()
        if fractal:
            from .fractal import fractal_stream
            try:
                row["fractal_dim"] = fractal_stream(sub)["dimension"]
            except ValueError:
                row["fractal_dim"] = math.nan
        if by is not None:
            row = {**{b: sub[b].iloc[0] for b in by}, **row}
        rows.append(row)
    return pd.DataFrame(rows)
