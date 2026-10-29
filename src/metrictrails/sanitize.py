"""Sanitising coordinates and red flags (port of R/sanitize_stream.R)."""

import math
import warnings

import numpy as np

from . import _rstats as rs
from ._utils import (as_frame, check_columns, diff_by_segment, is_first_of_segment, segment_id,
                     to_lonlat, to_metric, utm_epsg)
from .tobler import tobler_speed

METHODS = ("kalman", "loess", "gaussian", "savgol", "pipeline", "none")
ACTIONS = ("flag", "drop", "interpolate")
FLAG_NAMES = ["flag_time", "flag_speed", "flag_accel", "flag_jump", "flag_slope",
              "flag_vert_speed", "flag_idle_ele", "flag_gap", "flag_low_res"]


def sanitize_stream(data, method="kalman", action="flag", smooth_ele=False,
                    v_max=10, accel_max=6, vert_speed_max=1, slope_max=1.19,
                    gap=30, low_res=20, edge_jump=30, idle=0.1,
                    tobler=None, tol_factor=1.1,
                    sigma_obs=5, sigma_acc=1, rts=True,
                    alpha=0.05, bandwidth=10,
                    sg_window=9, sg_order=2,
                    window=25, jump_k=1.5, local_k=2):
    """Smooth the horizontal position of a stream and flag implausible readings.

    The report is stored in `result.attrs["sanitize_report"]`.
    """
    if method not in METHODS:
        raise ValueError(f"'method' must be one of {METHODS}")
    if action not in ACTIONS:
        raise ValueError(f"'action' must be one of {ACTIONS}")
    check_columns(data, ["lon", "lat"], "sanitize_stream")
    data = as_frame(data)
    has_time = "time" in data.columns
    has_ele = "ele" in data.columns

    # ---- 1. flags on raw values ------------------------------------------------
    lon = np.asarray(data["lon"], dtype=float)
    lat = np.asarray(data["lat"], dtype=float)
    with np.errstate(invalid="ignore"):
        flag_coord = (np.isnan(lon) | np.isnan(lat) | (np.abs(lat) > 90) | (np.abs(lon) > 180)
                      | ((lon == 0) & (lat == 0)))
    n_coord = int(flag_coord.sum())
    data = data.loc[~flag_coord].reset_index(drop=True)
    if len(data) < 3:
        raise ValueError("Fewer than 3 readings with valid coordinates.")

    epsg = utm_epsg(data["lon"], data["lat"])
    xy = to_metric(data["lon"], data["lat"], epsg)
    seg = segment_id(data)
    n = len(data)
    is_first = is_first_of_segment(seg)

    delta_d = np.sqrt(diff_by_segment(xy[:, 0], seg) ** 2 + diff_by_segment(xy[:, 1], seg) ** 2)
    with np.errstate(invalid="ignore"):
        flag_low_res = delta_d > low_res
    flag_time = np.zeros(n, dtype=bool)
    flag_gap = np.zeros(n, dtype=bool)
    flag_speed = np.zeros(n, dtype=bool)
    flag_accel = np.zeros(n, dtype=bool)
    flag_jump = np.zeros(n, dtype=bool)
    flag_slope = np.zeros(n, dtype=bool)
    flag_vert_speed = np.zeros(n, dtype=bool)
    speed = np.full(n, np.nan)
    flag_idle_ele = np.zeros(n, dtype=bool)

    ele = np.asarray(data["ele"], dtype=float) if has_ele else None
    delta_e = diff_by_segment(ele, seg) if has_ele else np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        if has_ele:
            moving = delta_d >= idle
            flag_slope = moving & (np.abs(delta_e / delta_d) > slope_max)
            flag_idle_ele = ~moving & ~is_first & (np.abs(delta_e) > 0)

        if has_time:
            tt = np.asarray(data["time"], dtype=float)
            delta_t = diff_by_segment(tt, seg)
            flag_time = ~is_first & (np.isnan(delta_t) | (delta_t <= 0))
            flag_gap = ~is_first & ~np.isnan(delta_t) & (delta_t > gap)
            ok_t = ~is_first & ~np.isnan(delta_t) & (delta_t > 0)
            speed[ok_t] = delta_d[ok_t] / delta_t[ok_t]
            if tobler is None:
                speed_max = np.full(n, float(v_max))
            else:
                slope = np.zeros(n)
                if has_ele:
                    slope[ok_t] = np.where(delta_d[ok_t] > idle, delta_e[ok_t] / delta_d[ok_t], 0)
                speed_max = tobler_speed(np.minimum(np.maximum(slope, -0.6), 0.6), tobler) * tol_factor
            flag_speed = ~np.isnan(speed) & (speed > speed_max)
            accel = diff_by_segment(speed, seg, first=np.nan) / delta_t
            accel[~ok_t] = np.nan
            flag_accel = ~np.isnan(accel) & (np.abs(accel) > accel_max)
            if has_ele:
                vert = np.full(n, np.nan)
                vert[ok_t] = delta_e[ok_t] / delta_t[ok_t]
                flag_vert_speed = ~np.isnan(vert) & (np.abs(vert) > vert_speed_max)
            for s in np.unique(seg):
                rows = np.where(seg == s)[0]
                k = rows.size
                if k < 7:
                    continue
                for i in range(1, 4):
                    if delta_d[rows[i]] > edge_jump:
                        flag_jump[rows[i - 1]] = True
                    if delta_d[rows[k - i]] > edge_jump:
                        flag_jump[rows[k - i]] = True

    if has_time:
        exceed = speed[~is_first]
        exceed = exceed[~np.isnan(exceed)]
        if exceed.size and np.mean(exceed > v_max) > 0.5:
            warnings.warn("More than half of the readings exceed the speed limit: time is not physical "
                          "and the stream is smoothed as a route without time.")
            has_time = False
            flag_speed[:] = False
            flag_accel[:] = False
            flag_vert_speed[:] = False
            flag_gap[:] = False
            flag_jump[:] = False

    flags = {"flag_time": flag_time, "flag_speed": flag_speed, "flag_accel": flag_accel,
             "flag_jump": flag_jump, "flag_slope": flag_slope, "flag_vert_speed": flag_vert_speed,
             "flag_idle_ele": flag_idle_ele, "flag_gap": flag_gap, "flag_low_res": flag_low_res}
    position_flag = flag_time | flag_speed | flag_accel | flag_jump
    ele_flag = flag_slope | flag_vert_speed | flag_idle_ele

    report = {
        "n_readings": n + n_coord,
        "flag_coord": n_coord,
        "counts": {k: int(v.sum()) for k, v in flags.items()},
        "share_integer_ele": (rs.r_mean((ele == np.round(ele))[~np.isnan(ele)])
                              if has_ele else float("nan")),
        "share_low_res": rs.r_mean(flag_low_res),
        "idle_gain": rs.r_sum(np.maximum(delta_e[flag_idle_ele], 0)) if has_ele else float("nan"),
        "idle_loss": rs.r_sum(np.minimum(delta_e[flag_idle_ele], 0)) if has_ele else float("nan"),
    }
    sie = report["share_integer_ele"]
    report["flag_ele_rounding"] = bool(not np.isnan(sie) and sie > 0.9)

    # ---- 2. act on flagged readings --------------------------------------------
    if action == "drop":
        keep = ~position_flag
        if has_ele:
            ele = ele.copy()
            ele[ele_flag] = np.nan
            data["ele"] = ele
        data = data.loc[keep].reset_index(drop=True)
        flags = {k: v[keep] for k, v in flags.items()}
        xy = xy[keep]
        seg = seg[keep]
    elif action == "interpolate":
        xy[position_flag] = np.nan
        if has_ele:
            ele = ele.copy()
            ele[ele_flag | position_flag] = np.nan
        tt = np.asarray(data["time"], dtype=float) if has_time else None
        for s in np.unique(seg):
            rows = np.where(seg == s)[0]
            base = tt[rows] if has_time else np.arange(1, rows.size + 1, dtype=float)
            for j in range(2):
                xy[rows, j] = fill_linear(base, xy[rows, j])
            if has_ele:
                ele[rows] = fill_linear(base, ele[rows])
        if has_ele:
            data["ele"] = ele

    # ---- 3. smoothing ----------------------------------------------------------
    if method != "none":
        tt = np.asarray(data["time"], dtype=float) if has_time else None
        ele_now = np.asarray(data["ele"], dtype=float).copy() if has_ele else None
        for s in _unique_in_order(seg):
            rows = np.where(seg == s)[0]
            if rows.size < 5:
                continue
            if has_time:
                t = tt[rows]
            else:
                t = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(xy[rows, 0]) ** 2
                                                             + np.diff(xy[rows, 1]) ** 2))])
            z = ele_now[rows] if (has_ele and smooth_ele) else None
            out = smooth_segment(xy[rows, 0].copy(), xy[rows, 1].copy(), z, t, method, has_time,
                                 sigma_obs, sigma_acc, rts, alpha, bandwidth, sg_window, sg_order,
                                 window, jump_k, local_k, edge_jump, gap)
            xy[rows, 0] = out[0]
            xy[rows, 1] = out[1]
            if z is not None:
                ele_now[rows] = out[2]
        if has_ele and smooth_ele:
            data["ele"] = ele_now

    ll = to_lonlat(xy[:, 0], xy[:, 1], epsg)
    data["lon"] = ll[:, 0]
    data["lat"] = ll[:, 1]
    for f in FLAG_NAMES:
        data[f] = flags[f]
    data.attrs["sanitize_report"] = report
    return data


def _unique_in_order(v):
    seen = []
    for x in v:
        if x not in seen:
            seen.append(x)
    return seen


def fill_linear(x, y):
    """Linear interpolation of the NaN values of y along x (ends take the nearest value)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    ok = ~np.isnan(y) & ~np.isnan(x)
    if ok.sum() < 2 or ok.all():
        return y
    y[~ok] = rs.approx(x[ok], y[ok], x[~ok], ties="mean", rule=2)
    return y


def smooth_segment(x, y, z, t, method, has_time, sigma_obs, sigma_acc, rts, alpha, bandwidth,
                   sg_window, sg_order, window, jump_k, local_k, edge_jump, gap):
    reset = gap if has_time else math.inf
    if method == "kalman":
        k = kalman_cv(np.column_stack([x, y]), t, sigma_obs, sigma_acc, rts, reset)
        zz = None if z is None else kalman_cv(z[:, None], t, sigma_obs, sigma_acc, rts, reset)[:, 0]
        return k[:, 0], k[:, 1], zz
    if method == "loess":
        def fit(v):
            ok = ~np.isnan(v)
            return rs.loess_direct(t[ok], v[ok], alpha, t)
        return fit(x), fit(y), (None if z is None else fit(z))
    if method == "gaussian":
        if not has_time:
            raise ValueError('method = "gaussian" needs a \'time\' column.')
        return (gaussian_kernel(t, x, bandwidth), gaussian_kernel(t, y, bandwidth),
                None if z is None else gaussian_kernel(t, z, bandwidth))
    if method == "savgol":
        return (savgol(x, sg_window, sg_order), savgol(y, sg_window, sg_order),
                None if z is None else savgol(z, sg_window, sg_order))

    # method == "pipeline" (Martinez, Sanchez & Diaz, 2026)
    k = kalman_cv(np.column_stack([x, y]), t, sigma_obs, sigma_acc, False, reset)
    x = k[:, 0].copy()
    y = k[:, 1].copy()
    if z is not None:
        z = kalman_cv(z[:, None], t, sigma_obs, sigma_acc, False, reset)[:, 0].copy()

    with np.errstate(invalid="ignore", divide="ignore"):
        dt = np.concatenate([[np.nan], np.diff(t)])
        dd = np.concatenate([[np.nan], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)])
        speed = dd / dt
        derived = [speed, np.concatenate([[np.nan], np.diff(speed)]) / dt]
        if z is not None:
            derived.append(np.concatenate([[np.nan], np.diff(z)]) / dd)
        outlier = np.zeros(x.size, dtype=bool)
        for v in derived:
            v = v.copy()
            v[~np.isfinite(v)] = np.nan
            q = rs.quantile7(v, [0.25, 0.75])
            iqr = q[1] - q[0]
            c1 = (v < q[0] - 1.5 * iqr) | (v > q[1] + 1.5 * iqr)
            dv = np.concatenate([[np.nan], np.diff(v)])
            c2 = np.abs(dv) > jump_k * rs.r_sd(dv)
            local_mean = rs.conv_filter_centered(v, window)
            local_sd = np.sqrt(np.maximum(rs.conv_filter_centered(v ** 2, window) - local_mean ** 2, 0))
            c3 = np.abs(v - local_mean) > local_k * local_sd
            votes = c1.astype(int) + c2.astype(int) + c3.astype(int)
            outlier = outlier | (votes >= 2)
    x[outlier] = np.nan
    y[outlier] = np.nan
    x = fill_linear(t, x)
    y = fill_linear(t, y)
    if z is not None:
        z[outlier] = np.nan
        z = fill_linear(t, z)

    x = savgol(x, sg_window, sg_order)
    y = savgol(y, sg_window, sg_order)
    if z is not None:
        z = savgol(z, sg_window, sg_order)

    n = x.size
    for i in range(1, 4):
        if n > 7 and math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) > edge_jump:
            x[:i] = x[i]
            y[:i] = y[i]
        j = n - i + 1
        if n > 7 and math.sqrt((x[j - 1] - x[j - 2]) ** 2 + (y[j - 1] - y[j - 2]) ** 2) > edge_jump:
            x[j - 1:] = x[j - 2]
            y[j - 1:] = y[j - 2]
    return x, y, z


def kalman_cv(obs, t, sigma_obs, sigma_acc, rts=True, reset=math.inf):
    """Constant-velocity Kalman filter (and RTS smoother) for every column of obs."""
    obs = np.asarray(obs, dtype=float)
    t = np.asarray(t, dtype=float)
    dt_all = np.concatenate([[0.0], np.diff(t)])
    with np.errstate(invalid="ignore"):
        breaks = np.concatenate([[True], np.isnan(dt_all[1:]) | (dt_all[1:] < 0) | (dt_all[1:] > reset)])
    block = np.cumsum(breaks)
    out = obs.copy()
    for b in np.unique(block):
        rows = np.where(block == b)[0]
        out[rows] = kalman_block(obs[rows], t[rows], sigma_obs, sigma_acc, rts)
    return out


def kalman_block(obs, t, sigma_obs, sigma_acc, rts):
    n = obs.shape[0]
    out = obs.copy()
    R = sigma_obs ** 2
    for axis in range(obs.shape[1]):
        z = obs[:, axis]
        valid = np.where(~np.isnan(z))[0]
        if valid.size == 0 or n < 2:
            continue
        first = valid[0]
        m = np.array([z[first], 0.0])
        P = np.diag([R, 100.0])
        m_filt = np.full((n, 2), np.nan)
        P_filt = np.full((n, 2, 2), np.nan)
        m_pred = np.full((n, 2), np.nan)
        P_pred = np.full((n, 2, 2), np.nan)
        for i in range(n):
            dt = 0.0 if i == 0 else max(t[i] - t[i - 1], 0.0)
            Fm = np.array([[1.0, dt], [0.0, 1.0]])
            Q = sigma_acc ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
            m = Fm @ m
            P = Fm @ P @ Fm.T + Q
            m_pred[i] = m
            P_pred[i] = P
            if not np.isnan(z[i]):
                K = P[:, 0] / (P[0, 0] + R)
                m = m + K * (z[i] - m[0])
                P = P - np.outer(K, P[0, :])
            m_filt[i] = m
            P_filt[i] = P
        if rts:
            m_s = m_filt.copy()
            P_s = P_filt.copy()
            for i in range(n - 2, -1, -1):
                dt = max(t[i + 1] - t[i], 0.0)
                Fm = np.array([[1.0, dt], [0.0, 1.0]])
                C = P_filt[i] @ Fm.T @ pseudo_inverse(P_pred[i + 1])
                m_s[i] = m_filt[i] + C @ (m_s[i + 1] - m_pred[i + 1])
                P_s[i] = P_filt[i] + C @ (P_s[i + 1] - P_pred[i + 1]) @ C.T
            m_filt = m_s
        out[:, axis] = m_filt[:, 0]
    return out


def pseudo_inverse(A):
    """Moore-Penrose inverse of a small symmetric matrix (robust to singularity)."""
    u, d, vt = np.linalg.svd(A)
    keep = d > max(A.shape) * d.max() * rs.DBL_EPSILON
    if not keep.any():
        return np.zeros((A.shape[1], A.shape[0]))
    return vt[keep].T @ (u[:, keep].T / d[keep][:, None])


def gaussian_kernel(t, v, bw):
    """Nadaraya-Watson regression with a Gaussian kernel of standard deviation bw."""
    out = np.asarray(v, dtype=float).copy()
    ok = ~np.isnan(out)
    tt = t[ok]
    vv = out[ok]
    res = np.empty(t.size)
    for i in range(t.size):
        near = np.abs(tt - t[i]) <= 4 * bw
        w = rs.dnorm(tt[near] - t[i], bw)
        res[i] = rs.r_sum(w * vv[near]) / rs.r_sum(w)
    return res


def savgol(v, window, order):
    """Savitzky-Golay smoothing; edges use the polynomials of the first and last windows."""
    v = np.asarray(v, dtype=float)
    n = v.size
    if window % 2 == 0:
        window += 1
    if n < window:
        return v.copy()
    half = (window - 1) // 2
    X = np.power.outer(np.arange(-half, half + 1, dtype=float), np.arange(order + 1, dtype=float))
    proj = X @ np.linalg.solve(X.T @ X, X.T)
    out = v.copy()
    coef = proj[half]
    for i in range(half, n - half):
        out[i] = np.sum(coef * v[i - half:i + half + 1])
    out[:half] = proj[:half] @ v[:window]
    out[n - half:] = proj[half + 1:] @ v[n - window:]
    return out
