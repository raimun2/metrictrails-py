"""Replicas of the R routines used by metrictrails.

numpy and scipy offer equivalents of most of these, but with different
conventions (quantile interpolation, tie handling, optimiser stopping rules).
The port must reproduce the R package, so the R algorithms are reimplemented
here from the R sources (stats::approx, findInterval, quantile type 7,
stats::filter, loess with surface = "direct", optimize, optim Nelder-Mead).
"""

import math

import numpy as np

DBL_EPSILON = np.finfo(float).eps


# ---- summaries ----------------------------------------------------------------

def r_mean(x):
    """mean(x) with R's two-pass refinement; NaN if any value is NaN."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    s = math.fsum(x) / x.size
    if not math.isfinite(s):
        return s
    return s + math.fsum(x - s) / x.size


def r_sum(x):
    x = np.asarray(x, dtype=float)
    return math.fsum(x) if x.size else 0.0


def na_omit(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def quantile7(x, probs):
    """stats::quantile(x, probs, type = 7) on non-missing x."""
    x = np.sort(na_omit(x))
    n = x.size
    scalar = np.isscalar(probs)
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    out = np.empty(probs.size)
    if n == 0:
        out[:] = np.nan
        return float(out[0]) if scalar else out
    for k, p in enumerate(probs):
        index = 1 + max(n - 1, 0) * p
        lo = int(math.floor(index))
        hi = int(math.ceil(index))
        qs = x[lo - 1]
        h = index - lo
        if index > lo and x[hi - 1] != qs:
            qs = (1 - h) * qs + h * x[hi - 1]
        out[k] = qs
    return float(out[0]) if scalar else out


def r_median(x):
    x = np.sort(na_omit(x))
    n = x.size
    if n == 0:
        return float("nan")
    half = (n + 1) // 2
    if n % 2 == 1:
        return float(x[half - 1])
    return r_mean(x[half - 1:half + 1])


def r_sd(x):
    x = na_omit(x)
    if x.size < 2:
        return float("nan")
    m = r_mean(x)
    return math.sqrt(math.fsum((x - m) ** 2) / (x.size - 1))


def r_iqr(x):
    q = quantile7(x, [0.25, 0.75])
    return float(q[1] - q[0])


def rank_average(x):
    """rank(x, ties.method = "average")."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def r_cor(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isnan(x).any() or np.isnan(y).any() or x.size < 2:
        return float("nan")
    mx = r_mean(x)
    my = r_mean(y)
    sxx = math.fsum((x - mx) ** 2)
    syy = math.fsum((y - my) ** 2)
    if sxx == 0 or syy == 0:
        return float("nan")
    return math.fsum((x - mx) * (y - my)) / math.sqrt(sxx * syy)


def cor_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isnan(x).any() or np.isnan(y).any():
        return float("nan")
    return r_cor(rank_average(x), rank_average(y))


# ---- sequences and intervals ----------------------------------------------------

def seq_by(start, end, by):
    """seq(from, to, by) for by > 0."""
    if end < start:
        raise ValueError("wrong sign in 'by' argument")
    del_ = end - start
    if del_ == 0 and end == 0:
        return np.array([end], dtype=float)
    n = del_ / by
    dd = abs(del_) / max(abs(end), abs(start))
    if dd < 100 * DBL_EPSILON:
        return np.array([start], dtype=float)
    n = int(n + 1e-10)
    x = start + np.arange(n + 1) * by
    return np.minimum(x, end)


def seq_length(start, end, length_out):
    """seq(from, to, length.out = n)."""
    n = int(length_out)
    if n == 1:
        return np.array([start], dtype=float)
    by = (end - start) / (n - 1)
    x = start + np.arange(n) * by
    if n > 1:
        x[-1] = end
    return x


def find_interval(x, vec, rightmost_closed=False):
    """findInterval(x, vec) (1-based result, 0 below the first value)."""
    vec = np.asarray(vec, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    idx = np.searchsorted(vec, x, side="right")
    if rightmost_closed and vec.size:
        last = vec[-1]
        hit = (idx == vec.size) & (x == last)
        idx[hit] = vec.size - 1
    return idx


def bincode(x, breaks):
    """.bincode(x, breaks, right = TRUE, include.lowest = TRUE): 1-based, 0 = NA."""
    x = np.asarray(x, dtype=float)
    nb = len(breaks)
    out = np.zeros(x.size, dtype=int)
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        lo = 0
        hi = nb - 1
        if v < breaks[lo] or breaks[hi] < v or (v == breaks[lo] and False):
            continue
        while hi - lo >= 2:
            mid = (hi + lo) // 2
            if v > breaks[mid]:
                lo = mid
            else:
                hi = mid
        out[i] = lo + 1
    return out


# ---- interpolation --------------------------------------------------------------

def _regularize(x, y, ties):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x = x[ok]
    y = y[ok]
    if ties == "ordered":
        return x, y
    if np.any(np.diff(x) < 0):
        o = np.argsort(x, kind="mergesort")
        x = x[o]
        y = y[o]
    ux, first, inverse = np.unique(x, return_index=True, return_inverse=True)
    if ux.size < x.size:
        # groups ordered by first appearance; x is sorted so this is ux order
        yy = np.empty(ux.size)
        for g in range(ux.size):
            yy[g] = r_mean(y[inverse == g])
        return ux, yy
    return x, y


def approx(x, y, xout, ties="mean", rule=1):
    """stats::approx(x, y, xout, method = "linear", ties, rule)$y."""
    x, y = _regularize(x, y, ties)
    xout = np.atleast_1d(np.asarray(xout, dtype=float))
    n = x.size
    if n < 2:
        raise ValueError("need at least two non-NA values to interpolate")
    ylow = y[0] if rule == 2 else np.nan
    yhigh = y[-1] if rule == 2 else np.nan
    out = np.empty(xout.size)
    for k, v in enumerate(xout):
        if np.isnan(v):
            out[k] = np.nan
            continue
        i = 0
        j = n - 1
        if v < x[i]:
            out[k] = ylow
            continue
        if v > x[j]:
            out[k] = yhigh
            continue
        while i < j - 1:
            ij = (i + j) // 2
            if v < x[ij]:
                j = ij
            else:
                i = ij
        if v == x[j]:
            out[k] = y[j]
        elif v == x[i]:
            out[k] = y[i]
        else:
            out[k] = y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))
    return out


# ---- filters and smoothers ------------------------------------------------------

def conv_filter_centered(x, window):
    """stats::filter(x, rep(1 / window, window), sides = 2)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    f = np.full(window, 1 / window)
    o = window // 2
    out = np.full(n, np.nan)
    for i in range(n):
        if i + o - (window - 1) < 0 or i + o >= n:
            continue
        z = 0.0
        bad = False
        for j in range(window):
            tmp = x[i + o - j]
            if np.isnan(tmp):
                bad = True
                break
            z += f[j] * tmp
        if not bad:
            out[i] = z
    return out


def loess_direct(t, v, span, newdata):
    """loess(v ~ t, span, degree = 2, family = "gaussian", surface = "direct").

    Local quadratic fit at every point of `newdata` on the floor(n * span)
    nearest observations with tricube weights.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    n = t.size
    q = int(math.floor(n * span + 1e-5))
    if q <= 0:
        raise ValueError("span is too small")
    newdata = np.asarray(newdata, dtype=float)
    out = np.full(newdata.size, np.nan)
    for k, x0 in enumerate(newdata):
        if np.isnan(x0):
            continue
        dist = np.abs(t - x0)
        h = np.partition(dist, q - 1)[q - 1] * max(1.0, span)
        if h <= 0:
            out[k] = np.nan
            continue
        w = np.where(dist < h, (1 - (dist / h) ** 3) ** 3, 0.0)
        use = w > 0
        d = t[use] - x0
        X = np.column_stack([np.ones(d.size), d, d * d])
        sw = np.sqrt(w[use])
        beta = np.linalg.lstsq(X * sw[:, None], v[use] * sw, rcond=None)[0]
        out[k] = beta[0]
    return out


def dnorm(x, sd):
    return np.exp(-0.5 * (x / sd) ** 2) / (math.sqrt(2 * math.pi) * sd)


# ---- linear models --------------------------------------------------------------

def wls(X, y, w=None):
    """Weighted least squares; returns coefficients and weighted residuals."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    sw = np.ones(y.size) if w is None else np.sqrt(np.asarray(w, dtype=float))
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    resid = (y - X @ beta) * sw
    return beta, resid


# ---- optimisers -----------------------------------------------------------------

def optimize_brent(f, lower, upper, tol=DBL_EPSILON ** 0.25):
    """stats::optimize(f, c(lower, upper))$minimum (Brent's fmin, as in R)."""
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(DBL_EPSILON)
    a = lower
    b = upper
    v = a + c * (b - a)
    w = v
    x = v
    d = 0.0
    e = 0.0
    fx = f(x)
    fv = fx
    fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = 0.0
        q = 0.0
        r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def nelder_mead(fn, par, maxit=500, reltol=math.sqrt(DBL_EPSILON), abstol=-math.inf,
                alpha=1.0, beta=0.5, gamma=2.0):
    """stats::optim(par, fn, method = "Nelder-Mead") (R's nmmin).

    Returns (par, value, fail).
    """
    big = 1.0e35
    n = len(par)
    bvec = np.array(par, dtype=float)
    P = np.zeros((n + 1, n + 2))
    f = fn(bvec.copy())
    if not math.isfinite(f):
        raise ValueError("function cannot be evaluated at initial parameters")
    funcount = 1
    convtol = reltol * (abs(f) + reltol)
    n1 = n + 1
    C = n + 2
    P[n1 - 1, 0] = f
    P[:n, 0] = bvec
    L = 1
    size = 0.0
    step = 0.0
    for i in range(n):
        if 0.1 * abs(bvec[i]) > step:
            step = 0.1 * abs(bvec[i])
    if step == 0.0:
        step = 0.1
    for j in range(2, n1 + 1):
        P[:n, j - 1] = bvec
        trystep = step
        while P[j - 2, j - 1] == bvec[j - 2]:
            P[j - 2, j - 1] = bvec[j - 2] + trystep
            trystep *= 10
        size += trystep
    oldsize = size
    calcvert = True
    fail = 0
    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    bvec = P[:n, j].copy()
                    f = fn(bvec.copy())
                    if not math.isfinite(f):
                        f = big
                    funcount += 1
                    P[n1 - 1, j] = f
            calcvert = False
        VL = P[n1 - 1, L - 1]
        VH = VL
        H = L
        for j in range(1, n1 + 1):
            if j != L:
                f = P[n1 - 1, j - 1]
                if f < VL:
                    L = j
                    VL = f
                if f > VH:
                    H = j
                    VH = f
        if VH <= VL + convtol or VL <= abstol:
            break
        for i in range(n):
            temp = -P[i, H - 1]
            for j in range(n1):
                temp += P[i, j]
            P[i, C - 1] = temp / n
        bvec = (1.0 + alpha) * P[:n, C - 1] - alpha * P[:n, H - 1]
        f = fn(bvec.copy())
        if not math.isfinite(f):
            f = big
        funcount += 1
        VR = f
        if VR < VL:
            P[n1 - 1, C - 1] = f
            for i in range(n):
                f = gamma * bvec[i] + (1 - gamma) * P[i, C - 1]
                P[i, C - 1] = bvec[i]
                bvec[i] = f
            f = fn(bvec.copy())
            if not math.isfinite(f):
                f = big
            funcount += 1
            if f < VR:
                P[:n, H - 1] = bvec
                P[n1 - 1, H - 1] = f
            else:
                P[:n, H - 1] = P[:n, C - 1]
                P[n1 - 1, H - 1] = VR
        else:
            if VR < VH:
                P[:n, H - 1] = bvec
                P[n1 - 1, H - 1] = VR
            bvec = (1 - beta) * P[:n, H - 1] + beta * P[:n, C - 1]
            f = fn(bvec.copy())
            if not math.isfinite(f):
                f = big
            funcount += 1
            if f < P[n1 - 1, H - 1]:
                P[:n, H - 1] = bvec
                P[n1 - 1, H - 1] = f
            elif VR >= VH:
                calcvert = True
                size = 0.0
                for j in range(n1):
                    if j + 1 != L:
                        for i in range(n):
                            P[i, j] = beta * (P[i, j] - P[i, L - 1]) + P[i, L - 1]
                            size += abs(P[i, j] - P[i, L - 1])
                if size < oldsize:
                    oldsize = size
                else:
                    fail = 10
                    break
        if funcount > maxit:
            break
    if funcount > maxit:
        fail = 1
    return P[:n, L - 1].copy(), P[n1 - 1, L - 1], fail
