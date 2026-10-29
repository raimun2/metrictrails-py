"""Parity of the Python port with the R package.

Every case in the golden directory (made by parity/make_golden.R from the
CRAN release) is run with the Python port and compared with R's output.
"""

import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

import metrictrails as mt
from metrictrails.dem import read_raster
from metrictrails._utils import datetime_to_seconds, seconds_to_datetime

# absolute + relative tolerance for doubles; logical, integer and text are exact
ATOL = 1e-6
RTOL = 1e-9


def read_table(path):
    types = json.load(open(path.replace(".csv", ".types.json")))
    df = pd.read_csv(path, keep_default_na=True, na_values=["NA"], dtype=str)
    out = {}
    for col in df.columns:
        kind = types.get(col, "double")
        v = df[col]
        if kind in ("double", "integer", "timestamp"):
            num = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
            if kind == "timestamp":
                out[col] = seconds_to_datetime(num).to_numpy()
            elif kind == "integer" and not np.isnan(num).any():
                out[col] = num.astype(np.int64)
            else:
                out[col] = num
        elif kind == "logical":
            out[col] = v.map({"TRUE": True, "FALSE": False}).to_numpy()
        else:
            out[col] = v.to_numpy(dtype=object)
    return pd.DataFrame(out), types


def as_numeric(series):
    s = pd.Series(series)
    if pd.api.types.is_datetime64_any_dtype(s):
        return datetime_to_seconds(s)
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy().astype(float)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def compare_values(name, got, want, kind, atol=None):
    """Return (ok, worst absolute error, message)."""
    if kind in ("character", "logical"):
        g = pd.Series(got).astype(object).where(pd.notna(pd.Series(got)), None).tolist()
        w = pd.Series(want).astype(object).where(pd.notna(pd.Series(want)), None).tolist()
        if kind == "logical":
            g = [None if x is None else bool(x) for x in g]
            w = [None if x is None else bool(x) for x in w]
        else:
            g = [None if x is None else str(x) for x in g]
            w = [None if x is None else str(x) for x in w]
        bad = [i for i, (a, b) in enumerate(zip(g, w)) if a != b]
        if len(g) != len(w):
            return False, math.inf, f"{name}: length {len(g)} vs {len(w)}"
        return (not bad), 0.0, (f"{name}: {len(bad)} mismatches, first at {bad[0]}: {g[bad[0]]!r} vs {w[bad[0]]!r}"
                                if bad else "")
    g = as_numeric(got)
    w = as_numeric(want)
    if g.size != w.size:
        return False, math.inf, f"{name}: length {g.size} vs {w.size}"
    na_g = np.isnan(g)
    na_w = np.isnan(w)
    if (na_g != na_w).any():
        i = int(np.where(na_g != na_w)[0][0])
        return False, math.inf, f"{name}: NA pattern differs at {i}: {g[i]} vs {w[i]}"
    inf_g = np.isinf(g)
    if (g[inf_g] != w[inf_g]).any():
        return False, math.inf, f"{name}: Inf differs"
    fin = ~na_g & ~inf_g
    err = np.abs(g[fin] - w[fin])
    worst = float(err.max()) if err.size else 0.0
    atol = ATOL if atol is None else atol
    ok = bool(np.all(err <= atol + RTOL * np.abs(w[fin])))
    if not ok:
        i = int(np.argmax(err - (atol + RTOL * np.abs(w[fin]))))
        return False, worst, f"{name}: max abs error {worst:.3g} (got {g[fin][i]!r}, want {w[fin][i]!r})"
    return True, worst, ""


def compare_tables(got, want, types):
    msgs = []
    worst = 0.0
    if list(got.columns) != list(want.columns):
        msgs.append(f"columns differ:\n  got  {list(got.columns)}\n  want {list(want.columns)}")
    if len(got) != len(want):
        msgs.append(f"rows {len(got)} vs {len(want)}")
        return False, worst, msgs
    for col in want.columns:
        if col not in got.columns:
            continue
        ok, w, msg = compare_values(col, got[col], want[col], types.get(col, "double"))
        worst = max(worst, w)
        if not ok:
            msgs.append(msg)
    return not msgs, worst, msgs


def compare_json(got, want, path="attrs"):
    msgs = []
    if isinstance(want, dict):
        if isinstance(got, pd.DataFrame):
            got = {c: got[c].tolist() for c in got.columns}
        for k, v in want.items():
            if isinstance(got, dict) and k in got:
                msgs += compare_json(got[k], v, f"{path}.{k}")
            else:
                msgs.append(f"{path}.{k} missing")
        return msgs
    if isinstance(want, list):
        if isinstance(got, dict):
            got = list(got.values())
        if isinstance(got, pd.DataFrame):
            got = got.to_dict("records")
        if isinstance(got, (np.ndarray, pd.Series)):
            got = list(got)
        if not isinstance(got, list) or len(got) != len(want):
            return [f"{path}: length differs ({got!r} vs {want!r})"]
        for i, (a, b) in enumerate(zip(got, want)):
            msgs += compare_json(a, b, f"{path}[{i}]")
        return msgs
    if want is None:
        if got is None or (isinstance(got, float) and math.isnan(got)):
            return []
        return [f"{path}: {got!r} vs NA"]
    if isinstance(want, bool) or isinstance(want, str):
        return [] if got == want else [f"{path}: {got!r} vs {want!r}"]
    try:
        g = float(got)
    except (TypeError, ValueError):
        return [f"{path}: {got!r} vs {want!r}"]
    if math.isnan(g) or abs(g - want) > ATOL + RTOL * abs(want):
        return [f"{path}: {got!r} vs {want!r}"]
    return []


def run_case(name, spec, golden):
    d = os.path.join(golden, name)
    inputs = {}
    for nm, rel in (spec["inputs"] or {}).items():
        inputs[nm] = read_table(os.path.join(golden, rel))[0]
    args = dict(spec["args"]) if isinstance(spec["args"], dict) else {}
    fun = spec["fun"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fun == "read_stream":
            result = mt.read_stream(mt.extdata(args.pop("file")), **args)
        elif fun == "write_stream":
            text = mt.write_stream_text(inputs["data"], **args)
            want = open(os.path.join(d, "output.gpx"), encoding="utf-8").read().replace("\r\n", "\n")
            return (text == want), 0.0, ([] if text == want else ["GPX text differs"])
        elif fun in ("ele_correction", "resample_dem"):
            args["dem"] = read_raster(mt.extdata(args["dem"]))
            if fun == "resample_dem":
                r = mt.resample_dem(**args)
                want = json.load(open(os.path.join(d, "output.json")))
                msgs = []
                for k in ("xmin", "xmax", "ymin", "ymax", "nrow", "ncol"):
                    if abs(getattr(r, k) - want[k]) > 1e-6:
                        msgs.append(f"{k}: {getattr(r, k)} vs {want[k]}")
                if not msgs:
                    w = np.array([np.nan if v is None else v for v in want["values"]], dtype=float)
                    # terra warps in 32-bit floats: allow 1 mm on bilinear refinement
                    tol = 1e-3 if args.get("method", "bilinear") == "bilinear" else None
                    ok, worst, m = compare_values("values", r.values.ravel(), w, "double", tol)
                    if not ok:
                        msgs.append(m)
                    return not msgs, worst, msgs
                return False, math.inf, msgs
            result = mt.ele_correction(inputs["data"], **args)
        elif fun == "tobler_speed":
            result = pd.DataFrame({"speed": mt.tobler_speed(inputs["data"]["slope"], args["params"])})
        elif fun == "compare_stream":
            result = mt.compare_stream(inputs["a"], inputs["b"], **args)
        elif fun == "detect_stream":
            result = mt.detect_stream(inputs["data"], inputs["route"], **args)
        elif fun == "locate_stream":
            result = mt.locate_stream(inputs["data"], inputs["reference"], **args)
        elif fun == "match_stream":
            if "lambda" in args:
                args["lambda_"] = args.pop("lambda")
            result = mt.match_stream(inputs["data"], inputs["reference"], inputs["params"], **args)
        elif fun == "fractal_stream":
            fd = mt.fractal_stream(inputs["data"], **args)
            result = fd["scaling"]
            result.attrs = {"dimension": fd["dimension"], "r_squared": fd["r_squared"]}
        elif fun == "sanitize_stream" and "tobler" in args:
            result = mt.sanitize_stream(inputs["data"], **args)
        else:
            result = getattr(mt, fun)(inputs["data"], **args)

    want, types = read_table(os.path.join(d, "output.csv"))
    ok, worst, msgs = compare_tables(result, want, types)
    attrs_want = json.load(open(os.path.join(d, "attrs.json")))
    if attrs_want:
        got_attrs = result.attrs
        if fun == "sanitize_stream":
            got_attrs = result.attrs.get("sanitize_report", {})
        msgs += compare_json(got_attrs, attrs_want)
    return not msgs, worst, msgs


def main(golden, only=None):
    manifest = json.load(open(os.path.join(golden, "manifest.json")))
    failures = 0
    for name, spec in manifest["cases"].items():
        if only and not any(o in name for o in only):
            continue
        try:
            ok, worst, msgs = run_case(name, spec, golden)
        except Exception as e:  # report and continue
            ok, worst, msgs = False, math.inf, [f"{type(e).__name__}: {e}"]
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} max|err|={worst:.2e}")
        for m in msgs[:6]:
            print("      ", m)
    print(f"\n{len(manifest['cases']) if not only else ''} cases, {failures} failing")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2:]) else 0)
