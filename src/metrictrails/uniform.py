"""Naming convention (port of R/uniform_stream.R)."""

import re
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ._utils import as_frame, check_columns, datetime_to_seconds, ensure_float, seconds_to_datetime

DICTIONARY = {
    "lon": ["lon", "lng", "long", "longitude", "position_long", "position_lon"],
    "lat": ["lat", "latitude", "position_lat"],
    "ele": ["ele", "elevation", "altitude", "alt", "enhanced_altitude", "elev"],
    "timestamp": ["timestamp", "datetime", "date_time"],
}

# formats tried by as.POSIXct() on character input, in order; the first one
# that parses the first non-missing value is used for every value, and
# trailing characters are ignored (as strptime does)
_FORMATS = [
    (r"^(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d*)?)", 6),
    (r"^(\d{4})/(\d{1,2})/(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d*)?)", 6),
    (r"^(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{1,2})", 5),
    (r"^(\d{4})/(\d{1,2})/(\d{1,2}) (\d{1,2}):(\d{1,2})", 5),
    (r"^(\d{4})-(\d{1,2})-(\d{1,2})", 3),
    (r"^(\d{4})/(\d{1,2})/(\d{1,2})", 3),
]


def _parse_with(fmt, s):
    m = re.match(fmt[0], s)
    if m is None:
        return None
    parts = list(m.groups())
    y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
    h = int(parts[3]) if fmt[1] >= 5 else 0
    mi = int(parts[4]) if fmt[1] >= 5 else 0
    sec = float(parts[5]) if fmt[1] == 6 else 0.0
    try:
        base = datetime(y, mo, d, h, mi, 0, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None
    return base + sec


def _character_to_seconds(values):
    texts = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
             for v in values]
    texts = [None if t is None else re.sub(r"Z$", "", t).replace("T", " ") for t in texts]
    first = next((t for t in texts if t is not None), None)
    if first is None:
        return np.full(len(texts), np.nan)
    fmt = next((f for f in _FORMATS if _parse_with(f, first) is not None), None)
    if fmt is None:
        raise ValueError("character string is not in a standard unambiguous format")
    out = np.array([np.nan if t is None else (_parse_with(fmt, t) or np.nan) for t in texts],
                   dtype=float)
    return out


def _to_seconds(v):
    """Seconds since the epoch (UTC) of a date-time column."""
    s = pd.Series(v)
    if pd.api.types.is_datetime64_any_dtype(s):
        return datetime_to_seconds(s)
    return _character_to_seconds(list(s))


def _is_datetime_like(v):
    s = pd.Series(v)
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return True
    return False


def uniform_stream(data):
    """Rename the columns of a stream to the metrictrails naming convention.

    `lon`, `lat`, `ele`, `timestamp` (UTC) and `time` (seconds since the first
    reading) become the first columns; other variables are kept unchanged.
    """
    data = as_frame(data)
    names = list(data.columns)
    lower = [str(c).lower() for c in names]
    for target, options in DICTIONARY.items():
        hits = [i for i, c in enumerate(lower) if c in options]
        if len(hits) > 1:
            raise ValueError(f"Several columns could be '{target}': "
                             + ", ".join(str(names[i]) for i in hits)
                             + ". Remove or rename all but one.")
        if len(hits) == 1:
            names[hits[0]] = target
    data.columns = names

    hit = [i for i, c in enumerate(names) if str(c).lower() == "time"]
    if len(hit) == 1:
        names[hit[0]] = "time"
        data.columns = names
        if _is_datetime_like(data["time"]):
            if "timestamp" in names:
                raise ValueError("Both 'timestamp' and a date-time 'time' column are present.")
            names[hit[0]] = "timestamp"
            data.columns = names

    check_columns(data, ["lon", "lat"], "uniform_stream")
    data["lon"] = ensure_float(data["lon"])
    data["lat"] = ensure_float(data["lat"])
    if "ele" in data.columns:
        data["ele"] = ensure_float(data["ele"])

    if "timestamp" in data.columns:
        secs = _to_seconds(data["timestamp"])
        data["timestamp"] = seconds_to_datetime(secs).to_numpy()
        data["time"] = secs - np.nanmin(secs) if np.any(~np.isnan(secs)) else secs
    elif "time" in data.columns:
        data["time"] = ensure_float(data["time"])

    if "time" in data.columns and len(data) > 2:
        d = np.diff(np.asarray(data["time"], dtype=float))
        d = d[~np.isnan(d)]
        increasing = np.mean(d > 0) if d.size else np.nan
        if np.isnan(increasing) or increasing < 0.5:
            warnings.warn("Time increases in less than half of the readings; it is removed and the "
                          "stream is treated as a route without time.")
            data = data.drop(columns=[c for c in ("time", "timestamp") if c in data.columns])

    first = [c for c in ("lon", "lat", "ele", "timestamp", "time") if c in data.columns]
    return data[first + [c for c in data.columns if c not in first]]
