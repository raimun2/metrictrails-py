"""Reading and writing GPX files (port of R/read_stream.R and R/write_stream.R).

The R package reads GPX through GDAL (sf::st_read). This port parses the GPX
text directly, reproducing what the GDAL GPX driver returns for track and
route points: coordinates, `ele`, `time`, and one segment per track segment
(or per route). Point extensions are parsed with the same regular
expressions as the R code.
"""

import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ._utils import check_columns, segment_id, seconds_to_datetime

_ATTR = re.compile(r"""([A-Za-z_:][A-Za-z0-9_.:-]*)\s*=\s*("([^"]*)"|'([^']*)')""")
_RESERVED = {"lon", "lat", "ele", "timestamp", "time", "segment", "lng", "longitude",
             "latitude", "elevation", "altitude", "alt", "datetime"}


def _strip_comments(text):
    return re.sub(r"(?s)<!--.*?-->", "", text)


def _local(tag):
    return tag.split(":")[-1]


def _attrs(start_tag):
    out = {}
    for m in _ATTR.finditer(start_tag):
        out[_local(m.group(1))] = m.group(3) if m.group(3) is not None else m.group(4)
    return out


def _child_text(block, name):
    m = re.search(r"(?s)<(?:[A-Za-z0-9_]+:)?" + name + r"(?:\s[^>]*)?>(.*?)</(?:[A-Za-z0-9_]+:)?"
                  + name + r"\s*>", block)
    if m is None:
        return None
    return m.group(1).strip()


def _parse_time(text):
    """ISO 8601 date-time to seconds since the epoch (UTC); None if invalid."""
    if text is None or text == "":
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(\.\d+)?"
                 r"(Z|[+-]\d{2}:?\d{2})?$", text.strip())
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    frac = float(m.group(7)) if m.group(7) else 0.0
    secs = datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp() + frac
    tz = m.group(8)
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        digits = tz[1:].replace(":", "")
        secs -= sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
    return secs


def _points(text, container, seg_tag, point_tag):
    """Points of every <container> (trk or rte), with their segment key."""
    rows = []
    cont_re = re.compile(r"(?s)<(?:[A-Za-z0-9_]+:)?" + container + r"[\s>].*?</(?:[A-Za-z0-9_]+:)?"
                         + container + r"\s*>")
    pt_re = re.compile(r"(?s)<(?:[A-Za-z0-9_]+:)?" + point_tag + r"(\s[^>]*?)?(/>|>(.*?)</(?:[A-Za-z0-9_]+:)?"
                       + point_tag + r"\s*>)")
    for fid, cm in enumerate(cont_re.finditer(text)):
        block = cm.group(0)
        if seg_tag is None:
            groups = [block]
        else:
            seg_re = re.compile(r"(?s)<(?:[A-Za-z0-9_]+:)?" + seg_tag + r"[\s>].*?</(?:[A-Za-z0-9_]+:)?"
                                + seg_tag + r"\s*>")
            groups = [m.group(0) for m in seg_re.finditer(block)]
        for sid, g in enumerate(groups):
            for pm in pt_re.finditer(g):
                attrs = _attrs(pm.group(1) or "")
                inner = pm.group(3) or ""
                # nested <extensions> may contain tags called ele or time
                core = re.sub(r"(?s)<extensions>.*?</extensions>", "", inner)
                ele = _child_text(core, "ele")
                tim = _child_text(core, "time")
                rows.append((float(attrs.get("lon", "nan")), float(attrs.get("lat", "nan")),
                             _num(ele), _parse_time(tim), f"{fid} {sid}"))
    return rows


def _num(s):
    if s is None:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_stream(file, type="auto"):
    """Read a GNSS stream from a GPX file.

    Returns a DataFrame with `lon`, `lat` and, when available, `ele`,
    `timestamp` (UTC), `segment` and point extension variables.
    """
    if type not in ("auto", "track", "route"):
        raise ValueError("'type' must be one of 'auto', 'track', 'route'")
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    if not re.search(r"\.gpx$", str(file), flags=re.IGNORECASE):
        raise ValueError("read_stream() only reads GPX files.")
    with open(file, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    return read_stream_text(raw, type=type, name=os.path.basename(file))


def read_stream_text(raw, type="auto", name="input"):
    """read_stream() on the text of a GPX file."""
    text = "\n".join(raw.splitlines())
    clean = _strip_comments(text)
    track = _points(clean, "trk", "trkseg", "trkpt")
    route = _points(clean, "rte", None, "rtept")
    if type == "auto":
        layer = "track" if len(track) > 0 else "route"
    else:
        layer = type
    rows = track if layer == "track" else route
    if len(rows) == 0:
        raise ValueError(f"No {layer} points found in {name} (waypoint-only files are not streams).")

    lon = np.array([r[0] for r in rows], dtype=float)
    lat = np.array([r[1] for r in rows], dtype=float)
    ele = np.array([r[2] for r in rows], dtype=float)
    tim = [r[3] for r in rows]
    stream = pd.DataFrame({"lon": lon, "lat": lat})
    if np.any(~np.isnan(ele)):
        stream["ele"] = ele
    if any(t is not None for t in tim):
        secs = np.array([np.nan if t is None else t for t in tim], dtype=float)
        stream["timestamp"] = seconds_to_datetime(secs).to_numpy()
    keys = [r[4] for r in rows]
    codes, _ = pd.factorize(pd.Series(keys), sort=False)
    stream["segment"] = codes.astype(int) + 1

    if "<extensions>" in text:
        tag = "trkpt" if layer == "track" else "rtept"
        blocks = re.findall(r"(?s)<" + tag + r"[ >].*?(?:</" + tag + r">|/>)", text)
        if len(blocks) == len(stream):
            ext = []
            for b in blocks:
                if "<extensions>" in b:
                    ext.append(re.sub(r"(?s)^.*?<extensions>(.*)</extensions>.*$", r"\1", b, count=1))
                else:
                    ext.append("")
            leaf_re = re.compile(r"<([A-Za-z0-9_]+:)?[A-Za-z0-9_]+>[^<]*</")
            names = []
            for e in ext:
                for m in leaf_re.finditer(e):
                    leaf = re.sub(r"^<([A-Za-z0-9_]+:)?([A-Za-z0-9_]+)>.*$", r"\2", m.group(0),
                                  flags=re.DOTALL)
                    if leaf not in names:
                        names.append(leaf)
            for leaf in names:
                pattern = re.compile(r"<([A-Za-z0-9_]+:)?" + leaf + r">([^<]*)</")
                value = np.full(len(ext), np.nan)
                for i, e in enumerate(ext):
                    m = pattern.search(e)
                    if m is not None:
                        value[i] = _num(m.group(2).strip())
                col = leaf
                if col.lower() in _RESERVED or col in stream.columns:
                    col = "ext_" + leaf
                if np.any(~np.isnan(value)) and col not in stream.columns:
                    stream[col] = value
        else:
            import warnings
            warnings.warn("Point extensions could not be aligned with the points and were skipped.")
    return stream


def _fmt_fixed(x, digits):
    return "%.*f" % (digits, x)


def write_stream(data, file, name=None):
    """Write a stream as a GPX 1.1 track; returns `file`."""
    check_columns(data, ["lon", "lat"], "write_stream")
    text = write_stream_text(data, name=name)
    with open(file, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return file


def write_stream_text(data, name=None):
    """GPX text that write_stream() would write."""
    check_columns(data, ["lon", "lat"], "write_stream")
    seg = segment_id(data)
    lat = np.asarray(data["lat"], dtype=float)
    lon = np.asarray(data["lon"], dtype=float)
    has_ele = "ele" in data.columns
    has_time = "timestamp" in data.columns
    if has_ele:
        ele = np.asarray(data["ele"], dtype=float)
    if has_time:
        ts = pd.to_datetime(data["timestamp"], utc=True)
    pts = []
    for i in range(len(data)):
        p = '<trkpt lat="' + _fmt_fixed(lat[i], 8) + '" lon="' + _fmt_fixed(lon[i], 8) + '">'
        if has_ele and not np.isnan(ele[i]):
            p += "<ele>" + _fmt_fixed(ele[i], 2) + "</ele>"
        if has_time and not pd.isna(ts.iloc[i]):
            p += "<time>" + ts.iloc[i].floor("s").strftime("%Y-%m-%dT%H:%M:%SZ") + "</time>"
        pts.append("      " + p + "</trkpt>")
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="metrictrails" xmlns="http://www.topografix.com/GPX/1/1">',
           "  <trk>"]
    if name is not None:
        out.append("    <name>" + str(name) + "</name>")
    for s in pd.unique(seg):
        out.append("    <trkseg>")
        out.extend(p for p, g in zip(pts, seg) if g == s)
        out.append("    </trkseg>")
    out += ["  </trk>", "</gpx>"]
    return "\n".join(out) + "\n"
