"""Digital Elevation Models (port of R/dem.R).

The R package uses terra (GDAL). Here a DEM is a small `Raster` object
(values, extent, resolution, CRS) so that the arithmetic that matters for the
results (bilinear extraction, refinement and aggregation) is written out in
full and shared with the JavaScript port. GDAL, through rasterio, is only
used to read GeoTIFF files and, in get_dem(), to reproject the downloaded
tiles, as terra does.
"""

import math
import os
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass

import numpy as np

from . import _rstats as rs
from ._utils import check_columns, segment_id, to_lonlat, to_metric, utm_epsg, _project

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"


@dataclass
class Raster:
    """Single-band raster: `values[row, col]`, row 0 at the top (north)."""

    values: np.ndarray
    xmin: float
    ymax: float
    xres: float
    yres: float
    crs: str

    @property
    def nrow(self):
        return self.values.shape[0]

    @property
    def ncol(self):
        return self.values.shape[1]

    @property
    def xmax(self):
        return self.xmin + self.ncol * self.xres

    @property
    def ymin(self):
        return self.ymax - self.nrow * self.yres

    def is_lonlat(self):
        from pyproj import CRS
        return CRS.from_user_input(self.crs).is_geographic


def read_raster(path):
    """Read the first band of a raster file (needs rasterio)."""
    import rasterio

    with rasterio.open(path) as src:
        v = src.read(1).astype(float)
        if src.nodata is not None and not math.isnan(src.nodata):
            v[v == src.nodata] = np.nan
        t = src.transform
        if t.b != 0 or t.d != 0:
            raise ValueError("Rotated rasters are not supported.")
        crs = src.crs.to_string() if src.crs is not None else ""
        epsg = src.crs.to_epsg() if src.crs is not None else None
        if epsg is not None:
            crs = f"EPSG:{epsg}"
        return Raster(v, t.c, t.f, t.a, -t.e, crs)


def write_raster(dem, path):
    """Write a Raster to GeoTIFF (needs rasterio)."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=dem.nrow, width=dem.ncol, count=1,
                       dtype="float64", crs=dem.crs, nodata=np.nan,
                       transform=from_origin(dem.xmin, dem.ymax, dem.xres, dem.yres)) as dst:
        dst.write(dem.values, 1)
    return path


def _as_raster(dem):
    if isinstance(dem, Raster):
        return dem
    if isinstance(dem, (str, os.PathLike)):
        return read_raster(dem)
    raise TypeError("'dem' must be a Raster or a path to a raster file.")


# ---- extraction -------------------------------------------------------------------

def extract_simple(dem, x, y):
    """Value of the cell containing every point (terra 'simple')."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.size, np.nan)
    col = np.floor((x - dem.xmin) / dem.xres)
    row = np.floor((dem.ymax - y) / dem.yres)
    col = np.where(x == dem.xmax, dem.ncol - 1, col)
    row = np.where(y == dem.ymin, dem.nrow - 1, row)
    ok = (col >= 0) & (col < dem.ncol) & (row >= 0) & (row < dem.nrow)
    out[ok] = dem.values[row[ok].astype(int), col[ok].astype(int)]
    return out


def _lerp(a, b, w):
    out = a + w * (b - a)
    out = np.where(np.isnan(a), b, out)
    out = np.where(np.isnan(b), a, out)
    return out


def extract_bilinear(dem, x, y):
    """Bilinear interpolation between the four nearest cell centres (terra 'bilinear').

    Inside the outer half cell the nearest centres are used (no extrapolation);
    a missing cell is replaced by its neighbour along the interpolation axis.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.size, np.nan)
    inside = (x >= dem.xmin) & (x <= dem.xmax) & (y >= dem.ymin) & (y <= dem.ymax)
    if not inside.any():
        return out
    cf = np.clip((x[inside] - dem.xmin) / dem.xres - 0.5, 0, dem.ncol - 1)
    rf = np.clip((dem.ymax - y[inside]) / dem.yres - 0.5, 0, dem.nrow - 1)
    c0 = np.minimum(np.floor(cf).astype(int), max(dem.ncol - 2, 0))
    r0 = np.minimum(np.floor(rf).astype(int), max(dem.nrow - 2, 0))
    c1 = np.minimum(c0 + 1, dem.ncol - 1)
    r1 = np.minimum(r0 + 1, dem.nrow - 1)
    wc = cf - c0
    wr = rf - r0
    v = dem.values
    top = _lerp(v[r0, c0], v[r0, c1], wc)
    bottom = _lerp(v[r1, c0], v[r1, c1], wc)
    out[inside] = _lerp(top, bottom, wr)
    return out


# ---- resampling -------------------------------------------------------------------

def _warp_same_crs(dem, xmin, ymax, res, nrow, ncol, method):
    """GDAL warp (bilinear or nearest) of `dem` to a grid in the same CRS."""
    xs = xmin + (np.arange(ncol) + 0.5) * res
    ys = ymax - (np.arange(nrow) + 0.5) * res
    px = (xs - dem.xmin) / dem.xres  # source pixel coordinates (0 = left edge)
    py = (dem.ymax - ys) / dem.yres
    return _warp_pixels(dem, px[None, :].repeat(nrow, 0), py[:, None].repeat(ncol, 1), method,
                        dem.xres / res, dem.yres / res)


def _warp_pixels(dem, px, py, method, xscale, yscale):
    v = dem.values
    nr, nc = v.shape
    out = np.full(px.shape, np.nan)
    inside = (px >= 0) & (px < nc) & (py >= 0) & (py < nr)
    if method == "near":
        ix = np.floor(px + 1e-10).astype(int)
        iy = np.floor(py + 1e-10).astype(int)
        ok = inside & (ix >= 0) & (ix < nc) & (iy >= 0) & (iy < nr)
        out[ok] = v[iy[ok], ix[ok]]
        return out
    # bilinear; when the destination is coarser the kernel widens (as in GDAL)
    xs = min(xscale, 1.0)
    ys = min(yscale, 1.0)
    rx = 1.0 / xs
    ry = 1.0 / ys
    pxi = px[inside]
    pyi = py[inside]
    acc = np.zeros(pxi.size)
    wsum = np.zeros(pxi.size)
    x0 = np.floor(pxi - 0.5 - rx + 1).astype(int)
    y0 = np.floor(pyi - 0.5 - ry + 1).astype(int)
    nx = int(math.ceil(2 * rx)) + 1
    ny = int(math.ceil(2 * ry)) + 1
    for dy in range(ny):
        iy = y0 + dy
        wy = 1 - np.abs((iy + 0.5 - pyi) * ys)
        for dx in range(nx):
            ix = x0 + dx
            wx = 1 - np.abs((ix + 0.5 - pxi) * xs)
            w = np.where((wx > 0) & (wy > 0), wx * wy, 0.0)
            ok = (w > 0) & (ix >= 0) & (ix < nc) & (iy >= 0) & (iy < nr)
            val = np.full(pxi.size, np.nan)
            val[ok] = v[iy[ok], ix[ok]]
            ok &= ~np.isnan(val)
            acc[ok] += w[ok] * val[ok]
            wsum[ok] += w[ok]
    res = np.full(pxi.size, np.nan)
    good = wsum >= 1e-6
    res[good] = acc[good] / wsum[good]
    out[inside] = res
    return out


def _aggregate(dem, fact, fun):
    nr = math.ceil(dem.nrow / fact)
    nc = math.ceil(dem.ncol / fact)
    out = np.full((nr, nc), np.nan)
    f = {"mean": rs.r_mean, "median": rs.r_median, "min": np.min, "max": np.max}[fun]
    for i in range(nr):
        for j in range(nc):
            block = dem.values[i * fact:(i + 1) * fact, j * fact:(j + 1) * fact].ravel()
            block = block[~np.isnan(block)]
            if block.size:
                out[i, j] = f(block)
    return Raster(out, dem.xmin, dem.ymax, dem.xres * fact, dem.yres * fact, dem.crs)


def resample_dem(dem, res, method="bilinear"):
    """Refine (bilinear, near) or coarsen (mean, median, min, max) a DEM in a metric CRS."""
    if method not in ("bilinear", "near", "mean", "median", "min", "max"):
        raise ValueError("'method' must be one of bilinear, near, mean, median, min, max")
    dem = _as_raster(dem)
    if dem.is_lonlat():
        raise ValueError("The DEM must be in a metric CRS; project it first.")
    if method in ("bilinear", "near"):
        nc = math.ceil((dem.xmax - dem.xmin) / res)
        nr = math.ceil((dem.ymax - dem.ymin) / res)
        values = _warp_same_crs(dem, dem.xmin, dem.ymax, res, nr, nc, method)
        return Raster(values, dem.xmin, dem.ymax, float(res), float(res), dem.crs)
    fact = round(res / dem.xres)
    if fact < 2:
        raise ValueError("Aggregation needs 'res' to be at least twice the current resolution "
                         f"({float(f'{dem.xres:.3g}'):g} m).")
    return _aggregate(dem, int(fact), method)


# ---- download -----------------------------------------------------------------------

def _tile_x(lon, z):
    return np.floor((lon + 180) / 360 * 2 ** z)


def _tile_y(lat, z):
    r = lat * math.pi / 180
    return np.floor((1 - np.log(np.tan(r) + 1 / np.cos(r)) / math.pi) / 2 * 2 ** z)


def dem_tiles(data, z=14, buffer=500):
    """Slippy-map tiles (x, y) that intersect a corridor of `buffer` metres around the path."""
    epsg = utm_epsg(data["lon"], data["lat"])
    xy = to_metric(data["lon"], data["lat"], epsg)
    ok = ~np.isnan(xy).any(axis=1)
    seg = segment_id(data)[ok]
    xy = xy[ok]
    max_lat = np.nanmax(np.abs(np.asarray(data["lat"], dtype=float)))
    tile_m = 40075016.7 * math.cos(max_lat * math.pi / 180) / 2 ** z
    spacing = min(buffer, tile_m / 2)
    px, py = [], []
    seen = []
    for sg in seg:
        if sg not in seen:
            seen.append(sg)
    for sg in seen:
        rows = np.where(seg == sg)[0]
        s_along = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(xy[rows, 0]) ** 2
                                                           + np.diff(xy[rows, 1]) ** 2))])
        grid = rs.seq_by(0.0, s_along.max(), max(spacing, 1))
        s_new = np.array(list(dict.fromkeys(np.concatenate([grid, [s_along.max()]]).tolist())))
        if rows.size == 1:
            px.append(xy[rows, 0])
            py.append(xy[rows, 1])
        else:
            px.append(rs.approx(s_along, xy[rows, 0], s_new, ties="mean"))
            py.append(rs.approx(s_along, xy[rows, 1], s_new, ties="mean"))
    px = np.concatenate(px)
    py = np.concatenate(py)
    offsets = rs.seq_length(-buffer, buffer, max(2, math.ceil(2 * buffer / (tile_m / 2)) + 1))
    tiles = []
    seen_t = set()
    # expand.grid(dx, dy): dx varies fastest
    for dyo in offsets:
        for dxo in offsets:
            ll = to_lonlat(px + dxo, py + dyo, epsg)
            for tx, ty in zip(_tile_x(ll[:, 0], z), _tile_y(ll[:, 1], z)):
                key = (int(tx), int(ty))
                if key not in seen_t:
                    seen_t.add(key)
                    tiles.append(key)
    return tiles, epsg, xy


def get_dem(data, z=14, res=None, method="bilinear", buffer=500, max_tiles=math.inf, cache_dir=None):
    """Download Terrain Tiles (AWS) around the stream, mosaic and project them to its UTM zone.

    Needs rasterio. Returns a `Raster`.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    check_columns(data, ["lon", "lat"], "get_dem")
    if z < 1 or z > 14:
        raise ValueError("'z' must be between 1 and 14.")
    tiles, epsg, xy = dem_tiles(data, z=z, buffer=buffer)
    box = (np.nanmin(xy[:, 0]) - buffer, np.nanmin(xy[:, 1]) - buffer,
           np.nanmax(xy[:, 0]) + buffer, np.nanmax(xy[:, 1]) + buffer)
    if len(tiles) > max_tiles:
        raise ValueError(f"{len(tiles)} tiles needed, more than 'max_tiles'.")
    if len(tiles) > 50:
        print(f"Downloading {len(tiles)} terrain tiles at zoom {z}.")
    cache_dir = cache_dir or tempfile.gettempdir()
    files = []
    for tx, ty in tiles:
        url = TILE_URL.format(z=z, x=tx, y=ty)
        dest = os.path.join(cache_dir, f"metrictrails_dem_{z}_{tx}_{ty}.tif")
        attempt = 0
        while not os.path.exists(dest):
            attempt += 1
            if attempt > 3:
                raise RuntimeError(f"Could not download a valid tile from {url}")
            part = dest + ".part"
            try:
                urllib.request.urlretrieve(url, part)
                with rasterio.open(part) as src:
                    src.read(1)
                os.replace(part, dest)
            except Exception:
                if os.path.exists(part):
                    os.remove(part)
        files.append(dest)

    srcs = [rasterio.open(f) for f in files]
    try:
        mosaic, transform = merge(srcs)
        src_crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    dst_crs = f"EPSG:{epsg}"
    h, w = mosaic.shape[1:]
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    dst_t, dw, dh = calculate_default_transform(src_crs, dst_crs, w, h, left, bottom, right, top)
    dst = np.full((dh, dw), np.nan)
    reproject(mosaic[0].astype(float), dst, src_transform=transform, src_crs=src_crs,
              dst_transform=dst_t, dst_crs=dst_crs, resampling=Resampling.bilinear,
              src_nodata=None, dst_nodata=np.nan)
    full = Raster(dst, dst_t.c, dst_t.f, dst_t.a, -dst_t.e, dst_crs)
    dem = crop(full, box)
    if res is not None:
        dem = resample_dem(dem, res, method=method)
    return dem


def crop(dem, box):
    """terra::crop to (xmin, ymin, xmax, ymax), snapped outwards to the cell grid."""
    xmin, ymin, xmax, ymax = box
    c0 = max(0, int(math.floor((xmin - dem.xmin) / dem.xres + 1e-9)))
    c1 = min(dem.ncol, int(math.ceil((xmax - dem.xmin) / dem.xres - 1e-9)))
    r0 = max(0, int(math.floor((dem.ymax - ymax) / dem.yres + 1e-9)))
    r1 = min(dem.nrow, int(math.ceil((dem.ymax - ymin) / dem.yres - 1e-9)))
    return Raster(dem.values[r0:r1, c0:c1].copy(), dem.xmin + c0 * dem.xres,
                  dem.ymax - r0 * dem.yres, dem.xres, dem.yres, dem.crs)


def ele_correction(data, dem=None, z=14, extract="bilinear", replace=True):
    """Replace `ele` with DEM values (or add them as `ele_dem`)."""
    if extract not in ("bilinear", "simple"):
        raise ValueError("'extract' must be 'bilinear' or 'simple'")
    check_columns(data, ["lon", "lat"], "ele_correction")
    if dem is None:
        dem = get_dem(data, z=z)
    dem = _as_raster(dem)
    pts = _project(data["lon"], data["lat"], "EPSG:4326", dem.crs)
    f = extract_bilinear if extract == "bilinear" else extract_simple
    ele = f(dem, pts[:, 0], pts[:, 1])
    if np.isnan(ele).any():
        warnings.warn(f"{int(np.isnan(ele).sum())} readings fall outside the DEM and get NA elevation.")
    data = data.copy()
    if replace:
        data["ele"] = ele
    else:
        data["ele_dem"] = ele
    return data
