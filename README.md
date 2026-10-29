# metrictrails (Python)

Python port of the R package [metrictrails](https://github.com/raimun2/metrictrails) (CRAN), from the [Metric Trails](https://metrictrails.com) lab: tools to read, clean and standardise GNSS streams recorded by sports watches, with a focus on trail and mountain running.

The R package is the source of truth. This port follows its CRAN releases (`R_VERSION`) and is tested against R on every function: the parity workflow runs the CRAN release on golden cases and requires this port to reproduce them (57 cases; differences are at floating-point level, see below).

```python
import metrictrails as mt

stream = mt.uniform_stream(mt.read_stream(mt.extdata("gpx_activity.gpx")))
stream = mt.sanitize_stream(stream)            # Kalman smoothing + red flags
stream = mt.resample_stream(stream)            # Nyquist resolution
stream = mt.ele_correction(stream, dem=mt.extdata("test_dem.tif"))
mt.summarise_stream(stream)
```

Function and argument names are those of the R package (`lambda` in `match_stream()` is `lambda_`, a reserved word in Python). Streams are pandas DataFrames; what R stores with `attr()` is in `DataFrame.attrs` (`sanitize_report`, `res`, `epsg`, `match_report`, `diagnosis`). Row numbers returned by `detect_stream()` and `match_stream()` are 1-based, as in R. DEM functions need `pip install metrictrails[dem]` (rasterio).

## Differences with R

* Results match R within 1e-6 (coordinates, distances, speeds) and usually within 1e-12.
* `resample_dem(method = "bilinear")`: terra computes in 32-bit floats; differences up to 0.1 mm.
* `read_stream()` parses GPX directly instead of going through GDAL; it has been checked on the package's example files only.
* `get_dem()` reprojects with GDAL through rasterio, like terra; it has not been compared with R yet (the tiles were not reachable from the test environment).

## Methods

See the R documentation and references: Sanchez and Villena (2020), Sanchez et al. (2024, 2025), Martinez, Sanchez and Diaz (2026).
