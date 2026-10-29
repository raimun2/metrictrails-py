"""metrictrails: processing of GNSS streams from sports watches.

Python port of the R package metrictrails (CRAN), which is the source of
truth. Function names and arguments follow the R package; see the R
documentation for the methods and references.
"""

from ._version import __version__, R_VERSION
from .io import read_stream, read_stream_text, write_stream, write_stream_text
from .uniform import uniform_stream
from .sanitize import sanitize_stream
from .resample import resolution_stream, resample_stream
from .measures import differential_stream, summarise_stream
from .fractal import fractal_stream
from .tobler import tobler_stream, tobler_speed
from .two_streams import (compare_stream, detect_stream, project_stream, locate_stream,
                          match_stream)
from .dem import Raster, read_raster, write_raster, get_dem, resample_dem, ele_correction
from .datasets import extdata, strava_streams, strava_metadata

__all__ = [
    "read_stream", "read_stream_text", "write_stream", "write_stream_text", "uniform_stream",
    "sanitize_stream", "resolution_stream", "resample_stream", "differential_stream",
    "summarise_stream", "fractal_stream", "tobler_stream", "tobler_speed", "compare_stream",
    "detect_stream", "project_stream", "locate_stream", "match_stream", "Raster", "read_raster",
    "write_raster", "get_dem", "resample_dem", "ele_correction", "extdata", "strava_streams",
    "strava_metadata", "__version__", "R_VERSION",
]
