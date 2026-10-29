# Rules for this port

This repository is the Python port of the R package metrictrails. The R package on
CRAN is the source of truth; this code is a translation, not a reimplementation.

- Translate the R code line by line. Same function names, same argument names
  and defaults, same column names and column order in the outputs, same error
  and warning messages. Do not "improve" methods here: improvements go to R first.
- Reproduce R's numerics, not the closest library function. R's quantile type 7,
  approx() tie rules, findInterval(), stats::filter(), loess(surface = "direct"),
  optimize() and optim("Nelder-Mead") are reimplemented in the stats module for
  that reason; reuse them.
- sf::st_nearest_feature breaks exact ties through the GEOS STRtree. The strtree
  module reproduces GEOS 3.12 packing (libstdc++ std::sort, node capacity 10) and
  its priority queue. Use it for every 2D nearest point or nearest segment search.
- A change is done when the parity workflow passes: every golden case made by
  parity/make_golden.R (R repo) from the CRAN release matches this port. When R
  adds a function or an argument, also add golden cases to make_golden.R in the
  R repo.
- Ports follow CRAN releases only, never the R main branch. R_VERSION holds the
  R version this port reproduces; the port's own version is the same number.
- Australian English in documentation and comments.

Test: pytest -q tests/test_parity.py (with METRICTRAILS_GOLDEN=golden)
