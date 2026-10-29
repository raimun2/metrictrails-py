"""pytest entry point: every golden case must match R.

Set METRICTRAILS_GOLDEN to the folder written by make_golden.R.
"""

import json
import os

import pytest

from parity import run_case

GOLDEN = os.environ.get("METRICTRAILS_GOLDEN", os.path.join(os.path.dirname(__file__), "..", "golden"))
MANIFEST = os.path.join(GOLDEN, "manifest.json")
CASES = json.load(open(MANIFEST))["cases"] if os.path.exists(MANIFEST) else {}


@pytest.mark.skipif(not CASES, reason="no golden fixtures; run make_golden.R first")
@pytest.mark.parametrize("name", sorted(CASES))
def test_parity(name):
    ok, worst, msgs = run_case(name, CASES[name], GOLDEN)
    assert ok, "\n".join(msgs)
