"""Example files and datasets shipped with the R package."""

import os

import pandas as pd

_HERE = os.path.join(os.path.dirname(__file__), "extdata")


def extdata(name):
    """Path of an example file (as system.file("extdata", name, package = "metrictrails"))."""
    path = os.path.join(_HERE, name)
    if not os.path.exists(path):
        raise FileNotFoundError(name)
    return path


def strava_streams():
    """GNSS streams of two activities from the Strava API (see ?strava_streams in R)."""
    df = pd.read_csv(extdata("strava_streams.csv"))
    df["moving"] = df["moving"].astype(bool)
    for col in ("cadence", "heartrate", "temp"):
        df[col] = df[col].astype("Int64")
    return df


def strava_metadata():
    """Metadata of the activities in strava_streams."""
    return pd.read_csv(extdata("strava_metadata.csv"))
