"""Lag, rolling-demand, calendar and store/department features for the global demand model."""

from __future__ import annotations

import pandas as pd

LAGS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)

CATEGORICAL_FEATURES = ["store_id", "dept_id"]
CALENDAR_FEATURES = ["wday", "snap", "is_event"]
LAG_FEATURES = [f"lag_{lag}" for lag in LAGS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + CALENDAR_FEATURES + LAG_FEATURES


def add_features(series: pd.DataFrame) -> pd.DataFrame:
    """Add `FEATURE_COLUMNS` to a long store-department frame.

    Input: frame with `preprocess.SERIES_COLUMNS` (future rows may carry NaN `sales`).
    Output: copy sorted by (series_id, date), original index labels kept, with lag and rolling
        features added and store/department cast to categoricals. Every lag and rolling value at
        date t is built from sales at t-7 or earlier within the same series.
    """
    df = series.sort_values(["series_id", "date"]).copy()
    by_series = df.groupby("series_id")["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = by_series.shift(lag)

    recent = by_series.shift(min(LAGS)).groupby(df["series_id"])
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = recent.transform(lambda s, w=window: s.rolling(w).mean())

    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype("category")
    return df
