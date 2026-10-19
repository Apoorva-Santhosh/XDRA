"""Naive and moving-average baseline forecasts, reported alongside every demand model (§5).

Every forecaster in the demand pipeline shares one signature:
    forecaster(history, future) -> pd.Series
where `history` is a long frame of past `series_id, date, sales` rows, `future` holds the
`series_id, date` rows to forecast (all later than history), and the returned Series of
non-negative unit forecasts is aligned to `future.index`.
"""

from __future__ import annotations

import pandas as pd

MOVING_AVERAGE_WINDOW = 28


def naive_forecast(history: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
    """Repeat each series' last observed daily sales across the whole horizon."""
    last = history.sort_values("date").groupby("series_id")["sales"].last()
    return future["series_id"].map(last).rename("forecast")


def moving_average_forecast(
    history: pd.DataFrame, future: pd.DataFrame, window: int = MOVING_AVERAGE_WINDOW
) -> pd.Series:
    """Repeat each series' mean daily sales over its last `window` days across the horizon."""
    recent = history.sort_values("date").groupby("series_id").tail(window)
    level = recent.groupby("series_id")["sales"].mean()
    return future["series_id"].map(level).rename("forecast")
