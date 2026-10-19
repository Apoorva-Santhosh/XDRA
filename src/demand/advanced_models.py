"""Holt-Winters and global LightGBM demand forecasters (signature documented in `baseline.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.demand.features import FEATURE_COLUMNS, LAG_FEATURES, add_features

SEASONAL_PERIOD = 7

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


def holt_winters_forecast(history: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
    """Per-series additive Holt-Winters with damped trend and weekly seasonality."""
    forecasts = []
    for series_id, target in future.groupby("series_id")["date"]:
        y = history.loc[history["series_id"] == series_id].set_index("date")["sales"].asfreq("D")
        fitted = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD,
            initialization_method="estimated",
        ).fit()
        path = fitted.forecast((target.max() - y.index[-1]).days)
        forecasts.append(pd.Series(path.reindex(target).to_numpy(), index=target.index))
    return pd.concat(forecasts).reindex(future.index).clip(lower=0).rename("forecast")


def lightgbm_forecast(history: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
    """One LightGBM model across all series, forecast recursively one day at a time.

    Lags that fall inside the horizon are filled from earlier predictions, so no actual sales
    from `future` are ever used.
    """
    frame = pd.concat([history, future.assign(sales=np.nan)], ignore_index=True)
    future_labels = frame.index[len(history) :]

    features = add_features(frame)
    fit_rows = features["sales"].notna() & features[LAG_FEATURES].notna().all(axis=1)
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(features.loc[fit_rows, FEATURE_COLUMNS], features.loc[fit_rows, "sales"])

    for date in np.sort(frame.loc[future_labels, "date"].unique()):
        step = features.index[features["date"] == date]
        frame.loc[step, "sales"] = np.clip(
            model.predict(features.loc[step, FEATURE_COLUMNS]), 0, None
        )
        features = add_features(frame)

    return pd.Series(
        frame.loc[future_labels, "sales"].to_numpy(), index=future.index, name="forecast"
    )
