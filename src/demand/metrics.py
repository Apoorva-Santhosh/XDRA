"""Forecast accuracy metrics required for demand by GUIDELINES.md §5: WAPE, MAE, RMSE and bias."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_arrays(actual: ArrayLike, forecast: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}.")
    return actual, forecast


def _total_actual(actual: np.ndarray) -> float:
    total = np.abs(actual).sum()
    if total == 0:
        raise ValueError("Undefined when every actual value is zero.")
    return total


def wape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Weighted absolute percentage error: sum(|actual - forecast|) / sum(|actual|)."""
    actual, forecast = _as_arrays(actual, forecast)
    return float(np.abs(actual - forecast).sum() / _total_actual(actual))


def mae(actual: ArrayLike, forecast: ArrayLike) -> float:
    actual, forecast = _as_arrays(actual, forecast)
    return float(np.abs(actual - forecast).mean())


def rmse(actual: ArrayLike, forecast: ArrayLike) -> float:
    actual, forecast = _as_arrays(actual, forecast)
    return float(np.sqrt(((actual - forecast) ** 2).mean()))


def bias(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Signed relative bias: sum(forecast - actual) / sum(|actual|); positive = over-forecast."""
    actual, forecast = _as_arrays(actual, forecast)
    return float((forecast - actual).sum() / _total_actual(actual))


def forecast_metrics(actual: ArrayLike, forecast: ArrayLike) -> dict[str, float]:
    """All four demand metrics for one forecast, keyed by metric name."""
    return {
        "wape": wape(actual, forecast),
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "bias": bias(actual, forecast),
    }
