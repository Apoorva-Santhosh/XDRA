import numpy as np
import pandas as pd
import pytest

from src.demand.baseline import moving_average_forecast, naive_forecast
from src.demand.preprocess import build_store_dept_series, load_m5, split_by_time
from src.demand.train import MODELS


@pytest.fixture
def split(m5_dir):
    data = load_m5(m5_dir, ("CA_1", "TX_1"))
    return split_by_time(build_store_dept_series(data.sales, data.calendar))


def test_naive_repeats_last_observed_value(split):
    history, holdout = split
    last = history.groupby("series_id")["sales"].last()

    forecast = naive_forecast(history, holdout.drop(columns="sales"))

    np.testing.assert_array_equal(forecast, holdout["series_id"].map(last))


def test_moving_average_uses_trailing_window(split):
    history, holdout = split
    level = history.groupby("series_id")["sales"].agg(lambda s: s.iloc[-7:].mean())

    forecast = moving_average_forecast(history, holdout.drop(columns="sales"), window=7)

    np.testing.assert_allclose(forecast, holdout["series_id"].map(level))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_forecaster_contract_and_no_future_leakage(split, name):
    history, holdout = split
    forecaster = MODELS[name]

    blind = forecaster(history, holdout.drop(columns="sales"))
    with_future_sales = forecaster(history, holdout.assign(sales=holdout["sales"] * 100))

    assert blind.index.equals(holdout.index)
    assert blind.notna().all() and (blind >= 0).all()
    pd.testing.assert_series_equal(blind, with_future_sales)
