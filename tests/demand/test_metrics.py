import math

import pytest

from src.demand.metrics import bias, forecast_metrics, mae, rmse, wape

ACTUAL = [10.0, 20.0, 30.0]
FORECAST = [12.0, 18.0, 33.0]


def test_metric_values():
    assert wape(ACTUAL, FORECAST) == pytest.approx(7 / 60)
    assert mae(ACTUAL, FORECAST) == pytest.approx(7 / 3)
    assert rmse(ACTUAL, FORECAST) == pytest.approx(math.sqrt(17 / 3))
    assert bias(ACTUAL, FORECAST) == pytest.approx(3 / 60)


def test_bias_is_negative_when_under_forecasting():
    assert bias(ACTUAL, [5.0, 10.0, 15.0]) == pytest.approx(-0.5)


def test_forecast_metrics_reports_all_metrics():
    assert set(forecast_metrics(ACTUAL, FORECAST)) == {"wape", "mae", "rmse", "bias"}


def test_relative_metrics_reject_all_zero_actuals():
    with pytest.raises(ValueError):
        wape([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        bias([0.0, 0.0], [1.0, 1.0])


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])
