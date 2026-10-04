import pandas as pd
import pytest

from src.demand.reorder import ReorderAssumptions, build_reorder_plan

SERIES = "CA_1_FOODS_1"
ASSUMPTIONS = ReorderAssumptions(
    starting_inventory_days=14,
    trailing_window_days=28,
    gross_margin=0.3,
    annual_holding_rate=0.25,
    usd_to_inr=83.0,
)


def _plan(history_sales, daily_forecast, unit_price_usd=None):
    history = pd.DataFrame(
        {
            "series_id": SERIES,
            "date": pd.date_range("2016-01-01", periods=len(history_sales)),
            "sales": history_sales,
        }
    )
    forecast = pd.DataFrame({"series_id": SERIES, "forecast": [daily_forecast] * 28})
    if unit_price_usd is None:
        unit_price_usd = pd.Series({SERIES: 2.0})
    return build_reorder_plan(forecast, history, unit_price_usd, ASSUMPTIONS).iloc[0]


def test_reorder_economics_match_hand_calculation():
    # Only the trailing 28 days (10/day) set the starting inventory, not the older 1000/day.
    plan = _plan([1000.0] * 28 + [10.0] * 28, daily_forecast=12.0)

    unit_price_inr = 2.0 * 83
    holding_per_unit_day = unit_price_inr * (1 - 0.3) * 0.25 / 365
    assert plan["starting_inventory"] == pytest.approx(140)
    assert plan["forecast_demand"] == pytest.approx(336)
    assert plan["expected_shortage"] == pytest.approx(196)
    assert plan["reorder_qty"] == 196
    assert plan["expected_benefit_inr"] == pytest.approx(196 * unit_price_inr * 0.3)
    assert plan["expected_cost_inr"] == pytest.approx(196 / 2 * 28 * holding_per_unit_day)
    assert plan["net_expected_value_inr"] == pytest.approx(
        plan["expected_benefit_inr"] - plan["expected_cost_inr"]
    )


def test_reorder_quantity_rounds_up_to_whole_units():
    plan = _plan([10.0] * 28, daily_forecast=10.01)

    assert plan["expected_shortage"] == pytest.approx(140.28)
    assert plan["reorder_qty"] == 141


def test_no_reorder_when_starting_inventory_covers_forecast():
    plan = _plan([10.0] * 28, daily_forecast=4.0)

    assert plan["reorder_qty"] == 0
    assert plan["expected_benefit_inr"] == 0
    assert plan["expected_cost_inr"] == 0
    assert plan["net_expected_value_inr"] == 0


def test_missing_unit_price_is_rejected():
    with pytest.raises(ValueError, match=SERIES):
        _plan([10.0] * 28, daily_forecast=12.0, unit_price_usd=pd.Series(dtype=float))
