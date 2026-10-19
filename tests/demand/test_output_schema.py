import numpy as np
import pandas as pd

from src.common.schema import REQUIRED_COLUMNS, validate_candidate_table
from src.demand.reorder import build_reorder_plan, to_candidates

SERIES = ["CA_1_FOODS_1", "CA_1_FOODS_3", "TX_1_HOBBIES_1"]


def _candidates():
    history = pd.DataFrame(
        {
            "series_id": np.repeat(SERIES, 28),
            "date": np.tile(pd.date_range("2016-01-01", periods=28), len(SERIES)),
            "sales": np.repeat([10.0, 50.0, 1.0], 28),
        }
    )
    # TX_1_HOBBIES_1 is fully covered by starting inventory, so it carries zero value.
    forecast = pd.DataFrame(
        {"series_id": np.repeat(SERIES, 28), "forecast": np.repeat([20.0, 60.0, 0.5], 28)}
    )
    unit_price_usd = pd.Series([2.0, 3.0, 5.0], index=SERIES)
    return to_candidates(build_reorder_plan(forecast, history, unit_price_usd))


def test_candidates_have_exactly_the_shared_schema_columns():
    candidates = _candidates()

    assert list(candidates.columns) == REQUIRED_COLUMNS
    validate_candidate_table(candidates, expected_domain="demand")


def test_one_candidate_per_store_department():
    candidates = _candidates()

    assert sorted(candidates["candidate_id"]) == sorted(SERIES)
    assert (candidates["domain"] == "demand").all()


def test_net_value_is_benefit_minus_cost_and_values_non_negative():
    candidates = _candidates()

    assert (candidates[["expected_benefit_inr", "expected_cost_inr"]] >= 0).all().all()
    np.testing.assert_allclose(
        candidates["net_expected_value_inr"],
        candidates["expected_benefit_inr"] - candidates["expected_cost_inr"],
    )


def test_candidates_are_ranked_by_net_expected_value():
    assert _candidates()["net_expected_value_inr"].is_monotonic_decreasing
