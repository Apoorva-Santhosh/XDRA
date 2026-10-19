import numpy as np
import pandas as pd
import pytest

from src.churn.expected_value import (
    CALL_COST_INR,
    DISCOUNT_COST_INR,
    P_SAVE_CALL,
    P_SAVE_DISCOUNT,
    REMAINING_TENURE_MONTHS,
    SCHEMA_COLUMNS,
    USD_TO_INR,
    compute_candidates,
)


def _raw(n=8):
    contracts = ["Month-to-month", "One year", "Two year", "Month-to-month"]
    return pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "MonthlyCharges": np.linspace(20, 110, n),
            "Contract": [contracts[i % 4] for i in range(n)],
        }
    )


def _probs(n=8):
    return np.linspace(0.05, 0.95, n)


def test_schema_columns_exact():
    out = compute_candidates(_raw(), _probs())
    assert list(out.columns) == SCHEMA_COLUMNS


def test_domain_and_ids():
    out = compute_candidates(_raw(), _probs())
    assert (out["domain"] == "churn").all()
    assert out["candidate_id"].is_unique
    assert out["candidate_id"].str.startswith("churn_").all()


def test_net_equals_benefit_minus_cost():
    out = compute_candidates(_raw(), _probs())
    expected = out["expected_benefit_inr"] - out["expected_cost_inr"]
    assert np.allclose(out["net_expected_value_inr"], expected, atol=0.011)


def test_no_nulls_and_costs_are_valid_tiers():
    out = compute_candidates(_raw(), _probs())
    assert not out.isna().any().any()
    assert set(out["expected_cost_inr"]).issubset({CALL_COST_INR, DISCOUNT_COST_INR})


def test_both_interventions_are_used():
    out = compute_candidates(_raw(), _probs())
    assert (out["expected_cost_inr"] == CALL_COST_INR).any()
    assert (out["expected_cost_inr"] == DISCOUNT_COST_INR).any()


def test_discount_chosen_only_when_it_nets_more():
    raw, probs = _raw(), _probs()
    out = compute_candidates(raw, probs)
    months = raw["Contract"].map(REMAINING_TENURE_MONTHS)
    at_risk = raw["MonthlyCharges"] * USD_TO_INR * months * probs
    net_call = at_risk * P_SAVE_CALL - CALL_COST_INR
    net_disc = at_risk * P_SAVE_DISCOUNT - DISCOUNT_COST_INR
    assert np.allclose(out["net_expected_value_inr"], np.maximum(net_call, net_disc), atol=0.011)


def test_zero_probability_gives_zero_benefit_and_call_only():
    out = compute_candidates(_raw(), np.zeros(8))
    assert (out["expected_benefit_inr"] == 0).all()
    assert (out["expected_cost_inr"] == CALL_COST_INR).all()
    assert (out["net_expected_value_inr"] == -CALL_COST_INR).all()


def test_benefit_increases_with_probability():
    raw = _raw()
    low = compute_candidates(raw, np.full(8, 0.1))
    high = compute_candidates(raw, np.full(8, 0.6))
    assert (high["expected_benefit_inr"] > low["expected_benefit_inr"]).all()


def test_unknown_contract_raises():
    raw = _raw()
    raw.loc[0, "Contract"] = "Weekly"
    with pytest.raises(ValueError):
        compute_candidates(raw, _probs())


def test_invalid_probability_raises():
    probs = _probs()
    probs[0] = 1.5
    with pytest.raises(ValueError):
        compute_candidates(_raw(), probs)
