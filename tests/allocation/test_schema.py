import pandas as pd
import pytest

from src.common.schema import SchemaValidationError, validate_candidate_table


def _valid_df():
    return pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "domain": ["churn", "churn"],
            "expected_benefit_inr": [1000.0, 500.0],
            "expected_cost_inr": [200.0, 100.0],
            "net_expected_value_inr": [800.0, 400.0],
        }
    )


def test_valid_table_passes():
    validate_candidate_table(_valid_df(), expected_domain="churn")


def test_missing_column_fails():
    df = _valid_df().drop(columns=["expected_cost_inr"])
    with pytest.raises(SchemaValidationError):
        validate_candidate_table(df, expected_domain="churn")


def test_duplicate_candidate_id_fails():
    df = _valid_df()
    df.loc[1, "candidate_id"] = "c1"
    with pytest.raises(SchemaValidationError):
        validate_candidate_table(df, expected_domain="churn")


def test_wrong_domain_fails():
    with pytest.raises(SchemaValidationError):
        validate_candidate_table(_valid_df(), expected_domain="fraud")


def test_negative_cost_fails():
    df = _valid_df()
    df.loc[0, "expected_cost_inr"] = -5.0
    with pytest.raises(SchemaValidationError):
        validate_candidate_table(df, expected_domain="churn")


def test_net_value_mismatch_fails():
    df = _valid_df()
    df.loc[0, "net_expected_value_inr"] = 999999.0
    with pytest.raises(SchemaValidationError):
        validate_candidate_table(df, expected_domain="churn")


def test_unknown_domain_fails():
    with pytest.raises(SchemaValidationError):
        validate_candidate_table(_valid_df(), expected_domain="not_a_domain")
