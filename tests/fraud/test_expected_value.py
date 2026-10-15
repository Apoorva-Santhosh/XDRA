import numpy as np
import pytest

from src.common.schema import SchemaValidationError, validate_candidate_table
from src.fraud.expected_value import build_candidate_table, write_candidate_table


def test_expected_value_arithmetic():
    table = build_candidate_table(
        row_ids=[10, 11, 12],
        amounts=[1000.0, 50.0, 0.0],
        probs=[0.5, 0.2, 0.9],
        investigation_cost_inr=200.0,
        amount_to_inr=1.0,
    )
    assert table["expected_benefit_inr"].tolist() == pytest.approx([500.0, 10.0, 0.0])
    assert table["expected_cost_inr"].tolist() == [200.0, 200.0, 200.0]
    assert table["net_expected_value_inr"].tolist() == pytest.approx([300.0, -190.0, -200.0])


def test_amount_conversion_scales_benefit_only():
    table = build_candidate_table([1], [100.0], [0.5], 200.0, amount_to_inr=90.0)
    assert table["expected_benefit_inr"].iloc[0] == pytest.approx(4500.0)
    assert table["expected_cost_inr"].iloc[0] == 200.0


def test_probabilities_are_clipped():
    table = build_candidate_table([1, 2], [100.0, 100.0], [1.7, -0.3])
    assert table["expected_benefit_inr"].tolist() == [100.0, 0.0]


def test_candidate_ids_are_unique_and_domain_is_fraud():
    table = build_candidate_table(np.arange(50), np.ones(50), np.full(50, 0.1))
    assert table["candidate_id"].is_unique
    assert set(table["domain"]) == {"fraud"}
    validate_candidate_table(table, expected_domain="fraud")


def test_positive_only_drops_unprofitable_candidates():
    table = build_candidate_table(
        [1, 2], [1000.0, 10.0], [0.5, 0.5], investigation_cost_inr=200.0, positive_only=True
    )
    assert table["candidate_id"].tolist() == ["fraud_1"]


def test_writer_refuses_a_table_that_breaks_the_schema(tmp_path):
    table = build_candidate_table([1, 2], [100.0, 100.0], [0.5, 0.5])
    table.loc[0, "net_expected_value_inr"] = 12345.0
    with pytest.raises(SchemaValidationError):
        write_candidate_table(table, tmp_path)
    assert not (tmp_path / "fraud_candidates.csv").exists()
