"""The pipeline's output must satisfy the shared schema (GUIDELINES.md section 4)."""

import pandas as pd

from src.common.schema import REQUIRED_COLUMNS, validate_candidate_table


def test_candidates_file_matches_shared_schema(pipeline_run):
    _, out_dir = pipeline_run
    table = pd.read_csv(out_dir / "fraud_candidates.csv")
    assert list(table.columns) == REQUIRED_COLUMNS
    validate_candidate_table(table, expected_domain="fraud")


def test_candidates_cover_exactly_the_test_window(pipeline_run):
    metrics, out_dir = pipeline_run
    table = pd.read_csv(out_dir / "fraud_candidates.csv")
    assert len(table) == metrics["dataset"]["test_rows"]
    assert table["candidate_id"].is_unique


def test_no_candidate_has_negative_benefit_or_cost(pipeline_run):
    _, out_dir = pipeline_run
    table = pd.read_csv(out_dir / "fraud_candidates.csv")
    assert (table["expected_benefit_inr"] >= 0).all()
    assert (table["expected_cost_inr"] >= 0).all()
