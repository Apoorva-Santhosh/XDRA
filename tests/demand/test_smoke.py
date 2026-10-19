import pandas as pd

from src.common.schema import REQUIRED_COLUMNS, validate_candidate_table
from src.demand.train import MODELS, run


def test_pipeline_runs_end_to_end_and_writes_valid_outputs(m5_dir, tmp_path):
    output_dir = tmp_path / "outputs"

    candidates = run(data_dir=m5_dir, output_dir=output_dir, stores=("CA_1", "TX_1"))

    written = pd.read_csv(output_dir / "demand_candidates.csv")
    assert list(written.columns) == REQUIRED_COLUMNS
    validate_candidate_table(written, expected_domain="demand")
    assert len(written) == len(candidates) == 4

    comparison = pd.read_csv(output_dir / "demand_model_comparison.csv")
    assert set(comparison["window"]) == {"validation", "test"}
    assert set(comparison["model"]) == set(MODELS)
    assert comparison.loc[comparison["selected"], "model"].nunique() == 1

    plan = pd.read_csv(output_dir / "demand_reorder_plan.csv")
    assert set(plan["candidate_id"]) == set(written["candidate_id"])
