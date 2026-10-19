"""End-to-end smoke tests: the pipeline runs and produces sensible, honest output."""

import json

import pytest

from src.fraud.train import parse_args, run_pipeline


def test_pipeline_produces_all_outputs(pipeline_run):
    metrics, out_dir = pipeline_run
    for name in ("fraud_candidates.csv", "fraud_metrics.json", "fraud_model.joblib"):
        assert (out_dir / name).exists(), name
    assert (out_dir / "fraud_calibration.png").exists()
    assert json.loads((out_dir / "fraud_metrics.json").read_text())["selected_model"] == (
        metrics["selected_model"]
    )


def test_every_requested_model_is_reported_with_resources(pipeline_run):
    metrics, _ = pipeline_run
    assert set(metrics["models"]) == {"logreg", "random_forest", "xgboost"}
    for report in metrics["models"].values():
        assert set(report["resources"]) == {
            "fit_seconds",
            "predict_ms_per_1k_rows",
            "model_size_kb",
        }


def test_models_learn_the_synthetic_signal(pipeline_run):
    metrics, _ = pipeline_run
    assert metrics["final"]["test"]["pr_auc"] > 0.5
    assert metrics["final"]["test"]["roc_auc"] > 0.8


def test_model_selection_uses_validation_only(pipeline_run):
    metrics, _ = pipeline_run
    best_on_val = max(metrics["models"], key=lambda n: metrics["models"][n]["val"]["pr_auc"])
    assert metrics["selected_model"] == best_on_val


def test_headline_numbers_report_fpr_and_recall_together(pipeline_run):
    metrics, _ = pipeline_run
    for key in ("test_at_threshold", "test_at_test_curve_threshold"):
        assert metrics["final"][key]["fpr"] is not None
        assert metrics["final"][key]["recall"] is not None
    assert "accuracy" not in json.dumps(metrics).lower()


def test_test_curve_threshold_respects_target_fpr(pipeline_run):
    metrics, _ = pipeline_run
    point = metrics["final"]["test_at_test_curve_threshold"]
    assert point["fpr"] <= metrics["settings"]["target_fpr"]


def test_calibrated_probabilities_match_observed_fraud_rate(pipeline_run):
    """Expected benefit sums P(fraud) * Amount, so total predicted fraud must be realistic.

    Calibration is NOT guaranteed to lower Brier on every dataset (when a model is already
    well calibrated and the validation window is small it can add noise), so we test the
    property the expected-value layer actually depends on instead.
    """
    metrics, _ = pipeline_run
    bins = metrics["calibration"]["bins_calibrated"]
    predicted_frauds = sum(m * c for m, c in zip(bins["mean_predicted"], bins["count"]))
    actual_frauds = metrics["dataset"]["test_fraud"]
    assert 0.5 <= predicted_frauds / actual_frauds <= 2.0


def test_unknown_model_is_rejected(synthetic_df, tmp_path):
    with pytest.raises(ValueError, match="Unknown models"):
        run_pipeline(synthetic_df, out_dir=tmp_path, model_names=("svm",), n_trials=0)


def test_isolation_forest_feature_mode_runs(synthetic_df, tmp_path):
    metrics = run_pipeline(
        synthetic_df,
        out_dir=tmp_path,
        model_names=("logreg",),
        n_trials=0,
        isolation_forest="feature",
        save_plot=False,
        save_model=False,
    )
    assert metrics["isolation_forest"]["used_as_feature"] is True
    assert metrics["isolation_forest"]["test_pr_auc"] > 0.0


def test_optuna_tuning_runs(synthetic_df, tmp_path):
    pytest.importorskip("optuna")
    metrics = run_pipeline(
        synthetic_df,
        out_dir=tmp_path,
        model_names=("xgboost",),
        n_trials=2,
        save_plot=False,
        save_model=False,
    )
    assert "max_depth" in metrics["settings"]["tuned_xgboost_params"]


def test_cli_defaults_match_guidelines():
    args = parse_args([])
    assert args.target_fpr == 0.05
    assert args.calibration == "sigmoid"
    assert args.trials >= 30
