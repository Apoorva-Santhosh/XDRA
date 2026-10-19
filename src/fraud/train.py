"""End-to-end fraud pipeline: load -> time split -> train -> calibrate -> candidates.

Run from the repo root:

    python -m src.fraud.train                      # full run, 30 Optuna trials
    python -m src.fraud.train --trials 0           # skip tuning (fast)
    python -m src.fraud.train --models logreg xgboost --positive-only

Outputs (all under outputs/, which is gitignored):
    fraud_candidates.csv   candidate table in the shared schema (GUIDELINES.md section 4)
    fraud_metrics.json     every metric, threshold and resource profile from the run
    fraud_model.joblib     the selected model with its calibrator and threshold
    fraud_calibration.png  reliability diagram, raw vs calibrated

Selection, calibration and the operating threshold all use the validation window. The
test window is only ever scored, never fitted to.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.fraud import config
from src.fraud.calibration import (
    METHODS,
    Calibrator,
    calibration_bins,
    save_calibration_plot,
)
from src.fraud.evaluate import (
    brier,
    evaluate_scores,
    operating_point,
    pr_auc,
    roc_auc,
    threshold_for_fpr,
)
from src.fraud.expected_value import build_candidate_table, write_candidate_table
from src.fraud.models import (
    AVAILABLE_MODELS,
    class_ratio,
    fit_and_profile,
    fit_isolation_forest,
    isolation_scores,
    make_model,
    tune_xgboost,
)
from src.fraud.preprocess import (
    add_isolation_feature,
    build_features,
    check_dataset_integrity,
    load_raw,
    time_split,
    validate_raw_columns,
)

logger = logging.getLogger(__name__)


def _jsonable(obj):
    """Recursively convert numpy types and NaN/inf so the structure is valid JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_pipeline(
    df: pd.DataFrame,
    out_dir: str | Path = config.OUTPUT_DIR,
    model_names=config.DEFAULT_MODELS,
    n_trials: int = config.DEFAULT_TRIALS,
    calibration: str = "sigmoid",
    target_fpr: float = config.TARGET_FPR,
    seed: int = config.SEED,
    positive_only: bool = False,
    isolation_forest: str = "off",
    save_model: bool = True,
    save_plot: bool = True,
) -> dict:
    """Run the full fraud pipeline on a raw transaction frame.

    Args:
        df: raw creditcard data (Time, V1..V28, Amount, Class).
        out_dir: where outputs are written.
        model_names: models to train, from ``AVAILABLE_MODELS``.
        n_trials: Optuna trials for XGBoost tuning; 0 uses default parameters.
        calibration: ``"sigmoid"``, ``"isotonic"`` or ``"none"``.
        target_fpr: false-positive-rate ceiling for the operating threshold.
        seed: random seed.
        positive_only: write only candidates with positive net expected value.
        isolation_forest: ``"off"``; ``"check"`` reports an unsupervised sanity-check
            score; ``"feature"`` also feeds the anomaly score to the supervised models.
        save_model: persist the selected model bundle.
        save_plot: write the calibration diagram.

    Returns:
        The metrics dictionary that is also written to ``fraud_metrics.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = [m for m in model_names if m not in AVAILABLE_MODELS]
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Choose from {AVAILABLE_MODELS}.")
    if isolation_forest not in ("off", "check", "feature"):
        raise ValueError("isolation_forest must be 'off', 'check' or 'feature'.")

    validate_raw_columns(df)
    warnings = check_dataset_integrity(df)
    for message in warnings:
        logger.warning(message)

    split = time_split(df)
    fs = build_features(split)
    ratio = class_ratio(fs.y_train)
    logger.info("Training class ratio: %.1f genuine per fraud", ratio)

    iso_report = None
    if isolation_forest != "off":
        iso = fit_isolation_forest(fs.X_train, seed)
        iso_report = {
            "val_pr_auc": pr_auc(fs.y_val, isolation_scores(iso, fs.X_val)),
            "test_pr_auc": pr_auc(fs.y_test, isolation_scores(iso, fs.X_test)),
            "test_roc_auc": roc_auc(fs.y_test, isolation_scores(iso, fs.X_test)),
            "used_as_feature": isolation_forest == "feature",
        }
        if isolation_forest == "feature":
            fs = add_isolation_feature(fs, iso)

    xgb_params: dict = {}
    if "xgboost" in model_names:
        xgb_params = tune_xgboost(fs.X_train, fs.y_train, fs.X_val, fs.y_val, n_trials, seed)

    fitted, val_scores, test_scores, model_report = {}, {}, {}, {}
    for name in model_names:
        logger.info("Training %s ...", name)
        model = make_model(name, seed, ratio, xgb_params if name == "xgboost" else None)
        model, s_val, profile = fit_and_profile(model, fs.X_train, fs.y_train, fs.X_val)
        s_test = model.predict_proba(fs.X_test)[:, 1]
        fitted[name], val_scores[name], test_scores[name] = model, s_val, s_test
        model_report[name] = {
            **evaluate_scores(fs.y_val, s_val, fs.y_test, s_test, target_fpr),
            "resources": profile,
        }
        logger.info(
            "%s: val PR-AUC %.4f | test PR-AUC %.4f",
            name,
            model_report[name]["val"]["pr_auc"],
            model_report[name]["test"]["pr_auc"],
        )

    # Selection uses validation PR-AUC only, so the test window plays no part in the choice.
    selected = max(model_names, key=lambda n: model_report[n]["val"]["pr_auc"])
    logger.info("Selected model: %s", selected)

    calibrator = Calibrator(calibration).fit(val_scores[selected], fs.y_val)
    p_val = calibrator.predict(val_scores[selected])
    p_test = calibrator.predict(test_scores[selected])
    final = evaluate_scores(fs.y_val, p_val, fs.y_test, p_test, target_fpr)

    test_curve_threshold = threshold_for_fpr(fs.y_test, p_test, target_fpr)
    bins_raw = calibration_bins(fs.y_test, test_scores[selected])
    bins_cal = calibration_bins(fs.y_test, p_test)

    candidates = build_candidate_table(
        fs.row_id_test, fs.amount_test, p_test, positive_only=positive_only
    )
    candidates_path = write_candidate_table(candidates, out_dir)

    paths = {"candidates": candidates_path}
    if save_plot:
        paths["calibration_plot"] = out_dir / "fraud_calibration.png"
        save_calibration_plot(bins_raw, bins_cal, paths["calibration_plot"])
    if save_model:
        paths["model"] = out_dir / "fraud_model.joblib"
        joblib.dump(
            {
                "model": fitted[selected],
                "model_name": selected,
                "calibrator": calibrator,
                "threshold": final["threshold_from_validation"],
                "target_fpr": target_fpr,
                "scaler": fs.scaler,
                "feature_columns": list(fs.X_train.columns),
            },
            paths["model"],
        )

    metrics = {
        "dataset": {
            "rows": len(df),
            "warnings": warnings,
            "train_rows": len(fs.X_train),
            "val_rows": len(fs.X_val),
            "test_rows": len(fs.X_test),
            "train_fraud": int(fs.y_train.sum()),
            "val_fraud": int(fs.y_val.sum()),
            "test_fraud": int(fs.y_test.sum()),
            "class_ratio_train": ratio,
        },
        "settings": {
            "seed": seed,
            "target_fpr": target_fpr,
            "calibration": calibration,
            "n_trials": n_trials,
            "isolation_forest": isolation_forest,
            "investigation_cost_inr": config.INVESTIGATION_COST_INR,
            "amount_to_inr": config.AMOUNT_TO_INR,
            "tuned_xgboost_params": xgb_params,
        },
        "models": model_report,
        "selected_model": selected,
        "calibration": {
            "method": calibration,
            "test_brier_raw": brier(fs.y_test, np.clip(test_scores[selected], 0, 1)),
            "test_brier_calibrated": brier(fs.y_test, p_test),
            "bins_raw": bins_raw,
            "bins_calibrated": bins_cal,
        },
        "final": {
            **final,
            "test_curve_threshold": test_curve_threshold,
            "test_at_test_curve_threshold": operating_point(
                fs.y_test, p_test, test_curve_threshold
            ),
        },
        "isolation_forest": iso_report,
        "candidates": {
            "rows": len(candidates),
            "positive_net_rows": int((candidates["net_expected_value_inr"] > 0).sum()),
            "total_positive_net_inr": float(
                candidates.loc[
                    candidates["net_expected_value_inr"] > 0, "net_expected_value_inr"
                ].sum()
            ),
        },
        "paths": paths,
    }
    metrics = _jsonable(metrics)
    (out_dir / "fraud_metrics.json").write_text(json.dumps(metrics, indent=2))
    _log_summary(metrics)
    return metrics


def _log_summary(metrics: dict) -> None:
    """Print the headline numbers, always reporting FPR and recall together."""
    final = metrics["final"]
    at_val = final["test_at_threshold"]
    at_test = final["test_at_test_curve_threshold"]
    logger.info("=== Fraud pipeline summary ===")
    logger.info("Selected model: %s", metrics["selected_model"])
    logger.info(
        "Test PR-AUC %.4f | ROC-AUC %.4f",
        final["test"]["pr_auc"],
        final["test"]["roc_auc"],
    )
    logger.info(
        "Threshold chosen on validation (target FPR %.1f%%): test FPR %.2f%% | recall %.2f%%",
        100 * metrics["settings"]["target_fpr"],
        100 * (at_val["fpr"] or 0.0),
        100 * (at_val["recall"] or 0.0),
    )
    logger.info(
        "Threshold chosen on test curve (reference only): test FPR %.2f%% | recall %.2f%%",
        100 * (at_test["fpr"] or 0.0),
        100 * (at_test["recall"] or 0.0),
    )
    logger.info(
        "Brier (test): raw %.5f -> calibrated %.5f",
        metrics["calibration"]["test_brier_raw"],
        metrics["calibration"]["test_brier_calibrated"],
    )
    logger.info("Candidates written: %d", metrics["candidates"]["rows"])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the X-DRA fraud pipeline.")
    parser.add_argument("--data-path", type=Path, default=config.DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument(
        "--models", nargs="+", choices=AVAILABLE_MODELS, default=list(config.DEFAULT_MODELS)
    )
    parser.add_argument("--trials", type=int, default=config.DEFAULT_TRIALS)
    parser.add_argument("--calibration", choices=METHODS, default="sigmoid")
    parser.add_argument("--target-fpr", type=float, default=config.TARGET_FPR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--isolation-forest", choices=("off", "check", "feature"), default="off")
    parser.add_argument("--no-model-file", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    df = load_raw(args.data_path)
    run_pipeline(
        df,
        out_dir=args.out_dir,
        model_names=args.models,
        n_trials=args.trials,
        calibration=args.calibration,
        target_fpr=args.target_fpr,
        seed=args.seed,
        positive_only=args.positive_only,
        isolation_forest=args.isolation_forest,
        save_model=not args.no_model_file,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
