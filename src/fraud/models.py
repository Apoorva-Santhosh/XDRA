"""Model factories, hyperparameter tuning and resource profiling for fraud detection.

Models are built in the order the project plan asks for: a class-weighted logistic
regression baseline, a random forest, then gradient-boosted trees (XGBoost, optionally
LightGBM). Everything runs on CPU.

Each model is also profiled (fit time, batch prediction latency, serialized size). Those
numbers feed the resource-aware model selection experiment later in the project, so
recording them now avoids retraining everything in Phase 4.
"""

from __future__ import annotations

import io
import logging
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ("logreg", "random_forest", "xgboost", "lightgbm")


def class_ratio(y) -> float:
    """Return negatives per positive (~577 on the full ULB data), used for class weighting."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive samples in the training data.")
    return n_neg / n_pos


def default_xgb_params(ratio: float) -> dict:
    """Sensible untuned XGBoost settings, with scale_pos_weight set to the class ratio."""
    return {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": ratio,
    }


def make_model(name: str, seed: int, ratio: float, xgb_params: dict | None = None):
    """Build an unfitted classifier by name.

    Args:
        name: one of ``AVAILABLE_MODELS``.
        seed: random seed for reproducibility.
        ratio: negatives per positive in the training data.
        xgb_params: tuned XGBoost parameters overriding the defaults (xgboost only).
    """
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        params = {**default_xgb_params(ratio), **(xgb_params or {})}
        return XGBClassifier(
            tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=seed, **params
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"Unknown model '{name}'. Choose from {AVAILABLE_MODELS}.")


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int, seed: int) -> dict:
    """Search XGBoost hyperparameters with Optuna, maximising validation PR-AUC.

    Tuning on the validation window keeps the test window untouched. scale_pos_weight is
    searched rather than fixed, since the exact class ratio is rarely optimal.

    Args:
        X_train, y_train: training window.
        X_val, y_val: validation window used as the tuning objective.
        n_trials: number of Optuna trials (the plan suggests 30-50). 0 skips tuning.
        seed: seed for the sampler and the models.

    Returns:
        The best parameters found, or an empty dict when ``n_trials <= 0``.
    """
    if n_trials <= 0:
        return {}
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Hyperparameter tuning needs optuna: `pip install optuna` (and add it to "
            "requirements.txt), or run with --trials 0 to use default parameters."
        ) from exc
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ratio = class_ratio(y_train)

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 1.0, max(2.0, 2 * ratio), log=True
            ),
        }
        model = XGBClassifier(
            tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=seed, **params
        )
        model.fit(X_train, y_train)
        return float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best tuned validation PR-AUC %.4f with %s", study.best_value, study.best_params)
    return dict(study.best_params)


def fit_and_profile(model, X_train, y_train, X_eval):
    """Fit a model and record its resource footprint.

    Args:
        model: an unfitted classifier.
        X_train, y_train: training window.
        X_eval: frame scored to measure prediction latency.

    Returns:
        ``(fitted_model, eval_scores, profile)`` where profile holds fit_seconds,
        predict_ms_per_1k_rows (batch latency) and model_size_kb (joblib-serialised).
    """
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    scores = model.predict_proba(X_eval)[:, 1]
    predict_seconds = time.perf_counter() - start

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    profile = {
        "fit_seconds": fit_seconds,
        "predict_ms_per_1k_rows": predict_seconds * 1_000_000 / len(X_eval),
        "model_size_kb": buffer.getbuffer().nbytes / 1024,
    }
    return model, scores, profile


def fit_isolation_forest(X_train, seed: int) -> IsolationForest:
    """Fit an unsupervised anomaly detector on the TRAINING features only (no labels)."""
    return IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(X_train)


def isolation_scores(iso: IsolationForest, X) -> np.ndarray:
    """Anomaly score where higher means more anomalous."""
    return -iso.score_samples(X)
