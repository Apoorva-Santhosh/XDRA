import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from src.churn.preprocess import load_and_clean, encode_features, split_data

def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
    return scores.mean()

def tune(X_train, y_train, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    print(f"\nBest PR-AUC (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params

def evaluate_tuned_model(best_params, X_train, X_test, y_train, y_test):
    best_params = {**best_params, "random_state": 42, "eval_metric": "logloss", "n_jobs": -1}
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    print(f"\nTuned XGBoost — test set performance")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    return model, {"accuracy": acc, "roc_auc": roc_auc, "pr_auc": pr_auc}

if __name__ == "__main__":
    df = load_and_clean()
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    best_params = tune(X_train, y_train, n_trials=50)
    tuned_model, tuned_metrics = evaluate_tuned_model(best_params, X_train, X_test, y_train, y_test)
