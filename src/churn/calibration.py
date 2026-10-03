import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from src.churn.preprocess import encode_features, load_and_clean, split_data

BEST_PARAMS = {
    "n_estimators": 366,
    "max_depth": 2,
    "learning_rate": 0.0586821397910316,
    "subsample": 0.851152547181656,
    "colsample_bytree": 0.8190391410281874,
    "min_child_weight": 7,
    "gamma": 2.1634970204332697,
    "reg_alpha": 0.7323889895083887,
    "reg_lambda": 3.7848968114467025,
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def calibrate_model(X_train, y_train, method="isotonic"):
    base_model = XGBClassifier(**BEST_PARAMS)
    calibrated = CalibratedClassifierCV(base_model, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_calibration(model, X_test, y_test, label):
    probs = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    print(f"\n{label}")
    print(f"Brier score: {brier:.4f} (lower is better)")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    return probs, {"brier": brier, "roc_auc": roc_auc, "pr_auc": pr_auc}


def plot_calibration_curve(y_test, prob_dict, save_path="outputs/churn_calibration_curve.png"):
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    for label, probs in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=label)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (actual churn rate)")
    plt.title("Calibration Curve — Churn Model")
    plt.legend()
    plt.tight_layout()
    import os

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(save_path)
    print(f"\nCalibration curve saved to {save_path}")


if __name__ == "__main__":
    df = load_and_clean()
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    uncalibrated = XGBClassifier(**BEST_PARAMS)
    uncalibrated.fit(X_train, y_train)
    uncal_probs, uncal_metrics = evaluate_calibration(
        uncalibrated, X_test, y_test, "Uncalibrated XGBoost"
    )

    calibrated = calibrate_model(X_train, y_train, method="isotonic")
    cal_probs, cal_metrics = evaluate_calibration(
        calibrated, X_test, y_test, "Calibrated XGBoost (isotonic)"
    )

    plot_calibration_curve(
        y_test, {"Uncalibrated": uncal_probs, "Calibrated (isotonic)": cal_probs}
    )
