from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.churn.preprocess import encode_features, load_and_clean, split_data


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    return model, {"accuracy": acc, "roc_auc": roc_auc, "pr_auc": pr_auc}


def random_forest_baseline(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    return evaluate_model(model, X_train, X_test, y_train, y_test, "Random Forest")


def xgboost_baseline(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", n_jobs=-1)
    return evaluate_model(model, X_train, X_test, y_train, y_test, "XGBoost")


if __name__ == "__main__":
    df = load_and_clean()
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    rf_model, rf_metrics = random_forest_baseline(X_train, X_test, y_train, y_test)
    xgb_model, xgb_metrics = xgboost_baseline(X_train, X_test, y_train, y_test)
