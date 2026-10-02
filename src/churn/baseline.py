import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from src.churn.preprocess import load_and_clean, encode_features, split_data


def majority_class_baseline(y_train, y_test):
    majority_class = y_train.mode()[0]
    preds = np.full(shape=y_test.shape, fill_value=majority_class)
    acc = accuracy_score(y_test, preds)
    print(f"Majority-class baseline — predicts all '{majority_class}'")
    print(f"Accuracy: {acc:.4f}")
    return acc


def logistic_regression_baseline(X_train, X_test, y_train, y_test):
    # Scale numeric features (LogReg is sensitive to feature scale)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    print(f"\nLogistic Regression baseline")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    return model, scaler, {"accuracy": acc, "roc_auc": roc_auc, "pr_auc": pr_auc}


if __name__ == "__main__":
    df = load_and_clean()
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    majority_class_baseline(y_train, y_test)
    logistic_regression_baseline(X_train, X_test, y_train, y_test)
