import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean(path="data/churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)

    # Fix TotalCharges: blank strings -> NaN -> 0 (these are tenure=0 new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop customerID - not a feature
    df = df.drop(columns=["customerID"])

    # Target: convert Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df):
    # Binary Yes/No columns -> 1/0
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Multi-category columns -> one-hot encode
    multi_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_and_clean()
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")
    print(f"TotalCharges NaN count after cleaning: {df['TotalCharges'].isna().sum()}")
