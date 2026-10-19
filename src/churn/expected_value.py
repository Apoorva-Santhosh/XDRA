"""Phase 4: turn churn probabilities into expected-value candidates (INR).

Output: outputs/churn_candidates.csv in the common candidate schema.
Every constant below is an invented business assumption and must stay in
sync with docs/assumptions_register.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src.churn.preprocess import encode_features, load_and_clean

DATA_PATH = "data/churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = Path("outputs/churn_candidates.csv")

# --- Assumptions (see docs/assumptions_register.md) -------------------------
USD_TO_INR = 85.0
P_SAVE_CALL = 0.30
P_SAVE_DISCOUNT = 0.45
CALL_COST_INR = 500.0
DISCOUNT_COST_INR = 1500.0
REMAINING_TENURE_MONTHS = {"Month-to-month": 12, "One year": 24, "Two year": 36}

# Best parameters from the Optuna run (tune_xgboost.py)
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

SCHEMA_COLUMNS = [
    "candidate_id",
    "domain",
    "expected_benefit_inr",
    "expected_cost_inr",
    "net_expected_value_inr",
]


def out_of_fold_probabilities(X, y):
    """Honest churn probabilities for every customer (no row scored by a
    model that saw its own label)."""
    model = XGBClassifier(**BEST_PARAMS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def compute_candidates(raw, churn_prob):
    """Build the candidate table.

    raw: DataFrame with columns customerID, MonthlyCharges, Contract.
    churn_prob: array of churn probabilities, same length as raw.
    """
    churn_prob = np.asarray(churn_prob, dtype=float)
    if len(churn_prob) != len(raw):
        raise ValueError("churn_prob and raw must have the same length")
    if ((churn_prob < 0) | (churn_prob > 1)).any():
        raise ValueError("churn_prob must be within [0, 1]")

    months = raw["Contract"].map(REMAINING_TENURE_MONTHS)
    if months.isna().any():
        raise ValueError("Unknown Contract value(s) found")

    clv = raw["MonthlyCharges"] * USD_TO_INR * months
    value_at_risk = clv * churn_prob

    # Two possible interventions per customer; fund whichever nets more.
    net_call = value_at_risk * P_SAVE_CALL - CALL_COST_INR
    net_discount = value_at_risk * P_SAVE_DISCOUNT - DISCOUNT_COST_INR
    use_discount = net_discount > net_call

    benefit = pd.Series(
        np.where(
            use_discount,
            value_at_risk * P_SAVE_DISCOUNT,
            value_at_risk * P_SAVE_CALL,
        ),
        index=raw.index,
    ).round(2)
    cost = pd.Series(
        np.where(use_discount, DISCOUNT_COST_INR, CALL_COST_INR),
        index=raw.index,
    )

    out = pd.DataFrame(
        {
            "candidate_id": "churn_" + raw["customerID"].astype(str),
            "domain": "churn",
            "expected_benefit_inr": benefit,
            "expected_cost_inr": cost,
            "net_expected_value_inr": (benefit - cost).round(2),
        }
    )
    return out.reset_index(drop=True)[SCHEMA_COLUMNS]


def main():
    raw = load_and_clean(DATA_PATH)
    ids = pd.read_csv(DATA_PATH, usecols=["customerID"])["customerID"]
    if len(ids) != len(raw):
        raise ValueError("customerID rows do not align with cleaned data")

    # encode_features edits its input, so give it a copy
    encoded = encode_features(raw.copy())
    X = encoded.drop(columns=["Churn"])
    y = encoded["Churn"]

    probs = out_of_fold_probabilities(X, y)
    print(f"Out-of-fold ROC-AUC: {roc_auc_score(y, probs):.4f}")
    print(f"Out-of-fold PR-AUC:  {average_precision_score(y, probs):.4f}")

    raw = raw.assign(customerID=ids.to_numpy())
    candidates = compute_candidates(raw, probs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUTPUT_PATH, index=False)

    net = candidates["net_expected_value_inr"]
    funded = candidates[net > 0]
    n_discount = (candidates["expected_cost_inr"] == DISCOUNT_COST_INR).sum()
    print(f"\nWrote {len(candidates)} candidates to {OUTPUT_PATH}")
    print(f"Positive net value: {len(funded)} ({len(funded) / len(candidates):.1%})")
    print(f"Discount-offer candidates: {n_discount}")
    print(
        f"Total net value if all positive funded: Rs {funded['net_expected_value_inr'].sum():,.0f}"
    )
    print(f"Total cost if all positive funded: Rs {funded['expected_cost_inr'].sum():,.0f}")


if __name__ == "__main__":
    main()
