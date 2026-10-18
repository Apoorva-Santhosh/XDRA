import numpy as np
import pytest

from src.fraud import config
from src.fraud.preprocess import (
    build_features,
    check_dataset_integrity,
    time_split,
    validate_raw_columns,
)


def test_split_is_chronological_and_disjoint(synthetic_df):
    split = time_split(synthetic_df)
    assert split.train["Time"].max() <= split.val["Time"].min()
    assert split.val["Time"].max() <= split.test["Time"].min()
    ids = [set(p["row_id"]) for p in (split.train, split.val, split.test)]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])
    assert sum(len(i) for i in ids) == len(synthetic_df)


def test_split_sizes_follow_config(synthetic_df):
    split = time_split(synthetic_df)
    n = len(synthetic_df)
    assert len(split.test) == pytest.approx(n * config.TEST_FRAC, abs=2)
    assert len(split.val) == pytest.approx(n * 0.8 * config.VAL_FRAC_OF_TRAIN, abs=2)


def test_split_ignores_input_row_order(synthetic_df):
    shuffled = synthetic_df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    split = time_split(shuffled)
    assert split.train["Time"].max() <= split.test["Time"].min()


def test_split_rejects_window_without_fraud(synthetic_df):
    df = synthetic_df.copy()
    late = df["Time"] > df["Time"].quantile(0.7)
    df.loc[late, "Class"] = 0
    with pytest.raises(ValueError, match="both classes"):
        time_split(df)


def test_features_exclude_time_and_label(synthetic_df):
    fs = build_features(time_split(synthetic_df))
    assert list(fs.X_train.columns) == config.FEATURE_COLUMNS
    assert "Time" not in fs.X_train.columns
    assert "Class" not in fs.X_train.columns
    assert fs.X_train["hour_of_day"].between(0, 23).all()


def test_amount_scaler_is_fit_on_training_window_only(synthetic_df):
    fs = build_features(time_split(synthetic_df))
    assert fs.X_train["amount_scaled"].mean() == pytest.approx(0.0, abs=1e-9)
    assert fs.X_train["amount_scaled"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_raw_amount_and_ids_preserved_for_pricing(synthetic_df):
    split = time_split(synthetic_df)
    fs = build_features(split)
    assert np.array_equal(fs.amount_test, split.test["Amount"].to_numpy())
    assert np.array_equal(fs.row_id_test, split.test["row_id"].to_numpy())


def test_validate_raw_columns_rejects_missing_column(synthetic_df):
    with pytest.raises(ValueError, match="missing"):
        validate_raw_columns(synthetic_df.drop(columns=["V7"]))


def test_validate_raw_columns_rejects_negative_amount(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Amount"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        validate_raw_columns(df)


def test_validate_raw_columns_rejects_non_binary_class(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_raw_columns(df)


def test_integrity_check_flags_non_ulb_data(synthetic_df):
    warnings = check_dataset_integrity(synthetic_df)
    assert any("284,807" in w for w in warnings)
    assert any("492" in w for w in warnings)
