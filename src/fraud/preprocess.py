"""Loading, validation, time-based splitting and feature building for fraud data.

The split rule here is non-negotiable (GUIDELINES.md section 3): transactions are sorted
by Time and split chronologically. A random split would leak future transaction patterns
into training and invalidate every downstream metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.fraud import config

logger = logging.getLogger(__name__)


@dataclass
class Split:
    """Chronological train / validation / test partitions of the raw data."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class FeatureSet:
    """Model-ready matrices plus the raw values needed to price each candidate."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    amount_test: np.ndarray  # raw (unscaled) Amount of each test transaction
    row_id_test: np.ndarray  # original row position of each test transaction
    scaler: StandardScaler  # fitted on the training window only


def validate_raw_columns(df: pd.DataFrame) -> None:
    """Check that the raw frame has the expected schema and sane values.

    Raises:
        ValueError: if columns are missing, Class is not binary, or Amount is negative.
    """
    missing = [c for c in config.RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fraud data is missing required columns: {missing}")
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must be binary (0 = genuine, 1 = fraud).")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative.")


def check_dataset_integrity(df: pd.DataFrame) -> list[str]:
    """Compare the loaded data against the known ULB dataset and list any discrepancies.

    Returns:
        A list of human-readable warnings; empty when the data matches the expected
        284,807 rows / 492 frauds with no missing values.
    """
    warnings: list[str] = []
    if len(df) != config.EXPECTED_ROWS:
        warnings.append(
            f"Expected {config.EXPECTED_ROWS:,} rows (ULB dataset) but found {len(df):,}. "
            "Target metrics are calibrated to the original dataset."
        )
    n_fraud = int((df["Class"] == 1).sum())
    if n_fraud != config.EXPECTED_FRAUD:
        warnings.append(f"Expected {config.EXPECTED_FRAUD} fraud rows but found {n_fraud}.")
    if df[config.RAW_REQUIRED_COLUMNS].isna().any().any():
        warnings.append("Missing values found; the ULB dataset should have none.")
    return warnings


def load_raw(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Read creditcard.csv and validate its schema.

    Args:
        path: location of the ULB credit card fraud CSV.

    Returns:
        The raw frame in file order, with a ``row_id`` column recording each row's
        original position (used as the transaction identifier downstream).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fraud dataset not found at {path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it there."
        )
    df = pd.read_csv(path)
    validate_raw_columns(df)
    df["row_id"] = np.arange(len(df))
    return df


def time_split(
    df: pd.DataFrame,
    test_frac: float = config.TEST_FRAC,
    val_frac_of_train: float = config.VAL_FRAC_OF_TRAIN,
) -> Split:
    """Split chronologically into train / validation / test.

    Args:
        df: raw transactions (must contain Time and Class).
        test_frac: share of transactions, from the end of the timeline, held out as test.
        val_frac_of_train: share of the pre-test period, again from its end, used as
            validation.

    Returns:
        A ``Split`` where max(train.Time) <= min(val.Time) <= max(val.Time) <= min(test.Time).

    Raises:
        ValueError: if any partition contains only one class, since PR-AUC, calibration
            and threshold selection are undefined without both classes.
    """
    if "row_id" not in df.columns:
        df = df.assign(row_id=np.arange(len(df)))
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)

    n = len(ordered)
    n_trainval = int(round(n * (1 - test_frac)))
    n_train = int(round(n_trainval * (1 - val_frac_of_train)))

    split = Split(
        train=ordered.iloc[:n_train].copy(),
        val=ordered.iloc[n_train:n_trainval].copy(),
        test=ordered.iloc[n_trainval:].copy(),
    )
    for name, part in (("train", split.train), ("validation", split.val), ("test", split.test)):
        n_pos = int(part["Class"].sum())
        if n_pos == 0 or n_pos == len(part):
            raise ValueError(
                f"The {name} window has {n_pos} fraud rows out of {len(part)}; every "
                "window needs both classes."
            )
        logger.info("%s window: %d rows, %d fraud", name, len(part), n_pos)
    return split


def _engineer(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Build the model features for one partition using an already-fitted scaler."""
    out = df[config.PCA_COLUMNS].copy()
    out["amount_scaled"] = scaler.transform(np.log1p(df[["Amount"]].to_numpy()))[:, 0]
    out["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(float)
    return out[config.FEATURE_COLUMNS]


def build_features(split: Split) -> FeatureSet:
    """Turn a chronological split into model matrices.

    Amount is log-transformed (it is heavily right-skewed) and standardised with a scaler
    fitted on the training window only, so nothing from validation or test informs it.
    The raw Amount of the test window is kept for expected-loss pricing.
    """
    scaler = StandardScaler().fit(np.log1p(split.train[["Amount"]].to_numpy()))
    return FeatureSet(
        X_train=_engineer(split.train, scaler),
        X_val=_engineer(split.val, scaler),
        X_test=_engineer(split.test, scaler),
        y_train=split.train["Class"].to_numpy(),
        y_val=split.val["Class"].to_numpy(),
        y_test=split.test["Class"].to_numpy(),
        amount_test=split.test["Amount"].to_numpy(dtype=float),
        row_id_test=split.test["row_id"].to_numpy(),
        scaler=scaler,
    )


def add_isolation_feature(fs: FeatureSet, iso) -> FeatureSet:
    """Append an unsupervised anomaly score as an extra feature.

    Args:
        fs: the feature set to extend.
        iso: an IsolationForest already fitted on the TRAINING features only.

    Returns:
        A new feature set whose matrices carry an ``iso_score`` column (higher = more
        anomalous).
    """

    def _with_score(X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(iso_score=-iso.score_samples(X[config.FEATURE_COLUMNS]))

    return replace(
        fs,
        X_train=_with_score(fs.X_train),
        X_val=_with_score(fs.X_val),
        X_test=_with_score(fs.X_test),
    )
