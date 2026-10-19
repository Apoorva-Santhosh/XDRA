"""Shared fixtures for the fraud tests.

The real ULB dataset is large and gitignored, so tests run on a small synthetic frame with
the same columns. Fraud rows are shifted on a few PCA components so they are learnable.
"""

import numpy as np
import pandas as pd
import pytest

from src.fraud import config
from src.fraud.train import run_pipeline


def make_synthetic_creditcard(n: int = 6000, fraud_rate: float = 0.03, seed: int = 0):
    """Build a frame shaped like creditcard.csv (Time, V1..V28, Amount, Class)."""
    rng = np.random.default_rng(seed)
    data = {"Time": np.sort(rng.uniform(0, 172_792, n))}
    labels = (rng.random(n) < fraud_rate).astype(int)
    for col in config.PCA_COLUMNS:
        data[col] = rng.normal(0, 1, n)
    for col in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        data[col] = data[col] - 2.0 * labels
    amounts = np.where(labels == 1, rng.lognormal(4.5, 1.2, n), rng.lognormal(3.5, 1.2, n))
    data["Amount"] = np.round(amounts, 2)
    data["Class"] = labels
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_creditcard()


@pytest.fixture(scope="session")
def pipeline_run(synthetic_df, tmp_path_factory):
    """Run the whole pipeline once and share the result across tests."""
    out_dir = tmp_path_factory.mktemp("fraud_out")
    metrics = run_pipeline(synthetic_df, out_dir=out_dir, n_trials=0)
    return metrics, out_dir
