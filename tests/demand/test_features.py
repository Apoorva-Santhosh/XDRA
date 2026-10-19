import numpy as np
import pandas as pd

from src.demand.features import FEATURE_COLUMNS, add_features

N_DAYS = 60


def _two_series():
    dates = pd.date_range("2016-01-01", periods=N_DAYS, freq="D")
    frames = [
        pd.DataFrame(
            {
                "series_id": f"CA_1_{dept}",
                "store_id": "CA_1",
                "dept_id": dept,
                "date": dates,
                "sales": np.arange(N_DAYS, dtype=float) + offset,
                "wday": dates.dayofweek + 1,
                "snap": 0,
                "is_event": 0,
            }
        )
        for dept, offset in [("FOODS_1", 0.0), ("HOBBIES_1", 1000.0)]
    ]
    # Shuffled input: features must not depend on incoming row order.
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=0)


def test_lags_and_rolling_means_use_only_sales_at_least_seven_days_old():
    features = add_features(_two_series())
    hobbies = features[features["dept_id"] == "HOBBIES_1"].reset_index(drop=True)
    sales = hobbies["sales"]

    for lag in (7, 14, 28):
        pd.testing.assert_series_equal(hobbies[f"lag_{lag}"], sales.shift(lag), check_names=False)
    for window in (7, 28):
        expected = sales.shift(7).rolling(window).mean()
        pd.testing.assert_series_equal(
            hobbies[f"rolling_mean_{window}"], expected, check_names=False
        )


def test_lags_do_not_cross_series_boundaries():
    features = add_features(_two_series())
    hobbies = features[features["dept_id"] == "HOBBIES_1"]

    assert hobbies["lag_7"].iloc[:7].isna().all()
    assert (hobbies["lag_7"].dropna() >= 1000).all()


def test_all_feature_columns_present_and_index_labels_kept():
    raw = _two_series()
    features = add_features(raw)

    assert set(FEATURE_COLUMNS) <= set(features.columns)
    assert sorted(features.index) == sorted(raw.index)
    pd.testing.assert_series_equal(features["sales"], raw.loc[features.index, "sales"])
