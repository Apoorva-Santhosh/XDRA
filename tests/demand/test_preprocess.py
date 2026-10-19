import pandas as pd
import pytest

from src.demand.preprocess import (
    SERIES_COLUMNS,
    M5Data,
    build_store_dept_series,
    load_m5,
    split_by_time,
    store_dept_unit_prices,
)

STORES = ("CA_1", "TX_1")


def _series(m5_dir):
    data = load_m5(m5_dir, STORES)
    return data, build_store_dept_series(data.sales, data.calendar)


def test_series_are_store_department_sums(m5_dir):
    data, series = _series(m5_dir)

    assert list(series.columns) == SERIES_COLUMNS
    assert series["series_id"].nunique() == 4
    assert not series.duplicated(["series_id", "date"]).any()

    items = data.sales[(data.sales["store_id"] == "TX_1") & (data.sales["dept_id"] == "HOBBIES_1")]
    day_10 = series[(series["series_id"] == "TX_1_HOBBIES_1") & (series["date"] == "2011-02-07")]
    assert day_10["sales"].item() == items["d_10"].sum()


def test_snap_flag_follows_store_state(m5_dir):
    data, series = _series(m5_dir)
    snap_by_date = data.calendar.set_index("date")

    for store, state in [("CA_1", "CA"), ("TX_1", "TX")]:
        store_rows = series[series["store_id"] == store]
        expected = snap_by_date.loc[store_rows["date"], f"snap_{state}"].to_numpy()
        assert (store_rows["snap"].to_numpy() == expected).all()


def test_calendar_without_day_column_is_supported(m5_dir):
    calendar_path = m5_dir / "calendar.csv"
    with_d = load_m5(m5_dir, STORES).calendar
    pd.read_csv(calendar_path).drop(columns="d").to_csv(calendar_path, index=False)

    without_d = load_m5(m5_dir, STORES).calendar

    pd.testing.assert_series_equal(with_d.set_index("date")["d"], without_d.set_index("date")["d"])


def test_unknown_store_is_rejected(m5_dir):
    with pytest.raises(ValueError, match="WI_1"):
        load_m5(m5_dir, ("CA_1", "WI_1"))


def test_split_holds_out_last_days_for_every_series(m5_dir):
    _, series = _series(m5_dir)

    history, holdout = split_by_time(series, horizon_days=28)

    assert history["date"].max() < holdout["date"].min()
    assert holdout["date"].nunique() == 28
    assert len(history) + len(holdout) == len(series)
    assert set(holdout["series_id"]) == set(history["series_id"]) == set(series["series_id"])


def test_split_requires_enough_dates(m5_dir):
    _, series = _series(m5_dir)

    with pytest.raises(ValueError):
        split_by_time(series, horizon_days=series["date"].nunique())


def test_unit_price_is_volume_weighted():
    calendar = pd.DataFrame(
        {"d": ["d_1", "d_2"], "date": pd.to_datetime(["2016-01-01", "2016-01-02"]), "wm_yr_wk": 1}
    )
    sales = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "dept_id": "FOODS_1",
            "store_id": "CA_1",
            "state_id": "CA",
            "d_1": [3, 1],
            "d_2": [3, 0],
        }
    )
    prices = pd.DataFrame(
        {"store_id": "CA_1", "item_id": ["A", "B"], "wm_yr_wk": 1, "sell_price": [1.0, 5.0]}
    )

    price = store_dept_unit_prices(
        M5Data(sales, calendar, prices), end_date=pd.Timestamp("2016-01-02"), window_days=2
    )

    assert price["CA_1_FOODS_1"] == pytest.approx((6 * 1.0 + 1 * 5.0) / 7)
