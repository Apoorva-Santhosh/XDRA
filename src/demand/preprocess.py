"""M5 loading, store-department aggregation and time-based splitting for the demand pipeline.

Scoping follows GUIDELINES.md §2.3 and §3: store-department granularity for the stores in
`SELECTED_STORES` (logged in docs/assumptions_register.md), and a time-based holdout of the
last `HORIZON_DAYS` days.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/demand")
SELECTED_STORES = ("CA_1", "TX_1", "WI_1")
HORIZON_DAYS = 28

SERIES_COLUMNS = ["series_id", "store_id", "dept_id", "date", "sales", "wday", "snap", "is_event"]


@dataclass(frozen=True)
class M5Data:
    sales: pd.DataFrame
    calendar: pd.DataFrame
    prices: pd.DataFrame


def load_m5(data_dir: Path | str = DATA_DIR, stores: tuple[str, ...] = SELECTED_STORES) -> M5Data:
    """Load the three M5 files, keeping only rows that belong to `stores`.

    Output:
        sales: one row per (item_id, store_id) with `item_id, dept_id, store_id, state_id` and
            integer daily unit sales in `d_1 ... d_N`.
        calendar: one row per day `d` with `date, wm_yr_wk, wday`, event and `snap_<state>` columns.
        prices: one row per (store_id, item_id, wm_yr_wk) with `sell_price` in USD.
    """
    data_dir = Path(data_dir)
    sales_path = data_dir / "sales_train_validation.csv"
    header = pd.read_csv(sales_path, nrows=0).columns
    sales = pd.read_csv(sales_path, dtype={c: np.int32 for c in header if c.startswith("d_")})
    sales = sales[sales["store_id"].isin(stores)].reset_index(drop=True)

    missing = sorted(set(stores) - set(sales["store_id"]))
    if missing:
        raise ValueError(f"Stores not found in {sales_path}: {missing}")

    calendar = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    if "d" not in calendar.columns:
        # The GitHub mirror of M5 drops Kaggle's `d` column; rows are the same days in order.
        calendar = calendar.sort_values("date", ignore_index=True)
        calendar["d"] = "d_" + (calendar.index + 1).astype(str)
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    prices = prices[prices["store_id"].isin(stores)].reset_index(drop=True)
    return M5Data(sales=sales, calendar=calendar, prices=prices)


def build_store_dept_series(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level sales into daily store-department series with calendar covariates.

    Input: `sales` and `calendar` as returned by `load_m5`.
    Output: long frame with `SERIES_COLUMNS`, one row per (series_id, date), sorted by series then
        date. `series_id` is `<store_id>_<dept_id>`; `snap` is the flag for the store's own state;
        `is_event` is 1 on days with any named calendar event.
    """
    keys = ["store_id", "dept_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = (
        sales.groupby(keys)[day_cols]
        .sum()
        .stack()
        .rename("sales")
        .rename_axis([*keys, "d"])
        .reset_index()
    )

    snap_cols = [c for c in calendar.columns if c.startswith("snap_")]
    snap = calendar.melt(id_vars="d", value_vars=snap_cols, var_name="state_id", value_name="snap")
    snap["state_id"] = snap["state_id"].str.removeprefix("snap_")

    calendar_features = calendar[["d", "date", "wday"]].assign(
        is_event=calendar["event_name_1"].notna().astype(int)
    )
    long = long.merge(calendar_features, on="d", how="left", validate="many_to_one").merge(
        snap, on=["d", "state_id"], how="left", validate="many_to_one"
    )

    long["series_id"] = long["store_id"] + "_" + long["dept_id"]
    long["sales"] = long["sales"].astype(float)
    return long[SERIES_COLUMNS].sort_values(["series_id", "date"], ignore_index=True)


def split_by_time(
    series: pd.DataFrame, horizon_days: int = HORIZON_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `horizon_days` dates of every series (GUIDELINES.md §3, never shuffled).

    Input: long frame with a `date` column.
    Output: (history, holdout) — holdout holds the final `horizon_days` dates, history everything
        strictly before them.
    """
    dates = np.sort(series["date"].unique())
    if len(dates) <= horizon_days:
        raise ValueError(f"Need more than {horizon_days} dates to split, got {len(dates)}.")
    is_holdout = series["date"] >= dates[-horizon_days]
    return series[~is_holdout].reset_index(drop=True), series[is_holdout].reset_index(drop=True)


def store_dept_unit_prices(data: M5Data, end_date: pd.Timestamp, window_days: int) -> pd.Series:
    """Volume-weighted average sell price per store-department over a trailing window.

    Input: `data` from `load_m5`; the window is the `window_days` days ending on `end_date`.
    Output: Series of USD unit prices indexed by `series_id`.
    """
    start_date = end_date - pd.Timedelta(days=window_days - 1)
    window = data.calendar.loc[
        data.calendar["date"].between(start_date, end_date), ["d", "wm_yr_wk"]
    ]
    units = data.sales.melt(
        id_vars=["item_id", "store_id", "dept_id"],
        value_vars=window["d"].tolist(),
        var_name="d",
        value_name="units",
    )
    priced = units.merge(window, on="d").merge(data.prices, on=["store_id", "item_id", "wm_yr_wk"])
    priced["revenue"] = priced["units"] * priced["sell_price"]

    totals = priced.groupby(["store_id", "dept_id"], as_index=False)[["revenue", "units"]].sum()
    if (totals["units"] <= 0).any():
        empty = totals.loc[totals["units"] <= 0, ["store_id", "dept_id"]].values.tolist()
        raise ValueError(f"No priced sales in the trailing window for: {empty}")

    series_id = totals["store_id"] + "_" + totals["dept_id"]
    return pd.Series(
        (totals["revenue"] / totals["units"]).to_numpy(), index=series_id, name="unit_price_usd"
    )
