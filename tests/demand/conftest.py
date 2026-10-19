from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STORES = {"CA_1": "CA", "TX_1": "TX"}
DEPTS = {"FOODS_1": "FOODS", "HOBBIES_1": "HOBBIES"}
ITEMS_PER_DEPT = 2
N_DAYS = 150


@pytest.fixture
def m5_dir(tmp_path: Path) -> Path:
    """A tiny M5-format dataset (2 stores x 2 departments x 2 items, 150 days) on disk."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")
    day = np.arange(N_DAYS)
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + day // 7,
            "weekday": dates.day_name(),
            "wday": day % 7 + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in day + 1],
            "event_name_1": np.where(day % 30 == 0, "SuperBowl", None),
            "event_type_1": np.where(day % 30 == 0, "Sporting", None),
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": (dates.day <= 10).astype(int),
            "snap_TX": ((dates.day > 5) & (dates.day <= 15)).astype(int),
            "snap_WI": ((dates.day > 10) & (dates.day <= 20)).astype(int),
        }
    )

    weekly = 1 + 0.3 * np.sin(2 * np.pi * day / 7)
    sales_rows, price_rows = [], []
    for store, state in STORES.items():
        for dept, cat in DEPTS.items():
            for i in range(ITEMS_PER_DEPT):
                item = f"{dept}_{i:03d}"
                units = rng.poisson(20 * weekly)
                sales_rows.append(
                    {
                        "item_id": item,
                        "dept_id": dept,
                        "cat_id": cat,
                        "store_id": store,
                        "state_id": state,
                        **{f"d_{k + 1}": v for k, v in enumerate(units)},
                    }
                )
                for week in calendar["wm_yr_wk"].unique():
                    price_rows.append(
                        {
                            "store_id": store,
                            "item_id": item,
                            "wm_yr_wk": week,
                            "sell_price": 2.0 + i,
                        }
                    )

    pd.DataFrame(sales_rows).to_csv(tmp_path / "sales_train_validation.csv", index=False)
    calendar.to_csv(tmp_path / "calendar.csv", index=False)
    pd.DataFrame(price_rows).to_csv(tmp_path / "sell_prices.csv", index=False)
    return tmp_path
