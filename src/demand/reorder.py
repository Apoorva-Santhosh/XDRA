"""Turn the demand forecast into store-department reorder decisions and candidate rows.

For each store-department over the forecast horizon:
    starting_inventory = starting_inventory_days x trailing average daily demand
    expected_shortage  = max(0, forecast_demand - starting_inventory)
    reorder_qty        = expected_shortage rounded up to whole units
    expected_benefit   = expected_shortage x stockout cost per unit (stockout loss avoided)
    expected_cost      = holding cost of the reorder, drawn down linearly (Q/2 on hand on average)

Every assumption value is logged in docs/assumptions_register.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.schema import REQUIRED_COLUMNS, validate_candidate_table

DOMAIN = "demand"


@dataclass(frozen=True)
class ReorderAssumptions:
    starting_inventory_days: float = 14.0
    trailing_window_days: int = 28
    gross_margin: float = 0.30
    annual_holding_rate: float = 0.25
    usd_to_inr: float = 83.0


def build_reorder_plan(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    unit_price_usd: pd.Series,
    assumptions: ReorderAssumptions = ReorderAssumptions(),
) -> pd.DataFrame:
    """Compute the reorder decision and its expected value for every store-department.

    Inputs:
        forecast: long frame of `series_id, forecast`, one row per series per horizon day.
        history: long frame of `series_id, date, sales` up to the forecast origin.
        unit_price_usd: average unit sell price indexed by `series_id`.
    Output: one row per series with `candidate_id, domain`, the intermediate quantities and
        unit economics, and the schema value columns `expected_benefit_inr, expected_cost_inr,
        net_expected_value_inr`.
    """
    a = assumptions
    by_series = forecast.groupby("series_id")["forecast"]
    plan = pd.DataFrame({"horizon_days": by_series.size(), "forecast_demand": by_series.sum()})

    trailing = history.sort_values("date").groupby("series_id").tail(a.trailing_window_days)
    avg_daily_demand = trailing.groupby("series_id")["sales"].mean().reindex(plan.index)
    plan["starting_inventory"] = a.starting_inventory_days * avg_daily_demand
    plan["expected_shortage"] = (plan["forecast_demand"] - plan["starting_inventory"]).clip(lower=0)
    plan["reorder_qty"] = np.ceil(plan["expected_shortage"])

    plan["unit_price_inr"] = unit_price_usd.reindex(plan.index) * a.usd_to_inr
    incomplete = plan.index[plan[["starting_inventory", "unit_price_inr"]].isna().any(axis=1)]
    if len(incomplete):
        raise ValueError(f"Missing sales history or unit price for: {list(incomplete)}")
    plan["stockout_cost_per_unit_inr"] = plan["unit_price_inr"] * a.gross_margin
    plan["holding_cost_per_unit_day_inr"] = (
        plan["unit_price_inr"] * (1 - a.gross_margin) * a.annual_holding_rate / 365
    )

    plan["expected_benefit_inr"] = plan["expected_shortage"] * plan["stockout_cost_per_unit_inr"]
    plan["expected_cost_inr"] = (
        plan["reorder_qty"] / 2 * plan["horizon_days"] * plan["holding_cost_per_unit_day_inr"]
    )
    plan["net_expected_value_inr"] = plan["expected_benefit_inr"] - plan["expected_cost_inr"]

    plan = plan.rename_axis("candidate_id").reset_index()
    plan.insert(1, "domain", DOMAIN)
    return plan


def to_candidates(plan: pd.DataFrame) -> pd.DataFrame:
    """Reduce a reorder plan to the shared candidate schema, ranked by net expected value.

    Input: output of `build_reorder_plan`.
    Output: frame with exactly `REQUIRED_COLUMNS`, validated against GUIDELINES.md §4.
    """
    candidates = plan[REQUIRED_COLUMNS].sort_values(
        "net_expected_value_inr", ascending=False, ignore_index=True
    )
    validate_candidate_table(candidates, expected_domain=DOMAIN)
    return candidates
