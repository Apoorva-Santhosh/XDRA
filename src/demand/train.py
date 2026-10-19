"""End-to-end demand pipeline: M5 data -> store-department forecasts -> reorder candidates.

Run with `python -m src.demand.train`. The final model is the one with the lowest WAPE on the
28 days before the test window; every model is then refit on the full history and scored on the
test window, whose forecast drives the reorder decisions. Writes to `outputs/`:
    demand_model_comparison.csv  WAPE / MAE / RMSE / bias per model, validation and test
    demand_reorder_plan.csv      reorder decision and unit economics per store-department
    demand_candidates.csv        shared candidate schema (GUIDELINES.md §4)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.demand.advanced_models import holt_winters_forecast, lightgbm_forecast
from src.demand.baseline import moving_average_forecast, naive_forecast
from src.demand.metrics import forecast_metrics
from src.demand.preprocess import (
    DATA_DIR,
    SELECTED_STORES,
    build_store_dept_series,
    load_m5,
    split_by_time,
    store_dept_unit_prices,
)
from src.demand.reorder import ReorderAssumptions, build_reorder_plan, to_candidates

OUTPUT_DIR = Path("outputs")

Forecaster = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
MODELS: dict[str, Forecaster] = {
    "naive": naive_forecast,
    "moving_average": moving_average_forecast,
    "holt_winters": holt_winters_forecast,
    "lightgbm": lightgbm_forecast,
}


def evaluate_models(
    history: pd.DataFrame, holdout: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Fit every model in `MODELS` on `history` and score its forecast of `holdout`.

    Input: long frames from `split_by_time`; `holdout` sales are hidden from the models.
    Output: (metrics indexed by model name, forecasts by model name aligned to `holdout.index`).
    """
    future = holdout.drop(columns="sales")
    forecasts = {name: model(history, future) for name, model in MODELS.items()}
    metrics = pd.DataFrame.from_dict(
        {name: forecast_metrics(holdout["sales"], f) for name, f in forecasts.items()},
        orient="index",
    )
    return metrics, forecasts


def run(
    data_dir: Path | str = DATA_DIR,
    output_dir: Path | str = OUTPUT_DIR,
    stores: tuple[str, ...] = SELECTED_STORES,
) -> pd.DataFrame:
    """Run the demand pipeline end to end, write its outputs and return the candidate table."""
    data = load_m5(data_dir, stores)
    series = build_store_dept_series(data.sales, data.calendar)
    history, test = split_by_time(series)
    fit, validation = split_by_time(history)

    validation_metrics, _ = evaluate_models(fit, validation)
    selected = validation_metrics["wape"].idxmin()
    test_metrics, test_forecasts = evaluate_models(history, test)

    comparison = pd.concat(
        {"validation": validation_metrics, "test": test_metrics}, names=["window", "model"]
    ).reset_index()
    comparison["selected"] = comparison["model"] == selected

    assumptions = ReorderAssumptions()
    unit_price_usd = store_dept_unit_prices(
        data, end_date=history["date"].max(), window_days=assumptions.trailing_window_days
    )
    forecast = test[["series_id", "date"]].assign(forecast=test_forecasts[selected])
    plan = build_reorder_plan(forecast, history, unit_price_usd, assumptions)
    candidates = to_candidates(plan)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "demand_model_comparison.csv", index=False)
    plan.to_csv(output_dir / "demand_reorder_plan.csv", index=False)
    candidates_path = output_dir / "demand_candidates.csv"
    candidates.to_csv(candidates_path, index=False)

    print(f"Selected model (lowest validation WAPE): {selected}\n")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n{len(candidates)} demand candidates written to {candidates_path}")
    return candidates


if __name__ == "__main__":
    run()
