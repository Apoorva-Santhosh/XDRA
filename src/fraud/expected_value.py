"""Turn calibrated fraud probabilities into allocation-ready candidate rows.

Each transaction becomes one candidate action ("investigate this transaction"):

    expected_benefit_inr   = P(fraud) * Amount * AMOUNT_TO_INR   (loss avoided if it is fraud)
    expected_cost_inr      = INVESTIGATION_COST_INR              (flat cost of a manual review)
    net_expected_value_inr = benefit - cost

The output follows the shared contract in GUIDELINES.md section 4 and is validated with
``src.common.schema.validate_candidate_table`` before it is written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.schema import validate_candidate_table
from src.fraud import config


def build_candidate_table(
    row_ids,
    amounts,
    probs,
    investigation_cost_inr: float = config.INVESTIGATION_COST_INR,
    amount_to_inr: float = config.AMOUNT_TO_INR,
    positive_only: bool = False,
) -> pd.DataFrame:
    """Build the fraud candidate table.

    Args:
        row_ids: original row position of each transaction, used to form candidate_id.
        amounts: raw transaction amounts (unscaled).
        probs: calibrated fraud probabilities, aligned with ``row_ids``.
        investigation_cost_inr: flat cost of investigating one transaction.
        amount_to_inr: multiplier converting dataset Amount into rupee-equivalent units.
        positive_only: keep only candidates whose net expected value is positive.
            Off by default so the table is lossless; the allocator discards negative
            candidates on its own.

    Returns:
        A frame with exactly the shared-schema columns.
    """
    row_ids = np.asarray(row_ids)
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    benefit = probs * np.asarray(amounts, dtype=float) * amount_to_inr
    cost = np.full(len(benefit), float(investigation_cost_inr))

    table = pd.DataFrame(
        {
            "candidate_id": [f"fraud_{i}" for i in row_ids],
            "domain": config.DOMAIN,
            "expected_benefit_inr": benefit,
            "expected_cost_inr": cost,
            "net_expected_value_inr": benefit - cost,
        }
    )
    if positive_only:
        table = table[table["net_expected_value_inr"] > 0].reset_index(drop=True)
    return table


def write_candidate_table(table: pd.DataFrame, out_dir: str | Path) -> Path:
    """Validate against the shared schema, then write ``fraud_candidates.csv``.

    Raises:
        SchemaValidationError: if the table breaks the contract, so a bad file is never
            written for the allocation engine to consume.
    """
    validate_candidate_table(table, expected_domain=config.DOMAIN)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fraud_candidates.csv"
    table.to_csv(path, index=False)
    return path
