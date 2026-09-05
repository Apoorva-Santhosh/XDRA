"""Shared output-schema contract for all three domain pipelines.

Every pipeline (fraud, churn, demand) must produce a candidate table matching
this schema before its output can be consumed by the allocation engine.
See GUIDELINES.md Section 4 for the full contract.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "candidate_id",
    "domain",
    "expected_benefit_inr",
    "expected_cost_inr",
    "net_expected_value_inr",
]

VALID_DOMAINS = {"fraud", "churn", "demand"}


class SchemaValidationError(ValueError):
    """Raised when a pipeline's output does not match the shared candidate schema."""


def validate_candidate_table(df: pd.DataFrame, expected_domain: str) -> None:
    """Validate a domain pipeline's output candidate table.

    Args:
        df: the candidate table produced by a domain pipeline.
        expected_domain: one of "fraud", "churn", "demand" — the domain this
            table is expected to belong to.

    Raises:
        SchemaValidationError: if any contract rule in GUIDELINES.md Section 4
            is violated.
    """
    if expected_domain not in VALID_DOMAINS:
        raise SchemaValidationError(
            f"Unknown domain '{expected_domain}'. Must be one of {VALID_DOMAINS}."
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")

    if df["candidate_id"].duplicated().any():
        raise SchemaValidationError("candidate_id must be unique within a domain's table.")

    if not (df["domain"] == expected_domain).all():
        raise SchemaValidationError(f"All rows must have domain == '{expected_domain}'.")

    for col in ("expected_benefit_inr", "expected_cost_inr"):
        if (df[col] < 0).any():
            raise SchemaValidationError(f"{col} must be >= 0 for all rows.")

    recomputed = df["expected_benefit_inr"] - df["expected_cost_inr"]
    if not (recomputed - df["net_expected_value_inr"]).abs().lt(1e-6).all():
        raise SchemaValidationError(
            "net_expected_value_inr must equal expected_benefit_inr - expected_cost_inr."
        )
