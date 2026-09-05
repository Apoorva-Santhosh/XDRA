# Assumptions Register

Every business-cost number that isn't directly present in a raw dataset gets logged here **before** it's used in any pipeline. This is the single place reviewers, teammates, and your project guide can check "where did this number come from" without digging through three separate codebases.

Add a new row whenever you introduce a new invented value. Do not silently change a value already in this table without updating the row and noting the change — treat this like a changelog, not a scratchpad.

| Domain | Assumption | Value | Used in | Justification |
|---|---|---|---|---|
| Churn | `P(save \| intervention)` | 30% (test 20%/30%/40% as a sensitivity range) | `src/churn/expected_value.py` | No field for this in Telco data; 30% is a defensible mid-range retention-offer success rate. Sensitivity range required — do not present 30% as fact in Experiment 4/5. |
| Churn | `RetentionCost` | ₹500 (call) / ₹1,500 (discount offer) | `src/churn/expected_value.py` | Documented flat/tiered assumption, not from data. |
| Churn | `CLV` proxy | `MonthlyCharges × expected_remaining_tenure` | `src/churn/expected_value.py` | Telco has no true CLV field; this is a standard heuristic proxy. |
| Fraud | `InvestigationCost` | ₹200 flat (or scaled by transaction complexity — decide and log which) | `src/fraud/expected_value.py` | Not in dataset; flat estimate for a manual review. |
| Demand | Starting inventory | "X days of trailing average demand" — **fill in X** | `src/demand/reorder.py` | M5 has sales, not stock levels; must be simulated. |
| Demand | `StockoutCost` | *fill in* (e.g. per-unit margin lost) | `src/demand/reorder.py` | Not in dataset. |
| Demand | `HoldingCost` | *fill in* (e.g. per-unit storage cost/day) | `src/demand/reorder.py` | Not in dataset. |

## Rules

1. A PR introducing a new invented cost/assumption number must update this file in the same PR — CI does not check this automatically (it's a judgment call, not a schema check), so reviewers must check it manually before approving.
2. If you change a previously-logged value, keep the old value visible (strike through or note "previously X") rather than deleting the row — history here matters for explaining results later.
3. Placeholder ranges (e.g. the churn save-rate sensitivity range) must actually be run as a sensitivity check in Phase 6 — don't pick one value and forget the range was supposed to be tested.
