# Assumptions Register

Every business-cost number that isn't directly present in a raw dataset gets logged here **before** it's used in any pipeline. This is the single place reviewers, teammates, and your project guide can check "where did this number come from" without digging through three separate codebases.

Add a new row whenever you introduce a new invented value. Do not silently change a value already in this table without updating the row and noting the change — treat this like a changelog, not a scratchpad.

| Domain | Assumption | Value | Used in | Justification |
|---|---|---|---|---|
| Churn | `P(save \| intervention)` | 30% (test 20%/30%/40% as a sensitivity range) | `src/churn/expected_value.py` | No field for this in Telco data; 30% is a defensible mid-range retention-offer success rate. Sensitivity range required — do not present 30% as fact in Experiment 4/5. |
| Churn | `RetentionCost` | ₹500 (call) / ₹1,500 (discount offer) | `src/churn/expected_value.py` | Documented flat/tiered assumption, not from data. |
| Churn | `CLV` proxy | `MonthlyCharges × expected_remaining_tenure` | `src/churn/expected_value.py` | Telco has no true CLV field; this is a standard heuristic proxy. |
| Fraud | `InvestigationCost` | ₹200 flat (or scaled by transaction complexity — decide and log which) | `src/fraud/expected_value.py` | Not in dataset; flat estimate for a manual review. |
| Demand | Store/department scope | `CA_1`, `TX_1`, `WI_1` × all 7 departments (21 store-department series) | `src/demand/preprocess.py` | §2.3 requires 2–3 stores at store-department level. One store per state covers all three state SNAP calendars; taking each state's first store avoids picking stores by forecast performance. |
| Demand | Starting inventory | 14 days of trailing 28-day average daily demand (previously: X to be filled in) | `src/demand/reorder.py` | M5 has sales, not stock levels; must be simulated. Two weeks of cover is a typical small-retailer stock level and leaves a shortage over the 28-day horizon, so a reorder decision exists. |
| Demand | `StockoutCost` | 30% gross margin × unit price, per unit of unmet demand (previously: to be filled in) | `src/demand/reorder.py` | Not in dataset. Lost margin per unit short; 30% is a mid-range retail gross margin. Unit price is the volume-weighted M5 `sell_price` over the trailing 28 days. |
| Demand | `HoldingCost` | 25% of unit cost per year, charged per unit per day; unit cost = unit price × (1 − 30% margin) (previously: to be filled in) | `src/demand/reorder.py` | Not in dataset. Within the standard 20–30%/yr carrying-cost range (capital, storage, shrinkage). Reorder stock is assumed drawn down linearly, so on average half the order is held over the 28-day horizon. |
| Demand | USD → INR rate | ₹83 per USD | `src/demand/reorder.py` | M5 `sell_price` is in USD; converted so demand values are in the same currency as the other domains. Held fixed for reproducibility. |

## Rules

1. A PR introducing a new invented cost/assumption number must update this file in the same PR — CI does not check this automatically (it's a judgment call, not a schema check), so reviewers must check it manually before approving.
2. If you change a previously-logged value, keep the old value visible (strike through or note "previously X") rather than deleting the row — history here matters for explaining results later.
3. Placeholder ranges (e.g. the churn save-rate sensitivity range) must actually be run as a sensitivity check in Phase 6 — don't pick one value and forget the range was supposed to be tested.
