## What changed

<!-- One or two sentences. What does this PR add/fix/change? -->

## Domain checklist (delete sections that don't apply)

- [ ] Output still matches the shared schema in `GUIDELINES.md` §4 (`candidate_id, domain, expected_benefit_inr, expected_cost_inr, net_expected_value_inr`)
- [ ] Correct split rule used for this domain (`GUIDELINES.md` §3 — random for churn, time-based for fraud/demand)
- [ ] Metrics reported match `GUIDELINES.md` §5 (no plain accuracy for fraud; WAPE not accuracy for demand)
- [ ] Any new invented business-cost number is logged in `docs/assumptions_register.md`
- [ ] Calibration checked (Brier score + calibration curve) if this PR touches a probability feeding an expected-value calculation
- [ ] Added/updated tests in `tests/<domain>/`

## How I tested this

<!-- What did you run locally? Attach relevant metric output if useful. -->

## Anything reviewers should look at closely

<!-- Optional — flag anything you're unsure about. -->
