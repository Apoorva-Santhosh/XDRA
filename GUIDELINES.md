# X-DRA Pipeline Guidelines

This file is binding for all three domain pipelines. The allocation engine (Section 10 of the project plan) only works if every pipeline's output is comparable — read this in full before writing training code, not after.

---

## 1. Team assignment

| Domain | Owner | Backup reviewer |
|---|---|---|
| Churn (easiest — build first) | *fill in* | *fill in* |
| Fraud | *fill in* | *fill in* |
| Demand (hardest — build last) | *fill in* | *fill in* |

Rules:
- Each pipeline has exactly one owner. The other two teammates are backup reviewers who must be able to run the pipeline end-to-end from a clean checkout — if only the owner can run it, it isn't done.
- Every PR touching a domain folder needs approval from at least one person who is *not* that domain's owner (enforced by CODEOWNERS + branch protection — see `docs/BRANCH_PROTECTION_SETUP.md`).
- The allocation engine (`src/allocation/`) is joint-owned by all three — no single-owner rule there.

---

## 2. Datasets — exact sources, no substitutions

Do not swap in a different version of a dataset "because it was easier to find" — the target metrics in §5 are calibrated to these exact datasets and the scoping decisions below. If a substitution is genuinely necessary, it must be discussed and logged in `docs/assumptions_register.md` before training starts, not after results come in.

### 2.1 Churn
- **Source:** IBM Telco Customer Churn (Kaggle), ~7,043 rows.
- **Location:** `data/churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Known issue:** `TotalCharges` has blank strings for zero-tenure customers — coerce to numeric, fill with 0. Do this in `src/churn/preprocess.py`, not ad hoc in a notebook.

### 2.2 Fraud
- **Source:** Credit Card Fraud Detection (Kaggle/ULB), ~284,807 rows, 492 fraud, PCA-anonymized `V1`–`V28` + `Amount` + `Time`.
- **Location:** `data/fraud/creditcard.csv`
- **Mandatory:** split by `Time`, sorted ascending — train on earlier transactions, test on later ones. A random split here invalidates the entire fraud pipeline's results; this is not a style preference.

### 2.3 Demand
- **Source:** M5 Forecasting — Accuracy (Kaggle/GitHub), `sales_train_validation.csv` + `calendar.csv` + `sell_prices.csv`.
- **Location:** `data/demand/`
- **Mandatory scoping:** aggregate to **store-department level** (e.g., `CA_1` × `FOODS_1`), 2–3 stores × their departments. Do **not** use the full 42,840-series hierarchy and do **not** collapse to raw SKU-item level. This exact granularity is what makes the ≤15% WAPE target achievable — see Section 30.4 of the project plan for why. If you're unsure which stores/departments to pick, ask before choosing arbitrarily; the choice should be documented in `docs/assumptions_register.md`.

None of the three raw datasets are committed to the repo (see `.gitignore`) — each teammate downloads their own copy locally.

---

## 3. Splitting rules — non-negotiable

| Domain | Split rule | Why |
|---|---|---|
| Churn | Stratified random 80/20 | No time dimension in this dataset |
| Fraud | Time-based (sort by `Time`, train on earlier) | Random split leaks future patterns into training |
| Demand | Time-based holdout, last 28 days as test | Random split on time series silently inflates every metric |

**Never** use a random shuffle on fraud or demand data — a PR that does this will fail review regardless of what the resulting metrics look like, because the metrics would be meaningless.

---

## 4. Output schema — the contract every pipeline must produce

Regardless of what each domain's model looks like internally, its final pipeline output **must** be a CSV with exactly these columns:

```
candidate_id, domain, expected_benefit_inr, expected_cost_inr, net_expected_value_inr
```

- `candidate_id` — string, unique within the domain (e.g. transaction ID, customer ID, product-store ID)
- `domain` — one of `fraud`, `churn`, `demand`
- `expected_benefit_inr` — float, ≥ 0
- `expected_cost_inr` — float, ≥ 0
- `net_expected_value_inr` — float, `expected_benefit_inr - expected_cost_inr` (computed, not independently estimated)

Written to `outputs/<domain>_candidates.csv`. A pipeline PR is not mergeable until it produces this file correctly — `tests/<domain>/test_output_schema.py` checks this automatically in CI (see `src/common/schema.py` for the shared validator; reuse it, don't reimplement it per domain).

---

## 5. Metric reporting rules — per domain

Do not deviate from these without a documented reason. Misreported metrics here are the kind of mistake that undermines the whole project's credibility, not just one pipeline.

### Churn
- Primary metric: **accuracy** (target ≥80%), reported alongside PR-AUC, ROC-AUC, Brier score, and a calibration curve.
- Do not report accuracy without also reporting the majority-class baseline (~73%) for context.

### Fraud
- **Never report plain accuracy** — it's meaningless at 0.17% imbalance and misrepresents the work.
- Primary metrics: PR-AUC, and **recall at a chosen operating threshold set to ≤5% FPR** (read the threshold off the ROC/PR curve after training — this is a threshold choice, not a training outcome).
- Report both the FPR and the recall achieved at that threshold together, always. Never report one without the other.

### Demand
- Accuracy/FPR do not apply — this is regression.
- Primary metric: **WAPE** (target ≤15%), reported alongside MAE, RMSE, and forecast bias.
- Always report the naive-forecast and moving-average baselines alongside your model's result, even though they'll be beaten easily — this shows the improvement is real, not assumed.

### All domains — calibration matters more here than usual
Every probability feeding into `expected_benefit_inr`/`expected_cost_inr` must be calibration-checked (Brier score + calibration curve) before being trusted in the expected-value formula. A miscalibrated model silently corrupts every downstream number in Phase 3 — this is not optional polish.

---

## 6. Business-cost assumptions — log everything

Every domain invents at least one business-cost number that the raw dataset doesn't provide (e.g. churn's `P(save|intervention)`, fraud's `InvestigationCost`, demand's `HoldingCost`/`StockoutCost`, starting inventory levels). None of these are committed to code as magic numbers.

**Rule:** every such assumption goes in `docs/assumptions_register.md`, with: the value chosen, which domain/file uses it, and one line of justification. A PR that introduces a new invented cost number without updating this file will fail review. See the file itself for the current register and required format.

---

## 7. Code standards

- Python 3.11+, one virtual environment per teammate (see README §3).
- Formatting: `black`, linting: `ruff` — both run in CI (see `.github/workflows/ci.yml`); a PR with lint errors cannot merge.
- Every function that transforms data needs a docstring stating input/output shape — this matters more than usual here because three different people are building three pipelines that must interoperate.
- No notebook code is ever the source of truth. Notebooks under `notebooks/` are for exploration only; anything that needs to run repeatably (preprocessing, training, evaluation) belongs in `src/<domain>/` as a proper module with tests.
- Every pipeline module needs at least one test in `tests/<domain>/` — at minimum, a schema test confirming the output matches §4, and a smoke test confirming the pipeline runs end-to-end on a small sample without errors. CI blocks merges on failing tests (see `docs/BRANCH_PROTECTION_SETUP.md`).

## 8. Definition of done, per pipeline

A domain pipeline is not "done" until all of the following are true — this is what reviewers should check a PR against, not just "does the code run":

- [ ] Produces `outputs/<domain>_candidates.csv` matching the schema in §4
- [ ] Uses the correct split rule from §3
- [ ] Reports the correct metrics from §5 (and does not report a metric that's misleading for that domain)
- [ ] Every invented cost/business assumption is logged in `docs/assumptions_register.md`
- [ ] Calibration checked (Brier score + calibration curve) for any probability feeding the expected-value formula
- [ ] Has at least one schema test and one smoke test in `tests/<domain>/`
- [ ] Runs cleanly on a teammate's machine other than the owner's, from a clean checkout
