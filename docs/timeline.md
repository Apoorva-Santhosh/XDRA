# X-DRA — Project Timeline

Roles:
A (Aadya) = Churn + Common ML/Calibration
B (Apoorva) = Fraud + Allocation Engine
C (Sanskriti) = Demand + Dashboard/Visualization

### Notes & Acceptance Criteria

- Work on the three domain pipelines happens in parallel wherever possible.
- Each member is the primary owner of one domain and is responsible for making that pipeline fully reproducible.
- Secondary responsibilities are assigned to balance workload, not to duplicate work.
- Do not implement the same feature separately in multiple branches. Shared functionality belongs in `src/common/`.
- Each pipeline must produce the common candidate schema before integration.
- Every invented business-cost assumption must be documented in `docs/assumptions_register.md`.
- Every pipeline requires schema and smoke tests.
- Weekly: create issues for major tasks and assign primary owner + reviewers.
- Checkpoint 1: end of Phase 3 — all three ML pipelines working.
- Checkpoint 2: end of Phase 5 — complete X-DRA research experiment.
- Final checkpoint: end of Phase 8 — complete system, report and demo.

---

## Phase 1 — Project setup & requirements finalization (Week 1) — Owners: All

- Finalize project scope, research question and success criteria.
- Set up repository, branches, environments and CI.
- Review `README.md`, `GUIDELINES.md`, `PROJECT_PLAN.md` and `assumptions_register.md`.
- Set up `src/`, `tests/`, `data/`, `outputs/` and documentation structure.
- Finalize the common candidate output schema.
- Set up common configuration and utility structure.
- Download and verify the three required datasets locally.
- Confirm exact preprocessing and splitting rules for each domain.
- Assign GitHub ownership/review responsibilities.

Deliverable: clean, reproducible repository with agreed interfaces, datasets and development workflow.

---

## Phase 2 — Domain data preparation & baseline models (Weeks 2–3) — Owners: A Churn, B Fraud, C Demand

### A — Churn

- Load and clean IBM Telco dataset.
- Handle `TotalCharges`.
- Encode categorical features.
- Prepare numerical features.
- Create stratified 80/20 train/test split.
- Establish majority-class baseline.
- Train Logistic Regression baseline.
- Evaluate accuracy, ROC-AUC and PR-AUC.

### B — Fraud

- Load Credit Card Fraud dataset.
- Sort transactions by `Time`.
- Create mandatory time-based train/test split.
- Scale `Amount`.
- Handle class imbalance.
- Train Logistic Regression baseline.
- Establish PR-AUC and FPR/recall evaluation.

### C — Demand

- Load M5 sales, calendar and price data.
- Select and document 2–3 stores/departments.
- Aggregate to store-department level.
- Reshape data into time-series format.
- Create last-28-day test split.
- Implement Naive baseline.
- Implement Moving Average baseline.
- Evaluate WAPE, MAE and RMSE.

Deliverable: each domain has a clean preprocessing pipeline and working baseline model/forecast.

---

## Phase 3 — Advanced models, tuning & domain evaluation (Weeks 3–4) — Owners: A Churn, B Fraud, C Demand

### A — Churn

- Train Random Forest.
- Train XGBoost/LightGBM.
- Tune the selected advanced model with Optuna.
- Compare models.
- Select final churn model.
- Perform probability calibration.
- Calculate Brier score and calibration curve.

### B — Fraud

- Train Random Forest.
- Train XGBoost/LightGBM.
- Tune class-imbalance parameters.
- Tune the selected model with Optuna.
- Compare models using PR-AUC.
- Generate ROC and PR curves.
- Select operating threshold at ≤5% FPR.
- Record recall at that threshold.
- Perform probability calibration.

### C — Demand

- Implement Exponential Smoothing/Holt-Winters.
- Create 7/14/28-day lag features.
- Create rolling demand features.
- Add day-of-week, SNAP and event features.
- Train global XGBoost/LightGBM.
- Compare against Naive and Moving Average.
- Select final forecasting model.
- Evaluate WAPE, MAE, RMSE and bias.

Deliverable: final candidate model selected for each domain with all required evaluation metrics.

Checkpoint: all three domain pipelines produce reproducible model results.

---

## Phase 4 — Expected business value & common pipeline outputs (Weeks 4–5) — Owners: A Churn, B Fraud, C Demand

### A — Churn

- Implement CLV proxy.
- Implement `P(save | intervention)`.
- Define retention intervention cost.
- Calculate expected retention benefit.
- Calculate retention cost.
- Calculate net expected value.
- Generate `outputs/churn_candidates.csv`.

### B — Fraud

- Implement expected financial loss calculation.
- Define investigation cost.
- Calculate expected investigation benefit.
- Calculate investigation cost.
- Calculate net expected value.
- Generate `outputs/fraud_candidates.csv`.

### C — Demand

- Convert forecast into reorder decision.
- Define starting inventory assumption.
- Define stockout cost.
- Define holding cost.
- Calculate reorder quantity.
- Calculate expected reorder benefit.
- Calculate net expected value.
- Generate `outputs/demand_candidates.csv`.

### All

- Validate outputs using the common schema.
- Add schema tests.
- Add smoke tests.
- Update `docs/assumptions_register.md`.
- Verify `net_expected_value_inr = expected_benefit_inr - expected_cost_inr`.
- Verify probabilities used in expected-value calculations are calibrated.

Deliverable: three real candidate CSVs using the exact same output schema.

---

## Phase 5 — Shared-budget allocation & core research experiment (Weeks 5–6) — Owners: B lead, A support, C support

### B — Allocation Engine

- Build allocation engine using the real candidate outputs.
- Combine fraud, churn and demand candidates.
- Accept a single shared MSME budget.
- Implement budget-constrained action selection.
- Implement joint allocation.
- Implement independent per-domain allocation baseline.
- Calculate total expected benefit, cost and net expected value.
- Calculate budget utilization and domain-wise allocation.

### A — Churn/Validation

- Integrate churn candidates into allocation.
- Verify churn actions are correctly ranked and selected.
- Test allocation under different churn candidate values.
- Review joint vs independent calculations.

### C — Demand/Validation

- Integrate demand candidates into allocation.
- Verify reorder actions are correctly handled.
- Test allocation under different demand candidate values.
- Review joint vs independent calculations.

### All

- Run the complete system using real outputs.
- Test multiple shared-budget values.
- Compare joint vs independent allocation.
- Verify that the budget constraint is never violated.

Deliverable: complete X-DRA allocation experiment.

### Core Research Question

Does jointly allocating one shared budget across fraud, churn and demand generate greater total expected business value than optimizing each domain independently?

Checkpoint: end-to-end research demo.

---

## Phase 6 — Resource-aware model selection & public-data experiment (Weeks 6–7) — Owners: C lead, A support, B support

### Resource-aware analysis

- Compare model performance across the three domains.
- Measure training time.
- Measure inference time.
- Compare computational/resource requirements.
- Compare predictive performance against resource requirements.
- Identify practical model choices for resource-constrained MSMEs.

### Public-data experiment

- Identify appropriate public context data.
- Establish business-data-only results.
- Add relevant public context where applicable.
- Retrain/evaluate affected pipelines.
- Compare prediction performance.
- Compare expected business value.
- Compare resulting allocation decisions.
- Document whether public data improves the final decision.

Deliverable: resource-aware model comparison and public-data augmentation results.

---

## Phase 7 — Sensitivity analysis, dashboard & explainability (Weeks 7–8) — Owners: A Churn/Sensitivity, B Allocation/Analysis, C Dashboard

### A — Churn & assumption sensitivity

- Test `P(save | intervention)` at 20%, 30% and 40%.
- Test retention-cost variations.
- Analyze how churn assumptions affect allocation.
- Prepare churn-related sensitivity visualizations.

### B — Allocation & cross-domain sensitivity

- Test different shared-budget sizes.
- Test fraud investigation-cost variations.
- Test demand stockout/holding-cost variations.
- Compare joint vs independent allocation across scenarios.
- Determine whether the main conclusion is robust.
- Prepare core research result tables.

### C — Dashboard

- Build unified dashboard.
- Show total budget and utilization.
- Show allocation by domain.
- Show selected fraud actions.
- Show selected churn actions.
- Show selected demand actions.
- Show expected benefit, cost and net value.
- Show joint vs independent results.
- Show model/resource comparison.
- Show sensitivity results.

### All

- Add explainability for selected actions.
- Explain why actions were prioritized.
- Verify dashboard values against pipeline outputs.

Deliverable: complete dashboard, explainability and robustness analysis.

---

## Phase 8 — Final validation, report & presentation (Weeks 8–10) — Owners: All

### Technical validation

- Run all three pipelines from a clean checkout.
- Run allocation engine using real outputs.
- Verify all tests.
- Verify Black and Ruff.
- Verify reproducibility.
- Verify assumptions register.
- Verify no raw datasets are committed.
- Verify final output files and schemas.

### Research documentation

- Finalize literature review.
- Finalize methodology.
- Document datasets and preprocessing.
- Document model selection.
- Document expected-value formulation.
- Document joint vs independent experiment.
- Document resource-aware experiment.
- Document public-data experiment.
- Document sensitivity analysis.
- Document limitations and threats to validity.
- Finalize research findings and conclusion.

### Final demo

- Prepare end-to-end demo.
- Prepare dashboard walkthrough.
- Prepare joint vs independent comparison.
- Prepare individual contribution explanations.
- Prepare viva questions.
- Finalize presentation.

Deliverable: reproducible X-DRA system, final report, dashboard and presentation.

---

## Final Project Flow

Phase 1 → Setup & Requirements
↓
Phase 2 → Domain Data + Baselines
↓
Phase 3 → Advanced Models + Evaluation
↓
Phase 4 → Expected Business Value + Common Outputs
↓
Phase 5 → Shared-Budget Allocation + Core Research Experiment
↓
Phase 6 → Resource-Aware + Public-Data Experiments
↓
Phase 7 → Sensitivity + Dashboard + Explainability
↓
Phase 8 → Final Validation + Report + Demo

---

## Responsibility Summary

### Aadya — Churn + Common ML/Calibration

Primary:
- Churn preprocessing
- Churn models
- Churn evaluation
- Churn calibration
- Churn expected-value calculation
- Churn candidate generation

Secondary:
- Common schema/validation
- Probability calibration review
- Churn sensitivity analysis
- Allocation integration review

### Apoorva — Fraud + Allocation Engine

Primary:
- Fraud preprocessing
- Fraud models
- Fraud evaluation
- Fraud threshold selection
- Fraud expected-value calculation
- Shared-budget allocation engine
- Joint vs independent experiment

Secondary:
- Allocation testing
- Cross-domain sensitivity analysis
- Core research result analysis

### Sanskriti — Demand + Dashboard

Primary:
- M5 preprocessing
- Demand forecasting
- Demand evaluation
- Inventory/reorder logic
- Demand expected-value calculation
- Unified dashboard

Secondary:
- Allocation integration testing
- Resource-aware model comparison
- Public-data experiment coordination

---

## Definition of Done

The project is complete when:

- Churn pipeline is working and reproducible.
- Fraud pipeline is working and reproducible.
- Demand pipeline is working and reproducible.
- All three pipelines produce the common candidate schema.
- Required domain-specific metrics are reported.
- Probabilities used for expected-value calculations are calibrated.
- All business assumptions are documented.
- Allocation engine works with real pipeline outputs.
- Joint and independent allocation strategies are implemented.
- Core joint-vs-independent experiment is completed.
- Resource-aware analysis is completed.
- Public-data experiment is completed.
- Sensitivity analysis is completed.
- Dashboard and explainability are completed.
- All tests and CI checks pass.
- All members can run the complete system from a clean checkout.
- Final report, presentation and demo are complete.