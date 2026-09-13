# X-DRA — Project Timeline

Roles:
A (Aadya) = Churn + Common ML/Calibration
B (Apoorva) = Fraud + Allocation Engine
C (Sanskriti) = Demand + Dashboard/Resource Analysis

### Notes & Acceptance Criteria

- Weekly: create issues for each major task, assign primary owner + two reviewers.
- Each member owns one domain pipeline and one secondary project responsibility.
- All three pipelines must follow `GUIDELINES.md` and produce the exact common candidate schema.
- No raw datasets are committed to the repository.
- Every invented business-cost assumption must be added to `docs/assumptions_register.md`.
- Every pipeline requires schema + smoke tests before it is considered complete.
- Demo checkpoint: Phase 4 (initial ML prototype).
- Core research checkpoint: Phase 6 (joint vs independent allocation).
- Final demo: Phase 10 (complete X-DRA system).

## Phase 1 — Project setup & common foundation (Week 1) — Owners: A lead, B support, C support

- Repo, branching strategy, environments and CI.
- Review `README.md`, `GUIDELINES.md` and project plan.
- Finalize common candidate output schema.
- Set up `src/common/` utilities and schema validation.
- Set up domain folders, tests and output directories.
- Download and verify all three datasets locally.
- Finalize initial `assumptions_register.md`.
- Deliverable: working repo, CI, common schema validator and verified datasets.

## Phase 2 — Allocation engine & synthetic validation (Weeks 1–2) — Owners: B lead, A support, C support

- Define candidate-action representation and expected-value interface.
- Implement shared-budget constraint.
- Implement joint allocation strategy.
- Implement independent per-domain allocation baseline.
- Generate synthetic fraud, churn and demand candidates.
- Test allocation with multiple budget sizes.
- Add unit tests for allocation logic and budget feasibility.
- Deliverable: allocation engine validated on synthetic data.

## Phase 3 — Domain data preparation & baseline models (Weeks 2–3) — Owners: A lead for Churn, B lead for Fraud, C lead for Demand

### Aadya — Churn

- Clean `TotalCharges`.
- Encode categorical variables and prepare features.
- Create stratified 80/20 split.
- Establish majority-class baseline.
- Train Logistic Regression baseline.
- Report accuracy, ROC-AUC and PR-AUC.

### Apoorva — Fraud

- Load and sort transactions by `Time`.
- Create mandatory time-based train/test split.
- Scale `Amount`.
- Handle class imbalance.
- Train Logistic Regression baseline.
- Establish PR-AUC and ≤5% FPR evaluation procedure.

### Sanskriti — Demand

- Select 2–3 stores and departments.
- Document store/department selection.
- Aggregate M5 data to store-department level.
- Create time-series structure.
- Create last-28-day test split.
- Implement Naive baseline.
- Implement Moving Average baseline.

- Deliverable: all three domains have clean preprocessing and working baseline models.

## Phase 4 — Advanced models & model evaluation (Weeks 3–4) — Owners: A for Churn, B for Fraud, C for Demand

### Aadya — Churn

- Random Forest.
- XGBoost/LightGBM.
- Initial Optuna tuning.
- Compare accuracy, ROC-AUC and PR-AUC.
- Select candidate model.

### Apoorva — Fraud

- Random Forest.
- XGBoost/LightGBM.
- Class-imbalance tuning.
- Optuna tuning.
- PR-AUC and ROC/PR curves.
- Select threshold at ≤5% FPR.
- Report recall at selected threshold.

### Sanskriti — Demand

- Exponential Smoothing/Holt-Winters.
- 7/14/28-day lag features.
- Rolling features.
- Day-of-week, SNAP and event features.
- Global XGBoost/LightGBM.
- Compare against Naive and Moving Average.

- Deliverable: best candidate model selected for each domain with correct domain-specific metrics.
- Checkpoint: initial ML prototype demo.

## Phase 5 — Calibration & expected business value (Weeks 4–5) — Owners: A for Churn/Calibration, B for Fraud, C for Demand

### Aadya — Churn

- Probability calibration.
- Brier score and calibration curve.
- CLV proxy.
- `P(save | intervention)`.
- Retention cost.
- Expected retention value.
- Generate churn candidate values.

### Apoorva — Fraud

- Probability calibration.
- Brier score and calibration curve.
- Expected fraud loss.
- Investigation cost.
- Expected investigation value.
- Generate fraud candidate values.

### Sanskriti — Demand

- Convert forecast into reorder decision.
- Simulate starting inventory.
- Define stockout cost.
- Define holding cost.
- Calculate reorder quantity.
- Calculate expected reorder utility.
- Generate demand candidate values.

### All

- Update `docs/assumptions_register.md`.
- Verify expected-value formulas.
- Verify `net_expected_value_inr = expected_benefit_inr - expected_cost_inr`.

- Deliverable: each pipeline produces validated expected-value candidate data.

## Phase 6 — Pipeline integration & core X-DRA experiment (Weeks 5–6) — Owners: B lead, A integration, C integration

- Generate `churn_candidates.csv`.
- Generate `fraud_candidates.csv`.
- Generate `demand_candidates.csv`.
- Validate all outputs using the shared schema.
- Connect all three outputs to the allocation engine.
- Run joint shared-budget allocation.
- Run independent allocation.
- Compare total expected benefit, cost, net expected value and budget utilization.
- Run the experiment across multiple budget levels.
- Deliverable: first real joint-vs-independent X-DRA experiment.
- Checkpoint: complete end-to-end research demo.

## Phase 7 — Resource-aware model selection (Weeks 6–7) — Owners: C lead, A model analysis, B validation

- Compare model performance against computational requirements.
- Measure training time.
- Measure inference time.
- Compare model complexity and resource requirements.
- Identify practical models for resource-constrained MSMEs.
- Evaluate whether the highest-performing model is also the most practical.
- Prepare cross-domain resource comparison.
- Deliverable: resource-aware model selection analysis.

## Phase 8 — Public-data augmentation (Weeks 7–8) — Owners: C lead, A churn experiment, B fraud experiment

- Identify relevant external/public context data.
- Establish business-data-only baseline.
- Add appropriate external context.
- Re-evaluate model performance.
- Recalculate expected business value.
- Compare downstream allocation before and after augmentation.
- Document positive, neutral or negative results.
- Deliverable: public-data augmentation experiment.

## Phase 9 — Sensitivity analysis, dashboard & explainability (Weeks 8–10) — Owners: C dashboard, A churn sensitivity, B allocation sensitivity

### Sensitivity analysis

- Test churn `P(save | intervention)` at 20%, 30% and 40%.
- Test different retention costs.
- Test fraud investigation costs.
- Test demand stockout costs.
- Test holding costs.
- Test starting inventory assumptions.
- Test multiple total budget levels.
- Determine whether the joint-allocation result remains robust.

### Dashboard

- Budget overview.
- Fraud recommendations.
- Churn recommendations.
- Demand reorder recommendations.
- Allocation by domain.
- Expected benefit, cost and net value.
- Joint vs independent comparison.
- Resource-aware model comparison.
- Sensitivity results.

### Explainability

- Explain why fraud actions were prioritized.
- Explain why customers were prioritized.
- Explain why reorder actions were selected.
- Show expected-value reasoning behind selected actions.

- Deliverable: functional unified dashboard, sensitivity analysis and explainability outputs.

## Phase 10 — Final validation, documentation & demo (Weeks 10–12) — Owners: All

- Run complete system from a clean checkout.
- Run all three pipelines end-to-end.
- Run allocation engine on real outputs.
- Verify reproducibility.
- Finalize experiments and results.
- Finalize assumptions register.
- Finalize README and technical documentation.
- Document limitations and threats to validity.
- Complete literature review and methodology.
- Finalize research findings.
- Prepare final presentation and demo script.
- Prepare individual viva questions.
- Run final CI, tests, linting and formatting checks.
- Deliverable: complete X-DRA system, final report, presentation and reproducible codebase.

## Final Project Flow

Phase 1 → Setup & Common Foundation  
↓  
Phase 2 → Allocation Engine on Synthetic Data  
↓  
Phase 3 → Churn + Fraud + Demand Baselines  
↓  
Phase 4 → Advanced Models & Evaluation  
↓  
Phase 5 → Calibration + Expected Business Value  
↓  
Phase 6 → Joint vs Independent Allocation  
↓  
Phase 7 → Resource-Aware Model Selection  
↓  
Phase 8 → Public-Data Augmentation  
↓  
Phase 9 → Sensitivity + Dashboard + Explainability  
↓  
Phase 10 → Final Validation + Report + Demo

## Definition of Done

- [ ] Churn pipeline complete and evaluated.
- [ ] Fraud pipeline complete and evaluated.
- [ ] Demand pipeline complete and evaluated.
- [ ] All three pipelines produce the common candidate schema.
- [ ] Calibration checked for probabilities used in expected-value calculations.
- [ ] All invented business assumptions documented.
- [ ] Joint allocation works on real pipeline outputs.
- [ ] Independent allocation baseline works.
- [ ] Joint-vs-independent experiment completed.
- [ ] Resource-aware analysis completed.
- [ ] Public-data experiment completed.
- [ ] Sensitivity analysis completed.
- [ ] Dashboard completed.
- [ ] Explainability completed.
- [ ] All tests pass.
- [ ] All members can run the system from a clean checkout.
- [ ] Final report and presentation completed.

## Core Research Question

Does allocating a shared, constrained MSME business budget jointly across fraud investigation, customer retention, and inventory reorder actions produce greater total expected business value than optimizing each domain independently?