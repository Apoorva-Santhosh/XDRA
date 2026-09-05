# X-DRA — Cross-Domain Resource-Constrained Decision Allocation for MSMEs

**Dayananda Sagar College of Engineering**
Department of CSE (IoT, Cybersecurity including Blockchain Technology)

| Name | USN |
|---|---|
| Aadya Jha | 1DS23IC001 |
| Apoorva Santhosh | 1DS23IC005 |
| Gogoi Sanskriti Dhrubajit | 1DS23IC011 |

---

## 1. What this project is

X-DRA tests whether treating **fraud investigation, customer retention, and inventory reorder decisions as competitors for one shared MSME budget** — rather than three independently optimized budgets — captures more total expected business value.

Three prediction pipelines (fraud, churn, demand) each generate `(candidate_action, expected_benefit, expected_cost)` rows. A shared-budget allocation engine then decides which candidates, across *all three domains*, get funded under one constrained budget — and we measure whether that joint decision beats optimizing each domain separately.

Full research plan, literature base, dataset specification, and target metrics: see [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) (the 30-section plan document — add it here before Phase 2 begins).

**Read [`GUIDELINES.md`](GUIDELINES.md) before writing any pipeline code.** It defines the exact datasets, preprocessing rules, output schema, and per-teammate responsibilities every pipeline must follow — the allocation engine only works if all three pipelines emit the same schema.

## 2. Repository structure

```
x-dra/
├── README.md                      # you are here
├── GUIDELINES.md                  # datasets, rules, schema contract, teammate assignments
├── docs/
│   ├── PROJECT_PLAN.md            # full 30-section research plan
│   ├── assumptions_register.md    # every invented business-cost number, logged and versioned
│   └── BRANCH_PROTECTION_SETUP.md # one-time GitHub settings walkthrough (not a code file)
├── src/
│   ├── common/                    # shared schema validation, config loading, utilities
│   ├── fraud/                     # Task 1 — fraud detection pipeline
│   ├── churn/                     # Task 2 — churn prediction pipeline
│   ├── demand/                    # Task 3 — demand forecasting pipeline
│   └── allocation/                # Core contribution — shared-budget allocation engine
├── tests/
│   ├── fraud/ churn/ demand/ allocation/   # unit tests, mirroring src/
├── notebooks/                     # exploratory work only — nothing here feeds the pipeline directly
├── data/                          # gitignored — raw datasets live locally, never committed
└── .github/
    ├── workflows/ci.yml           # required checks: lint + tests must pass before merge
    ├── pull_request_template.md
    └── CODEOWNERS
```

## 3. Setup

```bash
git clone <repo-url>
cd x-dra
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No GPU required anywhere in this project — every pipeline is scoped to run on a standard laptop or Google Colab's free CPU tier (see Section 30 of the project plan for the reasoning).

Place raw datasets under `data/<domain>/` locally (never commit them — see `.gitignore`). Sources:
- Fraud: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Churn: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Demand: [M5 Forecasting — Walmart](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

## 4. Running a pipeline

Each domain pipeline is self-contained and runnable independently:

```bash
python -m src.fraud.train
python -m src.churn.train
python -m src.demand.train
```

Each writes its output candidate table to `outputs/<domain>_candidates.csv`, in the shared schema defined in `GUIDELINES.md` §4. Once all three exist:

```bash
python -m src.allocation.run --budget 50000
```

This runs the joint-vs-independent comparison (the project's central experiment) and writes results to `outputs/allocation_comparison.csv`.

## 5. Branching & PR workflow

- Never commit directly to `main`. Branch per task: `fraud/xgboost-tuning`, `churn/calibration-fix`, `allocation/sensitivity-analysis`, etc.
- Every PR must pass CI (lint + tests) **and** get at least one teammate approval before it can be merged — both are enforced by GitHub branch protection, not optional. See `docs/BRANCH_PROTECTION_SETUP.md` for the one-time setup (done once, by whoever has admin rights on the repo).
- Fill out the PR template — it asks for what changed, what you tested, and whether the output schema (§4 of `GUIDELINES.md`) still holds.
- If CI is red, the merge button is disabled — don't ask for a manual override; fix the failing check.

## 6. Where things stand

Track phase progress here as the project moves forward (update this section — don't let it go stale):

- [ ] Phase 0 — Environment & repo setup
- [ ] Phase 1 — Allocation engine validated on synthetic data
- [ ] Phase 2 — Three domain pipelines (see `GUIDELINES.md` for assignments)
- [ ] Phase 3 — Real pipeline outputs wired into allocation engine
- [ ] Phase 4 — Resource-aware model selection layer
- [ ] Phase 5 — Public-data augmentation experiment
- [ ] Phase 6 — Sensitivity analysis
- [ ] Phase 7 — Dashboard
- [ ] Phase 8 — Explainability + writing
