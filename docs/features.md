# X-DRA — Project Features

## 1. Project Overview

X-DRA (Cross-Domain Resource-Constrained Decision Allocation) is an AI-driven decision allocation system for MSMEs that combines three business decision domains:

- Fraud investigation
- Customer retention
- Inventory reordering

Each domain independently produces candidate actions with an estimated business benefit and cost.

X-DRA then treats all three domains as competitors for a single shared business budget and determines how the available budget should be allocated to maximize total expected business value.

The core research contribution is the comparison between:

- Independent per-domain budget optimization
- Joint cross-domain shared-budget optimization

---

## 2. Fraud Detection & Investigation

### Dataset

- Credit Card Fraud Detection dataset
- Time-based train/test splitting
- Highly imbalanced fraud classification

### Prediction Features

- Fraud probability prediction
- Fraud risk scoring
- Cost-sensitive fraud ranking
- Expected financial loss estimation
- High-risk transaction identification

### Models

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM (if used)
- Isolation Forest as an optional unsupervised comparison

### Evaluation

- PR-AUC
- ROC-AUC
- Recall
- False Positive Rate
- Recall at selected threshold
- Operating threshold constrained to ≤5% FPR
- Precision-recall curve
- ROC curve
- Brier score
- Calibration curve

### Business Decision

For each transaction:

- Fraud probability
- Expected financial loss
- Investigation cost
- Expected investigation benefit
- Net expected value

### Output

Ranked fraud investigation candidates that can compete with churn and demand actions for the shared budget.

---

## 3. Customer Churn & Retention

### Dataset

- IBM Telco Customer Churn dataset
- Stratified 80/20 train/test split

### Prediction Features

- Customer churn probability
- Customer risk ranking
- Customer segmentation based on churn risk
- CLV-based customer prioritization
- Expected retention value

### Models

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Optuna-based hyperparameter tuning

### Evaluation

- Accuracy
- Majority-class baseline
- ROC-AUC
- PR-AUC
- Brier score
- Calibration curve

### Business Decision

For each customer:

- Churn probability
- CLV proxy
- `P(save | intervention)`
- Retention intervention cost
- Expected retention benefit
- Net expected retention value

The system prioritizes customers based on expected business value rather than churn probability alone.

### Output

Ranked customer-retention candidates that can compete with fraud investigations and demand reorders for the shared budget.

---

## 4. Demand Forecasting & Inventory Decisions

### Dataset

- M5 Walmart Forecasting dataset
- Store-department level aggregation
- 2–3 selected stores and their departments
- Last 28 days used as the test period

### Forecasting Features

- Historical demand
- 7-day lag
- 14-day lag
- 28-day lag
- Rolling 7-day demand
- Rolling 28-day demand
- Day-of-week effects
- SNAP indicators
- Event indicators
- Store information
- Department information

### Models

- Naive forecast
- Moving Average
- Exponential Smoothing / Holt-Winters
- XGBoost
- LightGBM
- ARIMA as an optional comparison
- Chronos as an optional zero-shot comparison

### Evaluation

- WAPE
- MAE
- RMSE
- Forecast bias
- Comparison against Naive baseline
- Comparison against Moving Average baseline

### Inventory Decision

For each store-department candidate:

- Forecast demand
- Current/simulated inventory
- Expected shortage
- Reorder quantity
- Stockout cost
- Holding cost
- Expected stockout loss avoided
- Expected reorder benefit
- Net expected reorder value

### Output

Ranked inventory reorder candidates that can compete with fraud and churn actions for the shared budget.

---

## 5. Common Candidate Representation

All three domain pipelines produce a common output format.

Every candidate contains:

- `candidate_id`
- `domain`
- `expected_benefit_inr`
- `expected_cost_inr`
- `net_expected_value_inr`

The three domains therefore become directly comparable at the allocation layer.

### Domains

- `fraud`
- `churn`
- `demand`

### Example

```text
candidate_id: transaction_123
domain: fraud
expected_benefit_inr: 18000
expected_cost_inr: 200
net_expected_value_inr: 17800