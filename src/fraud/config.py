"""Central configuration for the fraud pipeline.

Every business-cost number here is an assumption, not something the dataset
provides. Each one must also be logged in docs/assumptions_register.md.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "fraud" / "creditcard.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"

DOMAIN = "fraud"
SEED = 42

# --- Splitting (GUIDELINES.md section 3: time-based only, never random) ---------------
# The last TEST_FRAC of transactions (by Time) is the test window. The remaining earlier
# part is split again by Time: the final VAL_FRAC_OF_TRAIN of it becomes validation
# (used for model selection, calibration and threshold choice), the rest is train.
TEST_FRAC = 0.20
VAL_FRAC_OF_TRAIN = 0.20

# --- Operating point (GUIDELINES.md section 5) -----------------------------------------
TARGET_FPR = 0.05

# --- Business assumptions (log every change in docs/assumptions_register.md) ----------
INVESTIGATION_COST_INR = 200.0
# The ULB dataset's Amount column is in the cardholders' currency (European cardholders),
# not rupees. 1.0 treats Amount as rupee-equivalent units. Whatever convention is chosen
# MUST match the churn and demand pipelines, otherwise the shared-budget allocator
# compares incompatible currencies.
AMOUNT_TO_INR = 1.0

# --- Dataset sanity checks (ULB credit card fraud dataset) ----------------------------
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492

# --- Columns ----------------------------------------------------------------------------
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_REQUIRED_COLUMNS = ["Time", *PCA_COLUMNS, "Amount", "Class"]
# Raw `Time` is deliberately NOT a feature: under a time-based split every test value is
# larger than any training value, so a model keying on it cannot generalise. We use the
# hour-of-day instead, which repeats across the two days covered by the data.
FEATURE_COLUMNS = [*PCA_COLUMNS, "amount_scaled", "hour_of_day"]

DEFAULT_MODELS = ("logreg", "random_forest", "xgboost")
DEFAULT_TRIALS = 30
