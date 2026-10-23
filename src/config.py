"""Central configuration: paths and business assumptions.

Every number an interviewer might challenge lives here, so assumptions can be
changed in one place and the whole pipeline re-run.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DB_PATH = DATA_PROC / "credit.duckdb"
SQL_DIR = ROOT / "sql"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

DATA_URL = (
    "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
)

SEED = 42

# ---- Simulated performance window -------------------------------------------
N_VINTAGES = 48          # monthly origination cohorts, Sep-2022 … Aug-2026
SNAPSHOT_INDEX = 48      # snapshot month: vintage v is observed for 48 - v months
TIGHTENING_VINTAGE = 38  # credit policy tightened from this cohort (Nov-2025)
OOT_FROM_VINTAGE = 33    # scorecard: cohorts before = development, from here = out-of-time
VINTAGE_START = "2022-09-01"  # label for vintage index 0 → snapshot = Sep-2026 (cosmetic)

# Delinquency states, in order of severity
STATES = ["Current", "1-30", "31-60", "61-90", "91-120", "Charged-off", "Paid-off"]

# ---- Unit economics (annualised unless stated) ------------------------------
# Pricing in the source data is US-dollar consumer lending (~13% APR), so these
# are set for that context. For a Naira overdraft/loan book, swap in local APRs,
# cost of funds and LGD — every downstream analysis reads from here.
COST_OF_FUNDS = 0.04     # annual funding cost on outstanding balance
LGD = 0.75               # loss given default on balance at default (after debt-sale recoveries)
OPEX_PER_LOAN_MONTH = 3.0  # servicing cost per active loan per month (currency units)
ACQUISITION_COST = 50.0  # one-off cost per booked loan (KYC, bureau, marketing)

# ---- Scorecard scaling -------------------------------------------------------
SCORE_BASE = 600         # score at base odds
SCORE_BASE_ODDS = 4.0    # good:bad odds at the base score (≈ portfolio odds)
SCORE_PDO = 40           # points to double the odds

# ---- Affordability / limit policy -------------------------------------------
MAX_DTI_POST_LOAN = 0.40     # total debt service ≤ 40% of gross monthly income
INCOME_HAIRCUT = {           # discount for unverified stated income
    "verified": 1.00,
    "not verified": 0.80,
}
RISK_BAND_MULTIPLIER = {     # share of affordable amount granted, by risk band
    "A": 1.00, "B": 0.90, "C": 0.75, "D": 0.60, "E": 0.40,
}
ABSOLUTE_LIMIT_CAP = 35_000

# ---- A/B test ----------------------------------------------------------------
AB_ALPHA = 0.05
AB_POWER = 0.80
AB_MDE_PP = 0.03            # minimum detectable uplift in cure rate (3 pp)
AB_TRUE_UPLIFT_PP = 0.035   # effect baked into the simulated experiment

# ---- P&L forecast ------------------------------------------------------------
FORECAST_MONTHS = 12
STRESS_PD_MULTIPLIER = 1.5
