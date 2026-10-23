"""Step 1 — download, clean, and build the loan + monthly performance tables.

The public LendingClub extract (via H2O) holds application data and the final
outcome (`bad_loan`) for ~164k loans, but no origination dates and no monthly
payment history. Vintage curves and roll-rate matrices need both, so this
script:

1. assigns each loan to one of 48 monthly origination cohorts, with a
   deliberate risk-mix story (growth push that loosens the mix, then a policy
   tightening in the last 10 cohorts);
2. simulates a month-by-month delinquency path for every loan that is
   *consistent with its real outcome*: bad loans walk the 1-30 → … → 91-120 →
   charge-off path; good loans may have transient arrears that cure, prepay,
   or run to maturity;
3. censors everything at a common snapshot date, so young vintages are only
   partly observed — exactly like a live book.

Output: DuckDB database with `loans` and `performance` tables.
"""
import numpy as np
import pandas as pd
import duckdb

import config as C

rng = np.random.default_rng(C.SEED)


def load_raw() -> pd.DataFrame:
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = C.DATA_RAW / "loan.csv"
    if not path.exists():
        print(f"Downloading {C.DATA_URL} …")
        pd.read_csv(C.DATA_URL).to_csv(path, index=False)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "loan_id", np.arange(1, len(df) + 1))
    df["term"] = df["term"].str.extract(r"(\d+)").astype(int)
    df["emp_length_missing"] = df["emp_length"].isna().astype(int)
    for col in ["emp_length", "annual_inc", "revol_util", "delinq_2yrs",
                "total_acc", "longest_credit_length"]:
        df[col] = df[col].fillna(df[col].median())
    df["revol_util"] = df["revol_util"].clip(0, 150)
    df["annual_inc"] = df["annual_inc"].clip(lower=4_000)
    df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})
    df["monthly_rate"] = df["int_rate"] / 100 / 12
    r, n, p = df["monthly_rate"], df["term"], df["loan_amnt"]
    df["instalment"] = p * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return df


def assign_vintages(df: pd.DataFrame) -> np.ndarray:
    """Sample a vintage per loan: volume grows ~3% a month, the riskier half of
    applicants becomes steadily over-represented (growth push), then
    under-represented after the policy tightening."""
    v = np.arange(C.N_VINTAGES)
    growth = np.exp(0.03 * v)
    t = C.TIGHTENING_VINTAGE
    shift = np.minimum(v, t) / t - np.maximum(v - t, 0) / 9 * 1.3
    risk_pct = df["int_rate"].rank(pct=True).to_numpy()
    logits = 2.2 * (risk_pct[:, None] - 0.5) * shift[None, :]
    w = growth[None, :] * np.exp(logits)
    cdf = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
    u = rng.random(len(df))[:, None]
    return (u > cdf).sum(axis=1)


def scheduled_balance(p, r, n, k):
    """Outstanding principal after k scheduled payments on an annuity loan."""
    return p * ((1 + r) ** n - (1 + r) ** k) / ((1 + r) ** n - 1)


def simulate_performance(df: pd.DataFrame) -> pd.DataFrame:
    n_loans = len(df)
    term = df["term"].to_numpy()
    bad = df["bad_loan"].to_numpy() == 1
    p = df["loan_amnt"].to_numpy(float)
    r = df["monthly_rate"].to_numpy()
    vintage = df["vintage_idx"].to_numpy()
    obs_max = np.minimum(term, C.SNAPSHOT_INDEX - vintage)

    risk = df["int_rate"].rank(pct=True).to_numpy()
    # transient arrears hazard (per month) — riskier and bad loans miss more often
    hazard = 0.004 + 0.02 * risk + np.where(bad, 0.015, 0.0)
    prepay_hazard = np.where(bad, 0.0, 0.009 + 0.006 * (1 - risk))

    # month of first missed payment on the terminal path (bad loans only);
    # skewed early, as unsecured charge-offs peak in the first 18 months
    first_miss = 2 + np.floor(rng.beta(1.5, 3.2, n_loans) * (term - 6)).astype(int)
    first_miss = np.where(bad, first_miss, 10_000)

    state = np.zeros(n_loans, dtype=int)       # index into C.STATES
    paid_k = np.zeros(n_loans, dtype=int)       # scheduled payments made
    alive = np.ones(n_loans, dtype=bool)        # not yet absorbed
    rows = []

    for mob in range(1, term.max() + 1):
        observed = alive & (mob <= obs_max)
        if not observed.any():
            break
        prev = state.copy()
        u = rng.random(n_loans)
        new = prev.copy()

        # --- transient (non-terminal) dynamics -------------------------------
        cur = prev == 0
        enter = cur & (u < hazard)
        new[enter] = 1
        # transient arrears cure or roll; off the terminal path nobody rolls
        # past 61-90 — a 61-90 account off the path pays up the next month
        cure_p = {1: 0.62, 2: 0.48, 3: 1.0}
        for s, cp in cure_p.items():
            m = prev == s
            cures = m & (u < cp)
            new[cures] = 0
            new[m & ~cures] = min(s + 1, 3)
        # good loans prepaying
        prepay = (new == 0) & (rng.random(n_loans) < prepay_hazard)
        new[prepay] = 6

        # --- terminal path for bad loans (overrides transient dynamics) ------
        k = mob - first_miss            # 0 → 1-30, 1 → 31-60, …, 4 → charge-off
        on_path = bad & (k >= 0)
        new[on_path] = np.minimum(k[on_path] + 1, 5)

        # --- maturity --------------------------------------------------------
        mature = (mob >= term) & (new == 0)
        new[mature] = 6

        # --- cash flows ------------------------------------------------------
        # a paying loan settles every month it owes (a cure is a full catch-up);
        # interest is approximated on the opening balance
        pays = (new == 0) | (new == 6)
        months_settled = np.where(pays, np.maximum(mob - paid_k, 1), 0)
        bal_open = scheduled_balance(p, r, term, paid_k)
        interest = bal_open * r * months_settled
        paid_k = np.where(new == 0, mob, paid_k)
        paid_k = np.where(new == 6, term, paid_k)          # prepay / maturity
        bal_close = np.where(new == 6, 0.0, scheduled_balance(p, r, term, paid_k))
        co_amt = np.where((new == 5) & (prev != 5), bal_close, 0.0)

        idx = np.flatnonzero(observed)
        rows.append(pd.DataFrame({
            "loan_id": df["loan_id"].to_numpy()[idx],
            "vintage_idx": vintage[idx],
            "mob": mob,
            "cal_month_idx": vintage[idx] + mob,
            "state_from": prev[idx],
            "state": new[idx],
            "balance_open": bal_open[idx].round(2),
            "balance_close": bal_close[idx].round(2),
            "interest_paid": interest[idx].round(2),
            "chargeoff_amt": co_amt[idx].round(2),
        }))
        state = new
        alive &= ~np.isin(state, [5, 6])

    perf = pd.concat(rows, ignore_index=True)
    perf["state_from"] = perf["state_from"].map(dict(enumerate(C.STATES)))
    perf["state"] = perf["state"].map(dict(enumerate(C.STATES)))
    return perf


def main():
    raw = load_raw()
    loans = clean(raw)
    loans["vintage_idx"] = assign_vintages(loans)
    start = pd.Timestamp(C.VINTAGE_START)
    loans["vintage"] = loans["vintage_idx"].apply(lambda i: (start + pd.DateOffset(months=int(i))).strftime("%Y-%m"))
    perf = simulate_performance(loans)

    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    if C.DB_PATH.exists():
        C.DB_PATH.unlink()
    con = duckdb.connect(str(C.DB_PATH))
    con.register("loans_df", loans)
    con.register("perf_df", perf)
    con.execute("CREATE TABLE loans AS SELECT * FROM loans_df")
    con.execute("CREATE TABLE performance AS SELECT * FROM perf_df")
    con.execute(f"CREATE TABLE meta AS SELECT {C.SNAPSHOT_INDEX} AS snapshot_idx, DATE '{C.VINTAGE_START}' AS vintage_start")
    n_l, n_p = con.execute("SELECT (SELECT COUNT(*) FROM loans), (SELECT COUNT(*) FROM performance)").fetchone()
    con.close()
    print(f"loans: {n_l:,} rows | performance: {n_p:,} loan-months → {C.DB_PATH}")


if __name__ == "__main__":
    main()
