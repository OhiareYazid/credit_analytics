"""Loan-level lifetime economics shared by the cut-off, limit and P&L analyses."""
import numpy as np
import pandas as pd

import config as C

MEAN_FIRST_MISS_FRACTION = 1.5 / (1.5 + 3.2)     # mean of Beta(1.5, 3.2)


def balance_path(p, r, n, months):
    """Opening principal balance in each month of an annuity schedule."""
    k = np.arange(months)[None, :]
    bal = p[:, None] * ((1 + r[:, None]) ** n[:, None] - (1 + r[:, None]) ** k) / ((1 + r[:, None]) ** n[:, None] - 1)
    return np.where(k < n[:, None], bal, 0.0)       # balance at start of each month


def lifetime_profit(loans: pd.DataFrame, monthly_rate: np.ndarray) -> pd.DataFrame:
    p = loans["loan_amnt"].to_numpy(float)
    r = monthly_rate
    n = loans["term"].to_numpy()
    bal = balance_path(p, r, n, n.max())
    m = np.arange(n.max())[None, :]
    rows = np.arange(len(p))

    first_miss = np.floor(2 + MEAN_FIRST_MISS_FRACTION * (n - 6)).astype(int)   # MOB of first missed payment
    co_month = first_miss + 4                                                     # charged off at 120+ DPD

    # good loan: full schedule
    int_good = (bal * r[:, None]).sum(axis=1)
    fund_good = (bal * C.COST_OF_FUNDS / 12).sum(axis=1)
    profit_good = int_good - fund_good - C.OPEX_PER_LOAN_MONTH * n - C.ACQUISITION_COST

    # bad loan: pays months 1..first_miss-1, balance frozen until charge-off
    pays = m < (first_miss[:, None] - 1)
    ead = bal[rows, first_miss - 1]
    int_bad = (bal * r[:, None] * pays).sum(axis=1)
    bal_held = np.where(pays, bal, ead[:, None]) * (m < co_month[:, None])
    fund_bad = (bal_held * C.COST_OF_FUNDS / 12).sum(axis=1)
    profit_bad = (int_bad - fund_bad - C.OPEX_PER_LOAN_MONTH * co_month
                  - C.ACQUISITION_COST - C.LGD * ead)

    out = pd.DataFrame({"loan_id": loans["loan_id"].to_numpy(),
                        "profit_if_good": profit_good, "profit_if_bad": profit_bad})
    out["loan_profit"] = np.where(loans["bad_loan"].to_numpy() == 1, profit_bad, profit_good)
    out["breakeven_pd"] = profit_good / (profit_good - profit_bad)
    return out
