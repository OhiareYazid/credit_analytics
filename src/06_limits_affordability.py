"""Step 6 — credit limit and affordability rules.

Limit = min(requested, risk-adjusted affordable amount, absolute cap), where

    assessed income   = stated monthly income × haircut (unverified income ×0.8)
    existing service  = DTI × gross monthly income (debt already serviced)
    headroom          = 40% × assessed income − existing service
    affordable amount = PV of `headroom` over the loan term at the loan's rate
    risk-adjusted     = affordable amount × band multiplier (A 100% … E 40%)

Analysis:
1. Does "stretch" (requested ÷ affordable) predict default *beyond* the score?
   If yes, an affordability rule adds information — not just conservatism.
2. What does the policy do to exposure, expected losses and profit?

Simplifying assumption (stated in README): a borrower's default behaviour does
not change when the loan is smaller. In practice smaller loans default less,
so the benefit shown here is conservative.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

import config as C
from db import connect
from plotting import style, save
from economics import lifetime_profit

MIN_LOAN = 1_000
STRETCH_BINS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, np.inf]
STRETCH_LABELS = ["<25%", "25–50%", "50–75%", "75–100%", "100–150%", ">150%"]


def apply_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gross_m = df["annual_inc"] / 12
    haircut = df["verification_status"].map(C.INCOME_HAIRCUT).fillna(0.8)
    assessed = gross_m * haircut
    existing = df["dti"] / 100 * gross_m
    df["headroom"] = (C.MAX_DTI_POST_LOAN * assessed - existing).clip(lower=0)
    r, n = df["monthly_rate"], df["term"]
    annuity_factor = (1 - (1 + r) ** -n) / r
    df["affordable_amt"] = df["headroom"] * annuity_factor
    df["stretch"] = df["loan_amnt"] / df["affordable_amt"].replace(0, np.nan)
    df["stretch"] = df["stretch"].fillna(np.inf)
    df["risk_limit"] = (df["affordable_amt"] * df["band"].map(C.RISK_BAND_MULTIPLIER)).clip(upper=C.ABSOLUTE_LIMIT_CAP)
    df["assigned_amt"] = np.minimum(df["loan_amnt"], np.floor(df["risk_limit"] / 100) * 100)
    df["outcome"] = np.select(
        [df["assigned_amt"] < MIN_LOAN, df["assigned_amt"] < df["loan_amnt"]],
        ["Declined (no headroom)", "Limit reduced"], "Approved as requested")
    df.loc[df["outcome"] == "Declined (no headroom)", "assigned_amt"] = 0
    return df


def main():
    style()
    con = connect(read_only=True)
    df = con.sql("""
        SELECT l.*, s.band, s.pd, s.sample FROM loans l JOIN scores s USING (loan_id)
        WHERE s.sample = 'oot' ORDER BY loan_id
    """).df()
    # same flat APR as step 5: development-sample average
    flat_apr = con.sql(f"SELECT AVG(int_rate) FROM loans WHERE vintage_idx < {C.OOT_FROM_VINTAGE}").fetchall()[0][0]
    con.close()

    df = apply_policy(df)
    df["stretch_bucket"] = pd.cut(df["stretch"], STRETCH_BINS, labels=STRETCH_LABELS, right=False)

    # 1) does stretch add information beyond the score?
    by_stretch = df.groupby("stretch_bucket", observed=True).agg(
        n=("loan_id", "size"), bad_rate=("bad_loan", "mean"), avg_pd=("pd", "mean")).reset_index()
    by_stretch["bad_minus_score_pd"] = by_stretch["bad_rate"] - by_stretch["avg_pd"]
    within_band = df.pivot_table(index="stretch_bucket", columns="band", values="bad_loan",
                                 aggfunc="mean", observed=True)[list("ABCDE")]

    # 2) portfolio impact, under both pricing regimes used in step 5
    booked = (df["outcome"] != "Declined (no headroom)").to_numpy()
    regimes = {"risk_based": df["monthly_rate"].to_numpy(),
               "flat": np.full(len(df), flat_apr / 100 / 12)}
    policy_df = df.assign(loan_amnt=df["assigned_amt"].clip(lower=1))
    rows = []
    for pricing, rate in regimes.items():
        base = lifetime_profit(df, rate)["loan_profit"].to_numpy()
        pol = np.where(booked, lifetime_profit(policy_df, rate)["loan_profit"].to_numpy(), 0.0)
        for outcome, m in [*df.groupby("outcome").groups.items(), ("TOTAL", df.index)]:
            idx = df.index.get_indexer(m)
            rows.append({"pricing": pricing, "outcome": outcome, "n": len(idx),
                         "share_of_loans": len(idx) / len(df),
                         "bad_rate": df["bad_loan"].iloc[idx].mean(),
                         "requested": df["loan_amnt"].iloc[idx].sum(),
                         "assigned": df["assigned_amt"].iloc[idx].sum(),
                         "profit_base": base[idx].sum(), "profit_policy": pol[idx].sum()})
    impact = pd.DataFrame(rows)
    impact["exposure_change"] = impact["assigned"] / impact["requested"] - 1
    impact["profit_change"] = impact["profit_policy"] - impact["profit_base"]
    # loss at default scales with the amount lent (same timing assumptions as economics.py)
    ead_frac = 0.75
    losses = {"expected_loss_requested": (df["bad_loan"] * C.LGD * ead_frac * df["loan_amnt"]).sum(),
              "expected_loss_policy": (df["bad_loan"] * C.LGD * ead_frac * df["assigned_amt"]).sum()}

    by_stretch.round(4).to_csv(C.TAB_DIR / "affordability_stretch.csv", index=False)
    within_band.round(4).to_csv(C.TAB_DIR / "affordability_stretch_within_band.csv")
    impact.round(2).to_csv(C.TAB_DIR / "limit_policy_impact.csv", index=False)
    pd.Series(losses).round(0).to_csv(C.TAB_DIR / "limit_policy_losses.csv")

    # --- figures ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    for band in within_band.columns:
        ax.plot(within_band.index.astype(str), within_band[band], marker="o", label=f"Band {band}")
    ax.set(title="Bad rate rises with stretch within every risk band",
           xlabel="Requested amount ÷ affordable amount", ylabel="Bad rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(ncol=2)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    imp = impact[(impact.outcome != "TOTAL") & (impact.pricing == "flat")].set_index("outcome")
    money = FuncFormatter(lambda x, _: f"${x/1e6:.0f}M")
    x = np.arange(len(imp))
    ax.bar(x - 0.2, imp["requested"], 0.4, label="Requested", color="#8C8C8C")
    ax.bar(x + 0.2, imp["assigned"], 0.4, label="Assigned under policy", color="#2E5AAC")
    for i, (o, row) in enumerate(imp.iterrows()):
        ax.text(i, max(row.requested, row.assigned) * 1.02,
                f"{row.share_of_loans:.0%} of loans\nbad rate {row.bad_rate:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, imp.index, fontsize=8.5)
    ax.yaxis.set_major_formatter(money)
    ax.set_ylim(0, imp[["requested", "assigned"]].to_numpy().max() * 1.22)
    ax.set(title="Exposure: requested vs assigned under the limit policy")
    ax.legend(loc="upper right")
    save(fig, "06_limits_affordability")

    print(by_stretch.round(4).to_string(index=False))
    print(within_band.round(3).to_string())
    print(impact[impact.outcome == "TOTAL"].round(3).to_string(index=False))
    print(impact[impact.pricing == "flat"].round(3).to_string(index=False))
    print(losses)


if __name__ == "__main__":
    main()
