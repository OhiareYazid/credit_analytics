"""Step 5 — score cut-off analysis with profit.

For each loan, lifetime profit is built from contractual cash flows:
- good loan: interest over the full schedule − funding cost − servicing − acquisition;
- bad loan:  interest until the first missed payment − funding cost until
  charge-off − servicing − acquisition − LGD × balance at default.
Default timing uses the same "first missed payment" distribution as the
simulation (mean ≈ 32% of the way through the loan).

Two pricing regimes are compared:
- risk_based: each loan's actual LendingClub APR (riskier → dearer);
- flat: one APR for everyone (portfolio average), as on a single-price product.
Under flat pricing the cut-off is where marginal loans stop paying for their
losses; under risk-based pricing price does part of the cut-off's job.

We sweep the score cut-off on the out-of-time sample and compare the
profit-maximising cut-off with a risk-appetite cut-off (approved bad rate ≤
target), and report break-even PD per band.

Caveat for interview: the data contains booked loans only (no rejects), so
"approval rate" here is relative to LendingClub's already-approved population.
Lowering a cut-off below today's would need reject inference.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

import config as C
from db import connect, run_sql
from plotting import style, save
from economics import lifetime_profit

RISK_APPETITE_BAD_RATE = 0.15
def pick(curve: pd.DataFrame) -> pd.DataFrame:
    best = curve.loc[curve.total_profit.idxmax()]
    ok = curve[curve.bad_rate <= RISK_APPETITE_BAD_RATE]
    appetite = ok.loc[ok.cutoff.idxmin()]
    approve_all = curve.loc[curve.cutoff.idxmin()]
    return pd.DataFrame([
        {"strategy": "Approve all (status quo)", **approve_all.to_dict()},
        {"strategy": "Profit-maximising cut-off", **best.to_dict()},
        {"strategy": f"Risk appetite (bad rate <= {RISK_APPETITE_BAD_RATE:.0%})", **appetite.to_dict()},
    ])


def main():
    style()
    con = connect()
    loans = con.sql("SELECT * FROM loans ORDER BY loan_id").df()
    flat_apr = loans.loc[loans.vintage_idx < C.OOT_FROM_VINTAGE, "int_rate"].mean()
    regimes = {
        "risk_based": loans["monthly_rate"].to_numpy(),
        "flat": np.full(len(loans), flat_apr / 100 / 12),
    }
    lp = pd.concat([lifetime_profit(loans, r).assign(pricing=k) for k, r in regimes.items()])
    con.register("lp_df", lp)
    con.execute("CREATE OR REPLACE TABLE loan_profit AS SELECT * FROM lp_df")
    curve = run_sql(con, "04_cutoff_strategy.sql")
    band_econ = con.sql("""
        SELECT p.pricing, s.band, COUNT(*) AS n, AVG(l.bad_loan) AS bad_rate,
               AVG(p.breakeven_pd) AS breakeven_pd,
               SUM(p.loan_profit) AS total_profit, AVG(p.loan_profit) AS profit_per_loan
        FROM scores s JOIN loans l USING (loan_id) JOIN loan_profit p USING (loan_id)
        WHERE s.sample = 'oot' GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    con.close()

    curve = curve[curve.n_approved >= 500].reset_index(drop=True)
    decisions = pd.concat([pick(c).assign(pricing=k) for k, c in curve.groupby("pricing")])
    decisions.round(4).to_csv(C.TAB_DIR / "cutoff_decisions.csv", index=False)
    curve.round(4).to_csv(C.TAB_DIR / "cutoff_curve.csv", index=False)
    band_econ.round(4).to_csv(C.TAB_DIR / "band_unit_economics.csv", index=False)

    # --- figure ---------------------------------------------------------------
    money = FuncFormatter(lambda x, _: f"${x/1e6:.0f}M")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ax = axes[0]
    colours = {"flat": "#3A9E6F", "risk_based": "#7A5BA6"}
    names = {"flat": f"Flat APR {flat_apr:.1f}%", "risk_based": "Risk-based APR (actual)"}
    for k, c in curve.groupby("pricing"):
        ax.plot(c.cutoff, c.total_profit, color=colours[k], lw=2, label=f"Profit — {names[k]}")
        b = c.loc[c.total_profit.idxmax()]
        ax.scatter([b.cutoff], [b.total_profit], color=colours[k], zorder=5)
        ax.annotate(f"max @ {b.cutoff:.0f}", (b.cutoff, b.total_profit), xytext=(6, 6),
                    textcoords="offset points", fontsize=8.5, color=colours[k])
    ax.axhline(0, color="grey", lw=0.8)
    ax.yaxis.set_major_formatter(money)
    ax.set(title="Lifetime profit by score cut-off (OOT sample)", xlabel="Minimum score to approve")
    ax2 = ax.twinx()
    c = curve[curve.pricing == "flat"]
    ax2.plot(c.cutoff, c.approval_rate, color="#2E5AAC", lw=1.2, ls="--", label="Approval rate")
    ax2.plot(c.cutoff, c.bad_rate, color="#B9435B", lw=1.2, ls="--", label="Bad rate of approved")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.8)

    ax = axes[1]
    flat = band_econ[band_econ.pricing == "flat"].set_index("band")
    rbp = band_econ[band_econ.pricing == "risk_based"].set_index("band")
    x = np.arange(len(flat))
    ax.bar(x - 0.27, flat.bad_rate, 0.27, label="Observed bad rate", color="#B9435B")
    ax.bar(x, flat.breakeven_pd, 0.27, label="Break-even PD — flat APR", color="#3A9E6F")
    ax.bar(x + 0.27, rbp.breakeven_pd, 0.27, label="Break-even PD — risk-based APR", color="#7A5BA6")
    ax.set_xticks(x, flat.index)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set(title="A band is worth booking when bad rate < break-even PD", xlabel="Risk band")
    ax.legend(fontsize=7.8)
    save(fig, "05_cutoff_profit")

    cols = ["pricing", "strategy", "cutoff", "approval_rate", "bad_rate", "total_profit", "profit_margin"]
    print(f"flat APR = {flat_apr:.2f}%")
    print(decisions[cols].round(4).to_string(index=False))
    print(band_econ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
