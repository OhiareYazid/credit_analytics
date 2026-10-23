"""Step 4 — roll-rate matrices and net flow rates.

- Account- and balance-weighted transition matrices (last 12 months).
- Absorbing Markov chain: from each delinquency bucket, the probability of
  eventually charging off vs. curing to pay-off — what collections uses to value
  a bucket and prioritise effort.
- Net flow rates by calendar month (balance in bucket k this month ÷ balance in
  bucket k-1 last month), and the implied loss rate on current balances.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import config as C
from db import connect, run_sql
from plotting import style, save

DELQ = ["Current", "1-30", "31-60", "61-90", "91-120"]
ABSORB = ["Charged-off", "Paid-off"]
ORDER = DELQ + ABSORB


def matrix(long: pd.DataFrame, value: str) -> pd.DataFrame:
    m = long.pivot_table(index="state_from", columns="state_to", values=value, aggfunc="sum", fill_value=0)
    m = m.reindex(index=DELQ, columns=ORDER, fill_value=0)
    return m.div(m.sum(axis=1), axis=0)


def absorption(P: pd.DataFrame) -> pd.DataFrame:
    Q = P.loc[DELQ, DELQ].to_numpy()
    R = P.loc[DELQ, ABSORB].to_numpy()
    N = np.linalg.inv(np.eye(len(DELQ)) - Q)          # fundamental matrix
    out = pd.DataFrame(N @ R, index=DELQ, columns=["P(charge-off)", "P(pay-off)"])
    out["expected_months_to_resolution"] = N.sum(axis=1)
    return out


def heatmap(ax, M, title):
    im = ax.imshow(M.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=35, ha="right")
    ax.set_yticks(range(M.shape[0]), M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if v >= 0.0005:
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else "black")
    ax.set(title=title, xlabel="State at month end", ylabel="State at month start")
    ax.grid(False)
    return im


def main():
    style()
    con = connect(read_only=True)
    long = run_sql(con, "02_roll_rate_matrix.sql")
    buckets = run_sql(con, "03_bucket_balances.sql")
    entry_by_band = con.sql("""
        SELECT s.band,
               AVG(CASE WHEN p.state = '1-30' THEN 1 ELSE 0 END) AS current_to_1_30
        FROM performance p JOIN scores s USING (loan_id)
        WHERE p.state_from = 'Current'
          AND p.cal_month_idx > (SELECT snapshot_idx FROM meta) - 12
        GROUP BY 1 ORDER BY 1
    """).df()
    con.close()

    P_n = matrix(long, "n_accounts")
    P_b = matrix(long, "balance")
    absorb = absorption(P_n)

    P_n.round(4).to_csv(C.TAB_DIR / "roll_rate_matrix_accounts.csv")
    P_b.round(4).to_csv(C.TAB_DIR / "roll_rate_matrix_balance.csv")
    absorb.round(4).to_csv(C.TAB_DIR / "bucket_absorption_probabilities.csv")
    entry_by_band.round(4).to_csv(C.TAB_DIR / "roll_entry_by_band.csv", index=False)

    # net flow rates
    b = buckets.pivot(index="cal_month_idx", columns="state", values="balance").fillna(0)
    b = b.reindex(columns=["Current", "1-30", "31-60", "61-90", "91-120", "Charged-off"], fill_value=0)
    flows = pd.DataFrame(index=b.index[1:])
    chain = list(b.columns)
    for prev, cur in zip(chain[:-1], chain[1:]):
        flows[f"{prev} → {cur}"] = (b[cur] / b[prev].shift(1)).iloc[1:]
    flows = flows[(flows.index >= 12) & (flows.index <= C.SNAPSHOT_INDEX)]
    flows["implied loss rate (product)"] = flows.prod(axis=1)
    flows.round(5).to_csv(C.TAB_DIR / "net_flow_rates.csv")

    # --- figures -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    heatmap(axes[0], P_n, "Roll-rate matrix — % of accounts (last 12 months)")
    heatmap(axes[1], P_b, "Roll-rate matrix — % of balance (last 12 months)")
    save(fig, "04_roll_rate_matrices")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    absorb["P(charge-off)"].plot.bar(ax=axes[0], rot=0, color="#B9435B")
    for i, v in enumerate(absorb["P(charge-off)"]):
        axes[0].text(i, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.5)
    axes[0].set(title="Eventual charge-off probability by bucket", xlabel="Current bucket")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for c in flows.columns[1:-1]:
        axes[1].plot(flows.index, flows[c], label=c)
    axes[1].set(title="Net flow rates by calendar month (balance)", xlabel="Calendar month index")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].legend(fontsize=7.5)
    save(fig, "04_absorption_and_flows")

    print(P_n.round(3).to_string())
    print(absorb.round(3).to_string())
    print(entry_by_band.round(4).to_string(index=False))
    print(flows.tail(3).round(4).to_string())


if __name__ == "__main__":
    main()
