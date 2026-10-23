"""Step 3 — vintage curves.

Questions answered:
- How do cumulative charge-offs build up by months on book (MOB)?
- Which cohorts are performing worse, and is that because we lent to riskier
  people (mix) or because the same risk grades are behaving worse (performance)?
- Can we read young cohorts early? (ever-30+ DPD at MOB 6 as a leading indicator)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import config as C
from db import connect, run_sql
from plotting import style, save


def main():
    style()
    con = connect(read_only=True)
    vc = run_sql(con, "01_vintage_curves.sql")
    mix = con.sql("""
        SELECT l.vintage_idx, AVG(s.pd) AS expected_pd,
               AVG(CASE WHEN s.band IN ('D','E') THEN 1 ELSE 0 END) AS share_de
        FROM loans l JOIN scores s USING (loan_id) GROUP BY 1 ORDER BY 1
    """).df()
    # band-level curves isolate performance from mix
    # only half-years whose every cohort has reached MOB 12
    last_full_hy = (C.SNAPSHOT_INDEX - 12 + 1) // 6
    band_mob12 = con.sql(f"""
        WITH lb AS (
            SELECT l.loan_id, l.loan_amnt, l.vintage_idx // 6 AS half_year, s.band
            FROM loans l JOIN scores s USING (loan_id)
            WHERE l.vintage_idx // 6 < {last_full_hy}
        )
        SELECT lb.half_year, lb.band,
               SUM(COALESCE(p.co, 0)) / SUM(lb.loan_amnt) AS co_rate_mob12
        FROM lb
        LEFT JOIN (SELECT loan_id, SUM(chargeoff_amt) AS co FROM performance
                   WHERE mob <= 12 GROUP BY 1) p USING (loan_id)
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    con.close()

    vc.to_csv(C.TAB_DIR / "vintage_curves.csv", index=False)

    at = lambda m, col: vc[vc.mob == m].set_index("vintage_idx")[col]
    summary = pd.DataFrame({
        "vintage": vc.groupby("vintage_idx")["vintage"].first(),
        "n_loans": vc.groupby("vintage_idx")["n_loans"].first(),
        "ever30_mob6": at(6, "ever_30plus_rate"),
        "co_mob12": at(12, "cum_co_rate"),
        "co_mob24": at(24, "cum_co_rate"),
    }).join(mix.set_index("vintage_idx"))
    summary.round(4).to_csv(C.TAB_DIR / "vintage_summary.csv")

    # --- Figure 1: the vintage curves themselves ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    cmap = plt.get_cmap("viridis")
    shown = range(0, C.N_VINTAGES, 4)
    for v in shown:
        d = vc[vc.vintage_idx == v]
        col = cmap(v / (C.N_VINTAGES - 1))
        axes[0].plot(d.mob, d.cum_co_rate, color=col, lw=1.6, label=d.vintage.iloc[0])
        axes[1].plot(d.mob[d.mob <= 18], d.ever_30plus_rate[d.mob <= 18], color=col, lw=1.6)
    axes[0].set(title="Cumulative charge-off rate (% of originated amount)", xlabel="Months on book")
    axes[1].set(title="Ever 30+ DPD (% of loans) — early read", xlabel="Months on book")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].legend(title="Vintage", ncol=2, fontsize=7.5)
    save(fig, "03_vintage_curves")

    # --- Figure 2: mix vs performance -----------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.2))
    s = summary.dropna(subset=["ever30_mob6"])
    ax.bar(s.index, s["ever30_mob6"], color="#2E5AAC", alpha=0.85, label="Ever 30+ DPD at MOB 6 (observed)")
    ax2 = ax.twinx()
    ax2.plot(s.index, s["share_de"], color="#E07A2D", marker="o", ms=3, label="Share of bands D–E booked")
    ax2.grid(False)
    ax.axvline(C.TIGHTENING_VINTAGE - 0.5, color="#B9435B", ls="--", lw=1)
    ax.text(C.TIGHTENING_VINTAGE - 0.3, ax.get_ylim()[1] * 0.92, "policy\ntightening", color="#B9435B", fontsize=8.5)
    ax.set_xticks(s.index[::4], s["vintage"].iloc[::4], rotation=45, ha="right")
    ax.set(title="Early delinquency tracks the risk mix — deterioration is largely a mix effect", xlabel="Vintage")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", ncol=2)
    save(fig, "03_vintage_mix_vs_performance")

    # --- Figure 3: within-band CO at MOB 12 is flat → performance stable -------
    fig, ax = plt.subplots(figsize=(7.5, 4))
    piv = band_mob12.pivot(index="half_year", columns="band", values="co_rate_mob12")[list("ABCDE")]
    start = pd.Timestamp(C.VINTAGE_START)
    lab = lambda i: (start + pd.DateOffset(months=int(i))).strftime("%b%y")
    labels = [f"{lab(h * 6)}–{lab(h * 6 + 5)}" for h in piv.index]
    piv.index = labels
    piv.plot(ax=ax, marker="o")
    ax.set(title="Charge-off by MOB 12 within each risk band", xlabel="Origination half-year", ylabel="% of originated amount")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(title="Band", ncol=5)
    save(fig, "03_vintage_by_band")

    print(summary[["vintage", "ever30_mob6", "co_mob12", "co_mob24", "share_de"]].iloc[::3].round(4).to_string())
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
