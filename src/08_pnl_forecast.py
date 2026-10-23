"""Step 8 — simple portfolio P&L forecast (12 months), with a backtest.

Engine (cohort / vintage-curve method):
1. From history, learn per-MOB curves for each term (36/60), all as a share of
   the amount originated: outstanding balance, interest collected, charge-offs,
   and share of loans still active.
2. Risk-adjust each cohort's charge-off curve by its average scorecard PD
   relative to the PD of the loans the curves were learned on.
3. Forecast originations with Holt's linear-trend exponential smoothing on log
   monthly volume; new cohorts take the term mix and PD of the last 6 cohorts.
4. Roll every cohort (existing + new) forward and assemble the P&L:
       interest income − funding cost − net credit losses (charge-offs × LGD)
       − servicing opex − acquisition cost = net contribution
5. Backtest: learn curves on data up to month 36, forecast months 37–48 with
   actual originations, compare with actuals. The backtest bias on interest and
   credit losses becomes an explicit calibration overlay on the forward view
   (the score's PD only partly captures the riskier mix of recent cohorts).
6. Scenarios: base, and stress (charge-offs × 1.5).

Not modelled (call out in interview): IFRS 9 provisioning — a bank P&L books
expected-loss impairment ahead of charge-off; here losses are recognised at
charge-off, which is closer to a cash view.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import config as C
from db import connect
from plotting import style, save

MAX_MOB = 60
LINES = ["interest_income", "funding_cost", "net_credit_loss", "opex", "acquisition", "net_contribution"]


def month_label(idx):
    return (pd.Timestamp(C.VINTAGE_START) + pd.DateOffset(months=int(idx))).strftime("%b-%y")


def actual_pnl(con) -> pd.DataFrame:
    df = con.sql(f"""
        SELECT cal_month_idx AS m,
               SUM(interest_paid)                                   AS interest_income,
               SUM(balance_open) * {C.COST_OF_FUNDS} / 12           AS funding_cost,
               SUM(chargeoff_amt) * {C.LGD}                         AS net_credit_loss,
               COUNT(*) * {C.OPEX_PER_LOAN_MONTH}                   AS opex,
               SUM(CASE WHEN mob = 1 THEN 1 ELSE 0 END) * {C.ACQUISITION_COST} AS acquisition
        FROM performance GROUP BY 1 ORDER BY 1
    """).df().set_index("m")
    df["net_contribution"] = df.interest_income - df.funding_cost - df.net_credit_loss - df.opex - df.acquisition
    return df


def learn_curves(con, upto: int) -> tuple[dict, pd.DataFrame]:
    """Per-term, per-MOB curves as a share of originated amount (or count)."""
    raw = con.sql(f"""
        WITH obs AS (
            SELECT l.term, l.vintage_idx, p.mob,
                   SUM(CASE WHEN p.state NOT IN ('Charged-off','Paid-off') THEN p.balance_close ELSE 0 END) AS bal,
                   SUM(p.interest_paid) AS int_, SUM(p.chargeoff_amt) AS co,
                   SUM(CASE WHEN p.state NOT IN ('Charged-off','Paid-off') THEN 1 ELSE 0 END) AS active
            FROM performance p JOIN loans l USING (loan_id)
            WHERE p.cal_month_idx <= {upto}
            GROUP BY 1, 2, 3
        ),
        orig AS (
            SELECT term, vintage_idx, SUM(loan_amnt) AS amt, COUNT(*) AS n
            FROM loans WHERE vintage_idx < {upto} GROUP BY 1, 2
        )
        SELECT o.term, obs.mob, SUM(o.amt) AS amt, SUM(o.n) AS n,
               SUM(obs.bal) AS bal, SUM(obs.int_) AS int_, SUM(obs.co) AS co, SUM(obs.active) AS active
        FROM orig o JOIN obs USING (term, vintage_idx)
        WHERE obs.mob <= {upto} - o.vintage_idx
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    pd_base = con.sql(f"""
        SELECT l.term, SUM(s.pd * l.loan_amnt) / SUM(l.loan_amnt) AS pd
        FROM loans l JOIN scores s USING (loan_id) WHERE l.vintage_idx < {upto} GROUP BY 1
    """).df().set_index("term")["pd"]

    curves = {}
    for term, g in raw.groupby("term"):
        g = g.set_index("mob").reindex(range(1, MAX_MOB + 1))
        c = pd.DataFrame({
            "bal": g.bal / g.amt, "int": g.int_ / g.amt, "co": g.co / g.amt, "active": g.active / g.n,
        })
        # smooth the noisy tail (few cohorts reach high MOB)
        c = c.rolling(3, center=True, min_periods=1).mean()
        last = c["bal"].last_valid_index()
        if last < term:            # extend unobserved MOBs: balance runs off linearly to 0 at maturity
            yield_ = (c.loc[last - 5:last, "int"] / c.loc[last - 5:last, "bal"]).mean()
            co_rate = (c.loc[last - 5:last, "co"] / c.loc[last - 5:last, "bal"]).mean()
            act_rate = (c.loc[last, "active"] / c.loc[last, "bal"])
            for m in range(last + 1, term + 1):
                b = c.at[last, "bal"] * (term - m) / (term - last)
                c.loc[m] = [b, b * yield_, b * co_rate, b * act_rate]
        c.loc[c.index > term] = 0.0
        curves[term] = c.fillna(0.0)
    return curves, pd_base


def project(cohorts: pd.DataFrame, curves: dict, pd_base: pd.Series, months: range,
            co_multiplier: float = 1.0, int_multiplier: float = 1.0) -> pd.DataFrame:
    """cohorts: vintage_idx, term, amt, n, pd. Returns P&L lines by calendar month."""
    out = []
    for m in months:
        mob = m - cohorts.vintage_idx
        live = (mob >= 1) & (mob <= MAX_MOB)
        rows = cohorts[live]
        mob = mob[live]
        bal_open = int_ = co = active = 0.0
        new_n = cohorts.loc[cohorts.vintage_idx == m - 1, "n"].sum()
        for term, c in curves.items():
            sel = rows.term == term
            if not sel.any():
                continue
            r, k = rows[sel], mob[sel].to_numpy()
            risk = r.pd.to_numpy() / pd_base[term]
            prev_bal = np.where(k > 1, c["bal"].to_numpy()[np.clip(k - 2, 0, None)], 1.0)
            bal_open += (r.amt.to_numpy() * prev_bal).sum()
            int_ += (r.amt.to_numpy() * c["int"].to_numpy()[k - 1]).sum()
            co += (r.amt.to_numpy() * c["co"].to_numpy()[k - 1] * risk * co_multiplier).sum()
            active += (r.n.to_numpy() * c["active"].to_numpy()[k - 1]).sum()
        out.append({"m": m, "interest_income": int_ * int_multiplier, "funding_cost": bal_open * C.COST_OF_FUNDS / 12,
                    "net_credit_loss": co * C.LGD, "opex": active * C.OPEX_PER_LOAN_MONTH,
                    "acquisition": new_n * C.ACQUISITION_COST, "balance": bal_open})
    df = pd.DataFrame(out).set_index("m")
    df["net_contribution"] = df.interest_income - df.funding_cost - df.net_credit_loss - df.opex - df.acquisition
    return df


def cohort_table(con) -> pd.DataFrame:
    return con.sql("""
        SELECT l.vintage_idx, l.term, SUM(l.loan_amnt) AS amt, COUNT(*) AS n,
               SUM(s.pd * l.loan_amnt) / SUM(l.loan_amnt) AS pd
        FROM loans l JOIN scores s USING (loan_id) GROUP BY 1, 2 ORDER BY 1, 2
    """).df()


def forecast_originations(cohorts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    vol = cohorts.groupby("vintage_idx")["amt"].sum()
    fit = ExponentialSmoothing(np.log(vol.to_numpy()), trend="add", damped_trend=True).fit()
    fc = np.exp(fit.forecast(horizon))
    recent = cohorts[cohorts.vintage_idx >= cohorts.vintage_idx.max() - 5]
    mix = recent.groupby("term").agg(amt=("amt", "sum"), n=("n", "sum"),
                                     pdw=("pd", lambda s: (s * recent.loc[s.index, "amt"]).sum()))
    mix["share"] = mix.amt / mix.amt.sum()
    mix["avg_size"] = mix.amt / mix.n
    mix["pd"] = mix.pdw / mix.amt
    rows = []
    for i, total in enumerate(fc):
        v = int(vol.index.max()) + 1 + i
        for term, r in mix.iterrows():
            amt = total * r.share
            rows.append({"vintage_idx": v, "term": term, "amt": amt, "n": amt / r.avg_size, "pd": r.pd})
    return pd.DataFrame(rows)


def main():
    style()
    con = connect(read_only=True)
    actual = actual_pnl(con)
    cohorts = cohort_table(con)

    # --- backtest: learn on ≤ month 36, forecast 37–48 with actual originations ---
    bt_cut = C.SNAPSHOT_INDEX - 12
    curves_bt, pd_bt = learn_curves(con, bt_cut)
    bt = project(cohorts, curves_bt, pd_bt, range(bt_cut + 1, C.SNAPSHOT_INDEX + 1))
    comp = pd.DataFrame({f"{l}_actual": actual.loc[bt.index, l] for l in LINES} |
                        {f"{l}_forecast": bt[l] for l in LINES})
    err = pd.DataFrame({l: [actual.loc[bt.index, l].sum(), bt[l].sum()] for l in LINES},
                       index=["actual_12m", "forecast_12m"]).T
    err["error_pct"] = err.forecast_12m / err.actual_12m - 1
    err["monthly_mape"] = [np.mean(np.abs(bt[l] / actual.loc[bt.index, l] - 1)) for l in LINES]

    # --- forward forecast -------------------------------------------------------
    curves, pd_now = learn_curves(con, C.SNAPSHOT_INDEX)
    con.close()
    new = forecast_originations(cohorts, C.FORECAST_MONTHS)
    book = pd.concat([cohorts, new], ignore_index=True)
    horizon = range(C.SNAPSHOT_INDEX + 1, C.SNAPSHOT_INDEX + C.FORECAST_MONTHS + 1)
    k_int = 1 + -err.at["interest_income", "error_pct"] / (1 + err.at["interest_income", "error_pct"])
    k_ncl = 1 + -err.at["net_credit_loss", "error_pct"] / (1 + err.at["net_credit_loss", "error_pct"])
    err["calibration_overlay"] = np.nan
    err.loc[["interest_income", "net_credit_loss"], "calibration_overlay"] = [k_int, k_ncl]
    base = project(book, curves, pd_now, horizon, co_multiplier=k_ncl, int_multiplier=k_int)
    stress = project(book, curves, pd_now, horizon, co_multiplier=k_ncl * C.STRESS_PD_MULTIPLIER,
                     int_multiplier=k_int)

    summary = pd.DataFrame({
        "last_12m_actual": actual.loc[actual.index > C.SNAPSHOT_INDEX - 12, LINES].sum(),
        "next_12m_base": base[LINES].sum(),
        "next_12m_stress": stress[LINES].sum(),
    })
    last12 = actual.loc[actual.index > C.SNAPSHOT_INDEX - 12]
    summary.loc["avg_balance"] = [(last12.funding_cost / (C.COST_OF_FUNDS / 12)).mean(),
                                  base.balance.mean(), stress.balance.mean()]
    summary.loc["net_contribution_pct_avg_balance"] = summary.loc["net_contribution"] / summary.loc["avg_balance"]
    summary.round(4).to_csv(C.TAB_DIR / "pnl_forecast_summary.csv")
    err.round(4).to_csv(C.TAB_DIR / "pnl_backtest_error.csv")
    comp.round(0).to_csv(C.TAB_DIR / "pnl_backtest_monthly.csv")
    out = pd.concat([base.add_prefix("base_"), stress[["net_credit_loss", "net_contribution"]].add_prefix("stress_")], axis=1)
    out.insert(0, "month", [month_label(m) for m in out.index])
    out.round(0).to_csv(C.TAB_DIR / "pnl_forecast_monthly.csv")
    new.groupby("vintage_idx").amt.sum().rename(lambda v: month_label(v)).round(0).to_csv(C.TAB_DIR / "originations_forecast.csv")

    # --- figures ----------------------------------------------------------------
    money = FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    hist = actual.loc[actual.index >= 13]
    ax.plot(hist.index, hist.net_contribution, color="#2E5AAC", lw=1.8, label="Actual net contribution")
    ax.plot(bt.index, bt.net_contribution, color="#E07A2D", lw=1.5, ls="--", label="Backtest forecast")
    ax.plot(base.index, base.net_contribution, color="#3A9E6F", lw=2, label="Forecast — base")
    ax.plot(stress.index, stress.net_contribution, color="#B9435B", lw=2, label=f"Forecast — stress (CO ×{C.STRESS_PD_MULTIPLIER})")
    ax.fill_between(base.index, stress.net_contribution, base.net_contribution, color="#B9435B", alpha=0.08)
    ax.axvline(C.SNAPSHOT_INDEX + 0.5, color="grey", lw=0.8)
    ticks = list(range(13, base.index.max() + 1, 6))
    ax.set_xticks(ticks, [month_label(t) for t in ticks], rotation=35, ha="right")
    ax.yaxis.set_major_formatter(money)
    ax.set(title="Monthly net contribution: history, backtest and 12-month forecast")
    ax.legend(fontsize=7.8)

    ax = axes[1]
    s = summary.loc[LINES[:-1], ["last_12m_actual", "next_12m_base", "next_12m_stress"]].copy()
    s.loc[["funding_cost", "net_credit_loss", "opex", "acquisition"]] *= -1
    s.loc["net_contribution"] = summary.loc["net_contribution", s.columns]
    s.index = ["Interest income", "Funding cost", "Net credit losses", "Servicing opex", "Acquisition", "Net contribution"]
    s.columns = ["Last 12m (actual)", "Next 12m — base", "Next 12m — stress"]
    s.plot.barh(ax=ax, color=["#8C8C8C", "#3A9E6F", "#B9435B"])
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(money)
    ax.set(title="12-month P&L bridge")
    save(fig, "08_pnl_forecast")

    print(err.round(3).to_string())
    pct = summary.loc["net_contribution_pct_avg_balance"]
    print((summary.drop("net_contribution_pct_avg_balance") / 1e6).round(2).to_string())
    print("net contribution % of avg balance:", pct.round(4).to_dict())


if __name__ == "__main__":
    main()
