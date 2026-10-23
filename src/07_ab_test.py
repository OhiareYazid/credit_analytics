"""Step 7 — A/B test with power analysis: early-arrears repayment nudge.

Hypothesis: a personalised in-app + SMS nudge ("pay now to protect your
limit") sent on day 3 past due raises the share of 1-30 DPD accounts that cure
to Current by the next month end.

Design (decided BEFORE looking at outcomes):
- unit: account entering 1-30 DPD; 50/50 random split, stratified by risk band
- primary metric: cure rate (1-30 → Current next month)
- guardrail: roll rate 1-30 → 31-60 (must not rise)
- α = 0.05 two-sided, power = 80%, MDE = +3 pp on the observed baseline
- fixed horizon: no peeking; run until the pre-computed sample is reached

Execution: the "experiment" re-uses real 1-30 accounts from the performance
data. Control keeps its observed outcome; for treatment, a known uplift is
injected by converting a random share of non-cures into cures. Because the true
effect is known, the analysis can be checked against it.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (proportion_effectsize, proportions_ztest,
                                          confint_proportions_2indep)

import config as C
from db import connect
from plotting import style, save

rng = np.random.default_rng(C.SEED + 7)


def required_n(p0, mde, alpha=C.AB_ALPHA, power=C.AB_POWER):
    h = proportion_effectsize(p0 + mde, p0)
    return int(np.ceil(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1.0)))


def main():
    style()
    con = connect(read_only=True)
    pool = con.sql("""
        SELECT p.loan_id, p.cal_month_idx, p.balance_open, p.state AS next_state, s.band
        FROM performance p JOIN scores s USING (loan_id)
        WHERE p.state_from = '1-30'
        ORDER BY p.cal_month_idx DESC, p.loan_id
    """).df()
    con.close()
    absorb = pd.read_csv(C.TAB_DIR / "bucket_absorption_probabilities.csv", index_col=0)

    recent = pool[pool.cal_month_idx > C.SNAPSHOT_INDEX - 12]
    p0 = (recent.next_state == "Current").mean()
    monthly_inflow = len(recent) / 12

    # --- power analysis --------------------------------------------------------
    n_arm = required_n(p0, C.AB_MDE_PP)
    months_needed = 2 * n_arm / monthly_inflow
    mde_grid = np.round(np.arange(0.01, 0.061, 0.005), 3)
    power_tbl = pd.DataFrame({"mde_pp": mde_grid,
                              "n_per_arm": [required_n(p0, m) for m in mde_grid]})
    power_tbl["months_at_current_inflow"] = 2 * power_tbl.n_per_arm / monthly_inflow

    # --- run the experiment on real accounts -----------------------------------
    exp = pool.head(2 * n_arm).copy()
    exp["arm"] = "control"
    for band, idx in exp.groupby("band").groups.items():        # stratified 50/50
        idx = rng.permutation(np.asarray(idx))
        exp.loc[idx[: len(idx) // 2], "arm"] = "treatment"
    exp["cured"] = (exp.next_state == "Current").astype(int)
    exp["rolled"] = (exp.next_state == "31-60").astype(int)
    # inject the true effect: convert a share of treatment non-cures into cures
    p_cure_obs = exp.loc[exp.arm == "control", "cured"].mean()
    flip_p = C.AB_TRUE_UPLIFT_PP / (1 - p_cure_obs)
    t_noncure = exp.index[(exp.arm == "treatment") & (exp.cured == 0)]
    flip = t_noncure[rng.random(len(t_noncure)) < flip_p]
    exp.loc[flip, ["cured", "rolled"]] = [1, 0]

    # --- analysis ---------------------------------------------------------------
    g = exp.groupby("arm").agg(n=("cured", "size"), cures=("cured", "sum"),
                               cure_rate=("cured", "mean"), rolls=("rolled", "sum"),
                               roll_rate=("rolled", "mean"))
    srm_p = stats.chisquare(g["n"]).pvalue                       # sample-ratio mismatch check
    z, p_val = proportions_ztest(g.loc[["treatment", "control"], "cures"],
                                 g.loc[["treatment", "control"], "n"], alternative="two-sided")
    ci_lo, ci_hi = confint_proportions_2indep(g.at["treatment", "cures"], g.at["treatment", "n"],
                                              g.at["control", "cures"], g.at["control", "n"],
                                              method="newcomb")
    diff = g.at["treatment", "cure_rate"] - g.at["control", "cure_rate"]
    # guardrail: one-sided test that treatment does NOT roll more accounts to 31-60
    z_g, p_g = proportions_ztest(g.loc[["treatment", "control"], "rolls"],
                                 g.loc[["treatment", "control"], "n"], alternative="larger")

    # value of the effect: each extra cure moves an account from P(CO|1-30) to P(CO|Current)
    avg_bal = recent.balance_open.mean()
    loss_avoided_per_cure = (absorb.at["1-30", "P(charge-off)"] - absorb.at["Current", "P(charge-off)"]) * avg_bal * C.LGD
    annual_value = diff * monthly_inflow * 12 * loss_avoided_per_cure

    # heterogeneity by band — exploratory, Bonferroni-corrected
    het = []
    for band, d in exp.groupby("band"):
        t, c = d[d.arm == "treatment"], d[d.arm == "control"]
        _, p_b = proportions_ztest([t.cured.sum(), c.cured.sum()], [len(t), len(c)])
        het.append({"band": band, "n": len(d), "uplift_pp": t.cured.mean() - c.cured.mean(),
                    "p_value": p_b, "significant_bonferroni": p_b < C.AB_ALPHA / 5})
    het = pd.DataFrame(het)

    summary = pd.DataFrame([{
        "baseline_cure_rate": p0, "mde_pp": C.AB_MDE_PP, "n_per_arm_required": n_arm,
        "monthly_1_30_inflow": monthly_inflow, "months_to_run": months_needed,
        "srm_p_value": srm_p, "control_cure_rate": g.at["control", "cure_rate"],
        "treatment_cure_rate": g.at["treatment", "cure_rate"], "uplift_pp": diff,
        "ci95_low": ci_lo, "ci95_high": ci_hi, "p_value": p_val,
        "guardrail_roll_control": g.at["control", "roll_rate"],
        "guardrail_roll_treatment": g.at["treatment", "roll_rate"], "guardrail_p_worse": p_g,
        "true_injected_uplift_pp": C.AB_TRUE_UPLIFT_PP,
        "loss_avoided_per_extra_cure": loss_avoided_per_cure, "annualised_loss_avoided": annual_value,
    }]).T.rename(columns={0: "value"})
    summary.round(5).to_csv(C.TAB_DIR / "ab_test_summary.csv")
    power_tbl.round(2).to_csv(C.TAB_DIR / "ab_power_table.csv", index=False)
    het.round(4).to_csv(C.TAB_DIR / "ab_heterogeneity_by_band.csv", index=False)

    # --- figure -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(power_tbl.mde_pp * 100, power_tbl.n_per_arm, marker="o", color="#2E5AAC")
    ax.axvline(C.AB_MDE_PP * 100, color="#B9435B", ls="--", lw=1)
    ax.annotate(f"{n_arm:,} per arm\n≈ {months_needed:.1f} months of inflow",
                (C.AB_MDE_PP * 100, n_arm), xytext=(12, 18), textcoords="offset points", fontsize=8.5)
    ax.set_yscale("log")
    ax.set(title=f"Sample size vs minimum detectable effect (baseline {p0:.1%})",
           xlabel="MDE (percentage points)", ylabel="Accounts per arm (log scale)")
    ax = axes[1]
    rates = g["cure_rate"].loc[["control", "treatment"]]
    errs = [1.96 * np.sqrt(r * (1 - r) / g.at[a, "n"]) for a, r in rates.items()]
    ax.bar(rates.index, rates.values, yerr=errs, capsize=6, color=["#8C8C8C", "#3A9E6F"])
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylim(0, rates.max() * 1.25)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set(title=f"Cure rate: +{diff*100:.1f} pp (95% CI {ci_lo*100:.1f} to {ci_hi*100:.1f}), p = {p_val:.3g}")
    save(fig, "07_ab_test")

    print(summary.round(4).to_string())
    print(power_tbl.assign(mde_pp=power_tbl.mde_pp * 100).round(1).to_string(index=False))
    print(het.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
