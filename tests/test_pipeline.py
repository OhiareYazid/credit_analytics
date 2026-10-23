"""Sanity checks on pipeline outputs. Run after `python run_all.py`:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

TAB = C.TAB_DIR


@pytest.fixture(scope="module")
def con():
    if not C.DB_PATH.exists():
        pytest.skip("run `python run_all.py` first")
    c = duckdb.connect(str(C.DB_PATH), read_only=True)
    yield c
    c.close()


def test_every_loan_has_performance_and_a_score(con):
    missing = con.sql("""
        SELECT COUNT(*) FROM loans l
        LEFT JOIN (SELECT DISTINCT loan_id FROM performance) p USING (loan_id)
        LEFT JOIN scores s USING (loan_id)
        WHERE p.loan_id IS NULL OR s.loan_id IS NULL
    """).fetchall()[0][0]
    assert missing == 0


def test_only_bad_loans_charge_off(con):
    n = con.sql("""
        SELECT COUNT(*) FROM performance p JOIN loans l USING (loan_id)
        WHERE p.state = 'Charged-off' AND l.bad_loan = 0
    """).fetchall()[0][0]
    assert n == 0


def test_no_observation_after_snapshot(con):
    assert con.sql("SELECT MAX(cal_month_idx) FROM performance").fetchall()[0][0] <= C.SNAPSHOT_INDEX


def test_balances_never_negative_or_above_principal(con):
    bad = con.sql("""
        SELECT COUNT(*) FROM performance p JOIN loans l USING (loan_id)
        WHERE p.balance_close < 0 OR p.balance_open > l.loan_amnt + 0.01
    """).fetchall()[0][0]
    assert bad == 0


def test_roll_rate_rows_sum_to_one():
    m = pd.read_csv(TAB / "roll_rate_matrix_accounts.csv", index_col=0)
    assert np.allclose(m.sum(axis=1), 1.0, atol=1e-3)


def test_absorption_probabilities_rise_with_delinquency():
    a = pd.read_csv(TAB / "bucket_absorption_probabilities.csv", index_col=0)["P(charge-off)"]
    assert a.is_monotonic_increasing


def test_vintage_curves_are_cumulative():
    v = pd.read_csv(TAB / "vintage_curves.csv")
    assert (v.groupby("vintage_idx")["cum_co_rate"].diff().dropna() >= -1e-12).all()


def test_scorecard_bands_rank_order_risk():
    b = pd.read_csv(TAB / "risk_band_calibration.csv")
    oot = b[b["sample"] == "oot"].set_index("band").loc[list("ABCDE"), "bad_rate"]
    assert oot.is_monotonic_increasing


def test_scorecard_discriminates_out_of_time():
    m = pd.read_csv(TAB / "scorecard_metrics.csv").set_index("sample")
    assert m.at["oot", "gini"] > 0.25


def test_ab_test_recovers_injected_effect():
    s = pd.read_csv(TAB / "ab_test_summary.csv", index_col=0)["value"]
    assert s["ci95_low"] <= s["true_injected_uplift_pp"] <= s["ci95_high"]
    assert s["srm_p_value"] > 0.01


def test_limit_policy_never_raises_exposure():
    i = pd.read_csv(TAB / "limit_policy_impact.csv")
    assert (i["assigned"] <= i["requested"] + 1e-6).all()
