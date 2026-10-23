-- End-of-month balance in each delinquency bucket by calendar month.
-- Used to compute net flow rates (balance in bucket k this month ÷ balance in
-- bucket k-1 last month), the standard collections loss-forecasting view.

SELECT
    cal_month_idx,
    state,
    SUM(CASE WHEN state = 'Charged-off' THEN chargeoff_amt ELSE balance_close END) AS balance,
    COUNT(*) AS n_accounts
FROM performance
WHERE state <> 'Paid-off'
GROUP BY cal_month_idx, state
ORDER BY cal_month_idx, state;
