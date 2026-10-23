-- Roll-rate (transition) matrix over the last 12 calendar months:
-- for every loan-month, the delinquency state at the start (state_from) and at
-- the end of the month (state). Counts and opening balances are returned in
-- long form; Python pivots them and normalises each row to 100%.
-- Loans already absorbed (charged-off / paid-off) are excluded by construction,
-- because the performance table stops at the absorbing month.

SELECT
    state_from,
    state      AS state_to,
    COUNT(*)           AS n_accounts,
    SUM(balance_open)  AS balance
FROM performance
WHERE cal_month_idx > (SELECT snapshot_idx FROM meta) - 12
  AND cal_month_idx <= (SELECT snapshot_idx FROM meta)
GROUP BY state_from, state
ORDER BY state_from, state_to;
