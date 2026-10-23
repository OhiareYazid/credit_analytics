-- Score cut-off strategy table: approval rate, bad rate of approved loans and
-- realised profit at every candidate cut-off, computed on the out-of-time
-- sample, under two pricing regimes (`pricing` = 'flat' | 'risk_based').
-- Loan-level lifetime profit (`loan_profit`) is produced in Python from
-- contractual cash flows and the unit-economics assumptions in config.py.

WITH base AS (
    SELECT p.pricing, s.score, s.pd, l.bad_loan, p.loan_profit, l.loan_amnt
    FROM scores s
    JOIN loans l       USING (loan_id)
    JOIN loan_profit p USING (loan_id)
    WHERE s.sample = 'oot'
),
cutoffs AS (
    SELECT DISTINCT pricing, (score // 5) * 5 AS cutoff FROM base
),
totals AS (
    SELECT pricing, COUNT(*) AS n_total FROM base GROUP BY pricing
)
SELECT
    c.pricing,
    c.cutoff,
    COUNT(b.score)                               AS n_approved,
    COUNT(b.score) / ANY_VALUE(t.n_total)        AS approval_rate,
    AVG(b.bad_loan)                              AS bad_rate,
    AVG(b.pd)                                    AS expected_bad_rate,
    SUM(b.loan_amnt)                             AS booked_amount,
    SUM(b.loan_profit)                           AS total_profit,
    SUM(b.loan_profit) / NULLIF(SUM(b.loan_amnt), 0) AS profit_margin
FROM cutoffs c
JOIN totals t USING (pricing)
LEFT JOIN base b ON b.pricing = c.pricing AND b.score >= c.cutoff
GROUP BY c.pricing, c.cutoff
ORDER BY c.pricing, c.cutoff;
