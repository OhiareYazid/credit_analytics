-- Vintage curves: cumulative charge-off rate (% of originated amount) and
-- ever-30+ DPD rate (% of loans) by origination cohort and months on book.
-- Each vintage is only reported for the MOBs it has actually been observed,
-- so young cohorts stop early instead of showing false zeros.

WITH snapshot AS (
    SELECT snapshot_idx FROM meta
),
orig AS (
    SELECT vintage_idx, vintage,
           COUNT(*)       AS n_loans,
           SUM(loan_amnt) AS orig_amt
    FROM loans
    GROUP BY vintage_idx, vintage
),
grid AS (                                   -- one row per vintage × observed MOB
    SELECT o.*, m.mob
    FROM orig o
    CROSS JOIN range(1, 61) AS m(mob)
    WHERE m.mob <= (SELECT snapshot_idx FROM snapshot) - o.vintage_idx
),
chargeoffs AS (
    SELECT vintage_idx, mob, SUM(chargeoff_amt) AS co_amt
    FROM performance
    GROUP BY vintage_idx, mob
),
first_30plus AS (                           -- first month each loan hit 31+ DPD
    SELECT loan_id, vintage_idx, MIN(mob) AS mob
    FROM performance
    WHERE state IN ('31-60', '61-90', '91-120', 'Charged-off')
    GROUP BY loan_id, vintage_idx
),
ever30 AS (
    SELECT vintage_idx, mob, COUNT(*) AS n_new
    FROM first_30plus
    GROUP BY vintage_idx, mob
)
SELECT
    g.vintage_idx,
    g.vintage,
    g.mob,
    g.n_loans,
    g.orig_amt,
    SUM(COALESCE(c.co_amt, 0)) OVER w / g.orig_amt AS cum_co_rate,
    SUM(COALESCE(e.n_new, 0))  OVER w / g.n_loans  AS ever_30plus_rate
FROM grid g
LEFT JOIN chargeoffs c USING (vintage_idx, mob)
LEFT JOIN ever30     e USING (vintage_idx, mob)
WINDOW w AS (PARTITION BY g.vintage_idx ORDER BY g.mob
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
ORDER BY g.vintage_idx, g.mob;
