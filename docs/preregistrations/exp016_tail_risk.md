# PRE-REGISTRATION — Experiment 016

**Tail Risk of NIFTY Returns: How Fat Are the Tails?**

> Status: **PRE-REGISTERED (locked before running).** Definitions, method, and
> decision rules fixed in advance.

---

## Scientific Questions
1. How **non-normal** are NIFTY daily returns (skewness, excess kurtosis)?
2. How **fat are the tails** — what is the tail index / GPD shape, and how many
   moments are finite?
3. Are the **left (crash) and right (rally) tails asymmetric**?
4. How badly does a **Gaussian VaR/ES understate** true tail risk vs an
   empirical / EVT estimate?

## Object of study
NIFTY daily **close-to-close** log returns, 2024–2026 (~490 days), built from
the 5-minute candle warehouse (last close of each session). Losses (left tail)
are the risk-management focus; the right tail is examined for asymmetry.

## Definitions & method (standard EVT)
- **Moments:** mean, std, skewness, excess kurtosis; **Jarque-Bera** normality
  test (H0: normal).
- **Hill estimator** (`stat_tests.hill_estimator`): tail index `alpha` for each
  tail, scanned over a range of `k` (top 5–20% order statistics) for stability.
  `alpha` bounds finite moments (moments up to < `alpha` exist); `xi = 1/alpha`.
- **Peaks-over-threshold / GPD** (`stat_tests.pot_gpd_fit`): fit a Generalized
  Pareto to exceedances over the 95th percentile of losses; report shape `xi`
  (>0 = heavy/power-law), scale `beta`, and `tail_df = 1/xi` (equivalent
  Student-t d.o.f.). Repeat for the right tail.
- **VaR/ES comparison** at 99% and 99.5%: empirical vs Gaussian(μ,σ) vs
  GPD-EVT. Report the ratio (EVT or empirical) / Gaussian to quantify Gaussian
  under-statement. ES = expected shortfall (mean loss beyond VaR).
- **σ-event count:** observed count of |return| > 3σ, 4σ, 5σ vs the Gaussian
  expectation over the same sample size.

## Decision Rules (LOCKED — descriptive study, but verdicts fixed)
- **HEAVY-TAILED** if ALL: Jarque-Bera rejects normality (p<0.01), **and** the
  loss-tail GPD `xi` > 0 with a bootstrap/asymptotic indication it is not ~0
  (`tail_df` finite, roughly < 10), **and** Hill `alpha` is finite and < ~5
  (i.e. 4th/5th moments questionable).
- **NEAR-GAUSSIAN TAILS** if Jarque-Bera does not reject and GPD `xi` ≈ 0.
- **ASYMMETRIC TAILS** if the loss-tail `xi` (or Hill `alpha`) differs
  materially from the gain-tail (rule of thumb: shapes differ by > ~0.10, or one
  tail's Hill `alpha` CI excludes the other's point estimate). Report direction
  (which tail is fatter).
- **GAUSSIAN UNDERSTATES RISK** if empirical/EVT 99.5% ES exceeds Gaussian ES by
  > 20%.

## Minimum sample & caveats
~490 daily returns. EVT tail estimates are **data-hungry**: with only ~25
exceedances above the 95th percentile, `xi` has a wide CI. Results are therefore
reported as **indicative**, with the Hill-vs-`k` stability plot (printed table)
and the small-sample caveat explicit. This is a *characterisation*, and its main
practical use is **position sizing** (feeding fractional-Kelly / risk limits) —
not a trading signal.

## Expected outcome (honest prior)
Equity-index daily returns are almost universally fat-tailed and
negatively-skewed (crashes sharper than rallies). The likely result is
HEAVY-TAILED + left tail fatter than right + Gaussian materially understating
99.5% ES. The value is a *quantified* tail (a number for `xi`/`alpha` and an ES
multiple), not the qualitative fact.

## Scope
Characterisation for risk sizing. **No trading rule, no strategy.**

---

## Scoring (filled AFTER running — 495 daily returns, 2024-06 → 2026-06)
Daily return: mean +0.004%, std 0.825% (annualised 13.1%).

- **Q1 — non-normality: clearly non-normal.** Excess kurtosis **+2.08** (fat),
  skewness +0.05 (~symmetric), Jarque-Bera stat 89.5, **p = 3.6e-20 → reject
  normal decisively.**
- **Q2 — tail fatness (estimator conflict, honest):**
  - **Hill tail index ≈ 3** for *both* tails across k = 5–10% (left α 2.9–3.3,
    right α 2.2–3.5). α≈3 ⇒ variance exists but the **4th moment is borderline**
    — the textbook signature of fat tails.
  - **σ-events (clearest evidence):** 7 days beyond 3σ vs 1.34 Gaussian-expected
    (**~5×**); 2 beyond 4σ vs 0.03 expected (**~66×**). Unambiguous fat tails.
  - **GPD @95th pct:** left `xi = −0.02` (looks thin!), right `xi = +0.27`
    (heavy). This *contradicts* Hill and the σ-counts, and is an artifact of only
    **25 exceedances** — the GPD shape is extremely noisy at that count and the
    threshold choice dominates.
- **Q3 — asymmetry: DO NOT over-claim.** The locked rule flagged "asymmetric,
  right tail fatter," but that rests entirely on the noisy 25-point GPD; Hill
  shows the two tails ≈ equal (α≈3 each). The apparent right-fatness is likely a
  **sample artifact of the 2024–26 bull run** (the largest single-day moves were
  up moves), not a structural feature. Verdict: **near-symmetric on the robust
  (Hill/σ-count) evidence; any right-skew is fragile and sample-specific.**
- **Q4 — Gaussian understates tail risk: YES.** 99.5% ES: Gaussian 2.38% vs
  empirical 2.90% / EVT 3.00% → **~1.22×**. 99% ES: 1.23×. Gaussian VaR/ES
  materially under-states real tail losses.

## Verdict against the LOCKED rules (no goalpost-moving)
The pre-registered HEAVY-TAILED rule required the *left-tail GPD* `xi > 0.05`,
which came out ≈ 0 → the mechanical verdict printed **"NOT clearly heavy."** I am
**not** changing the locked rule. But I flag honestly that the rule hinged on a
single noisy estimator (25-point left GPD), while the **preponderance of robust
evidence — JB p=3.6e-20, excess kurtosis +2.08, Hill α≈3, and 5×/66× too many
3σ/4σ days — says NIFTY daily returns ARE fat-tailed.** The lesson recorded for
future preregs: for ~500-obs tail work, gate the verdict on Hill + σ-counts (data
we have enough of), not on a high-threshold GPD shape (data we don't).

## Practical takeaway (the point of the experiment)
For position sizing and risk limits, **use the empirical / EVT expected
shortfall (~1.2× the Gaussian number), not Gaussian VaR.** Concretely, budget
~3.0% single-day 99.5% ES rather than the Gaussian 2.4%. This feeds the
fractional-Kelly / risk-limit layer (`stat_tests.fractional_kelly`), which
already defaults to conservative half-Kelly precisely because of fat tails and
estimation error — Exp 016 quantifies *how* fat.

## Reusable output
Added `hill_estimator` and `pot_gpd_fit` to `nifty_quant/analytics/stat_tests.py`
(3 new validated tests; suite 16 → 19).

---
*Pre-registered 2026-07-01, executed same day as `scripts/exp016_tail_risk.py`.*
