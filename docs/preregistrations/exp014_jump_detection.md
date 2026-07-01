# PRE-REGISTRATION — Experiment 014

**Jump Detection: Decomposing NIFTY Volatility into Continuous + Jump**

> Status: **PRE-REGISTERED (locked before running).** Definitions, method, and
> decision rules fixed in advance. Do not change thresholds after seeing results.

---

## Scientific Questions
1. **How much of NIFTY intraday volatility comes from jumps** vs the continuous
   diffusion component?
2. **Are jump days clustered** (does a jump today raise the chance of a jump
   tomorrow), or are they serially independent?
3. *(Secondary, pre-registered)* Does separating realized variance into
   **continuous + jump** parts improve next-day RV forecasting beyond plain HAR
   (i.e. is a HAR-CJ decomposition worthwhile)?

## Object of study
Daily NIFTY intraday returns at 5-minute frequency (09:15–15:30, overnight gap
excluded), 2024–2026, ~470–500 days. `M` ≈ 74 intraday returns per day.

## Definitions (standard: Barndorff-Nielsen–Shephard 2004/06; Huang-Tauchen 2005; Andersen-Bollerslev-Diebold 2007)
Let `r_i` be the i-th 5-minute log return of a day, `M` the count.
- **Realized Variance:** `RV = Σ r_i²` (total variation).
- **Bipower Variation:** `BV = μ1⁻² · (M/(M−1)) · Σ_{i=2}^M |r_i|·|r_{i−1}|`,
  with `μ1 = √(2/π)`. BV is robust to jumps → estimates the **continuous** part.
- **Jump Variation:** `JV = max(RV − BV, 0)`.
- **Relative jump:** `RJ = (RV − BV) / RV`.
- **Tripower quarticity** `TQ` (for the test's standard error), with
  `μ_{4/3} = 2^{2/3}·Γ(7/6)/Γ(1/2)`.
- **Jump test statistic (Huang-Tauchen ratio form):**
  `ZJ = √M · RJ / √( ((π/2)² + π − 5) · max(1, TQ/BV²) )`, `ZJ ~ N(0,1)` under
  the no-jump null. A day has a **significant jump** if `ZJ > Φ⁻¹(1−α)`
  (one-sided upper). Report at α = 0.05, 0.01, 0.001.
- **Significant jump component:** `J_t = 1[ZJ_t > Φ⁻¹(1−α)] · max(RV_t − BV_t, 0)`;
  **continuous component** `C_t = RV_t − J_t` (α = 0.01 for the decomposition).

## Method (fixed)
1. **Q1 — contribution:** report mean & median `JV/RV` and `RJ`; fraction of days
   with a significant jump at each α; mean jump share on jump days.
2. **Q2 — clustering:** build the daily significant-jump indicator (α=0.01).
   - Ljung-Box on the indicator series (lags 1,5,10) — H0: no autocorrelation.
   - 2×2 contingency: P(jump_t | jump_{t−1}) vs P(jump_t | no jump_{t−1}),
     Fisher exact / χ² test.
3. **Q3 (secondary) — HAR-CJ:** in log-RV space (identical protocol to Exp 003/
   012), compare
   - baseline: `log RV_{t+1} ~ har_d + har_w + har_m`
   - augmented: baseline **+ lagged `RJ_t`** (the continuous relative-jump, well
     defined every day).
   HAC t-stat on `RJ`; 4-fold expanding walk-forward incremental OOS R².

## Decision Rules (LOCKED)
- **Q1** is descriptive — no pass/fail; report the numbers with the small-`M`
  caveat below.
- **Q2 — CLUSTERED** if Ljung-Box rejects (p<0.05 at any tested lag) **and** the
  contingency test is significant (p<0.05) in the positive direction
  (P(jump|prior jump) > P(jump|no prior jump)). **NOT CLUSTERED** if neither
  rejects. **MIXED** if only one rejects (report, don't over-claim).
- **Q3 — JUMP DECOMPOSITION HELPS** only if `RJ` HAC |t| ≥ 2.5 **and** mean
  incremental OOS R² > 0 in ≥3 of 4 folds. Otherwise **NO INCREMENTAL VALUE**
  (report as such). This is the same discipline as Exp 012.

## Minimum sample & caveat
~470 daily obs is ample for Q1/Q2. **Caveat:** with only `M≈74` intraday
returns, the BNS/BV asymptotics ($M→∞$) are approximate; the jump test is known
to over-reject slightly at small M. Results are therefore reported as
*indicative*, and α=0.001 is included as a conservative cross-check.

## Expected failure modes
- Small-M over-rejection inflating the jump count (α=0.001 guards this).
- Jumps too rare for the contingency test to have power → Q2 lands MIXED/
  INSUFFICIENT.
- `RJ` redundant with HAR terms → Q3 NO INCREMENTAL VALUE (plausible).

## Scope
Characterisation + one forecasting-increment test. **No trading rule, no sizing,
no strategy.**

---

## Scoring (filled AFTER running — 494 daily obs, median M=74)
- **Q1 — jump contribution:** mean relative jump `RJ` = **8.4%**, median 5.8%.
  So NIFTY intraday variance is **predominantly continuous (~90%+)**; jumps are a
  modest minority. Significant-jump days: 20.9% at α=0.05, **9.3% at α=0.01**,
  3.2% at α=0.001. On flagged days the jump share is larger (≈24–37%), i.e. when
  a jump happens it is material, but such days are the minority.
- **Q2 — clustering: NOT CLUSTERED (serially independent).** Ljung-Box on the
  daily jump indicator does not reject (lag-1 p=0.079, lag-5 p=0.20, lag-10
  p=0.44). Contingency: P(jump | prior-day jump) = **2.2%** vs P(jump | no prior
  jump) = **10.1%** — if anything *lower* after a jump (Fisher one-sided p=0.99,
  i.e. no positive dependence). Jumps do not beget jumps at the daily horizon;
  they look like isolated events, mildly *anti*-persistent if anything.
- **Q3 — HAR-CJ increment: NO INCREMENTAL VALUE.** Lagged `RJ` in the
  leverage-free HAR: coef −0.055, HAC t = **−0.29** (not significant); in-sample
  R² unchanged (0.315→0.315); mean walk-forward incremental OOS R² = **−0.0088**
  (only 1/4 folds positive). Separating the jump component does **not** improve
  next-day RV forecasting here.

## Interpretation (honest)
NIFTY daily volatility is overwhelmingly a **continuous** process; jumps are
occasional, material when they occur, but **rare, isolated, and not
autocorrelated**, and knowing yesterday's jump size adds nothing to the HAR
forecast. This is consistent with the broader finding that the *continuous*
volatility (which HAR captures well) is the persistent, forecastable part — and
it reinforces Exp 013's result that plain HAR is the right workhorse. Caveat: at
M≈74 the jump test is approximate and may slightly over-count jumps, which would
only *weaken* an already-null clustering/forecast result — so the negative
conclusions are robust.

---
*Pre-registered 2026-07-01, executed same day as
`scripts/exp014_jump_detection.py`.*
