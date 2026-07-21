# PRE-REGISTRATION — Experiment 027

**Granular VRP: Implied vs Term-Matched Realized Volatility on Our Own Collected Chains**

> Status: **PRE-REGISTERED (locked before running).** This is the decisive
> gatekeeper for the intraday/granular volatility risk premium, gated at 20
> collected trading days (now met). Method and decision rules fixed in advance.

---

## Why this experiment matters
Exp 028 proved the VRP is real at the **macro** level (India VIX vs monthly
realized, 10 years). The interim readout (`vrp_derived_check.py`) showed the
same *direction* on our own chains, but with an explicit, flagged flaw: its
"realized" leg was **intraday-only** (session open to session close), which
**omits every overnight/weekend gap the option is actually exposed to** and
therefore **structurally inflates** the apparent VRP. Exp 027 fixes that: it
measures realized volatility **term-matched** to each option's actual
days-to-expiry (DTE), using close-to-close returns across the real calendar
(including weekends/overnight), the same convention used for macro VIX in
Exp 028.

## Scientific Question
On our own collected option chains, does the ATM implied volatility at entry
exceed the volatility **actually realized over the option's own remaining life**
(entry date -> expiry date), and is the premium's sign consistent with the
macro result (Exp 028: implied > realized ~80% of the time)?

## Data
All collected option-chain days as of 2026-07-20 (the run date), spanning
2026-06-22 -> 2026-07-20 (21 complete trading days; the in-progress day is
excluded). Source: `data/option_chain/...` (raw quotes; IV derived via
Black-Scholes, per the non-destructive `nifty_quant/analytics/options.py` /
`nifty_quant/research/derive_iv.py` machinery already in the codebase).

## Definitions
- **Daily closing spot**: the underlying NIFTY spot recorded in the LAST
  snapshot of each collected day (spot is expiry-independent; any chain in that
  snapshot gives the same spot).
- **Entry ATM IV**: mean ATM implied vol (call+put average, Black-Scholes) across
  the NEAR-expiry chain's snapshots in the FIRST 15 minutes of the session
  (09:15-09:30) -- a well-defined, look-ahead-free entry point, less noisy than
  a single snapshot.
- **DTE**: trading days from entry date to the near expiry's date, counted by
  position in our own collected trading-day calendar (not raw calendar days --
  avoids weekend/holiday miscounts).
- **Term-matched realized vol**: `RV = sqrt(252/DTE * sum(r_i^2)) * 100`, where
  `r_i` are close-to-close log returns of the daily closing spot from entry date
  to the expiry date, inclusive of every calendar day in between (so weekend/
  overnight gaps are captured exactly as they are for VIX in Exp 028).
- **VRP = Implied - Realized** (vol points, annualized).
- **Eligibility**: a day is included in the term-matched analysis ONLY IF its
  expiry date is <= the last collected trading day (i.e. we actually observed
  the full holding period). Days whose expiry falls beyond our data are
  reported separately as "window insufficient" -- NOT forced into the estimate.

## Method (fixed)
1. Build the daily closing-spot series across all 21 days (one number/day).
2. For each entry day with DTE >= 1 and expiry within the window, compute
   Implied, Realized (term-matched), and VRP as defined above.
3. For DTE = 0 (expiry days themselves), report separately: entry ATM IV vs
   that SAME day's own intraday realized move (the 0-DTE case is definitionally
   intraday -- there is no forward multi-day window).
4. Report: n eligible, mean/median VRP, % positive, per-DTE breakdown, and the
   excluded "window insufficient" days (transparency, not hidden).
5. One-sided sign test (robust to n this small) on the eligible VRP sample.

## Decision Rules (LOCKED)
- **CONFIRMS MACRO DIRECTION** if VRP > 0 on **> 60%** of eligible days AND the
  sign test does not reject positive dependence outright (this is a SMALL
  sample -- n≈15-20 -- so this is a DIRECTIONAL check, not a significance claim).
- **INCONCLUSIVE** if eligible n < 8 (too few multi-day windows survived the
  expiry-within-window filter) -- report descriptively, no verdict.
- **CONTRADICTS MACRO** if VRP <= 0 on the majority of eligible days -- an
  important, surprising result to report honestly, not explain away.
- **Explicitly NOT claimed regardless of outcome:** statistical significance
  (n is far too small for that), a trading strategy, or a precise VRP magnitude.
  This is a DIRECTIONAL confirmation/disconfirmation only. A real significance
  claim needs ~60-100+ days (see roadmap).

## Honest sample-size caveat
With ~21 collected days and a same-window-expiry filter, the eligible sample is
likely n≈12-18, split across small per-DTE buckets. This is intentionally framed
as a **directional gate-check**, not a definitive VRP estimate. The Exp 016/028
discipline (do not claim more than the sample supports) applies fully.

## Scope
Directional confirmation on live-collected data. **No trading rule, no strategy.**

---

## Scoring (filled AFTER running — 21 complete collected days, 2026-06-22 → 2026-07-20)

### VERDICT: **CONFIRMS MACRO DIRECTION** — VRP positive on 92% of 13 eligible days.

- **Eligible (term-matched, full window observed): 13.** Mean VRP **+4.3** vol
  points, median +4.5, **positive on 12/13 days (92%)**. One-sided sign test
  p=0.002 — but per the locked rule this is explicitly **not** treated as a
  significance claim; n=13 is a directional gate-check, not an estimate.
- **The one negative day (2026-06-22, VRP −8.2):** entry IV 14.8% vs realized
  22.9% over its 1-day term — a day where the market moved MORE than implied
  priced, the tail case the premium doesn't always protect against. Reported
  honestly, not excluded.
- **0-DTE bucket (reported separately, intraday-only, 4 days):** mean VRP
  **+12.7**, positive 100%. Consistent with the interim readout's earlier
  finding that 0-DTE IV is elevated — expected microstructure, not part of the
  term-matched claim.
- **4 days excluded as "window insufficient"** (07-15 through 07-20 — their
  07-21 expiry hadn't been fully observed at run time). Correctly excluded per
  the locked eligibility rule rather than forced into the estimate.
- **Magnitude vs the interim readout:** term-matching (incl. weekend/overnight
  gaps) brought mean VRP down from the interim's inflated **+8.0** to a more
  realistic **+4.3** — closer to, though still above, the macro Exp 028 mean of
  **+2.5**. This is the expected direction of correction: fixing the flagged
  intraday-only flaw reduced, but did not eliminate, the apparent premium.

## Honest interpretation
The intraday/granular VRP is **directionally consistent with the macro result**
on the first real look at our own collected data: implied consistently priced
above what was subsequently realized, in 12 of 13 clean windows, with one
informative negative counterexample. This is NOT yet a statistically
established, tradeable estimate — n=13 is far too small, entirely from a single
calm-VIX regime (11.8–17.3 mean IV throughout), and has not been tested through
any volatility spike. The next milestone is accumulating toward ~60-100 days
across mixed regimes before this becomes an estimate to build on, and Exp 030's
cost/tail lessons (a modest premium easily erased by execution costs and a fat
left tail) apply here exactly as they did to the macro backtest.

## Reusable output
`scripts/exp027_granular_vrp.py` -- term-matched, look-ahead-safe VRP builder
reusable as the dataset grows; correctly separates 0-DTE from term-matched
windows and excludes days whose expiry hasn't yet been observed.

---
*Pre-registered 2026-07-21, executed same day as `scripts/exp027_granular_vrp.py`
on 21 collected trading days (06-22 → 07-20).*
