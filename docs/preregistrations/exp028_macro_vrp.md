# PRE-REGISTRATION — Experiment 028

**Macro Volatility Risk Premium (VRP) — the 10-Year Go/No-Go Dry Run**

> Status: **PRE-REGISTERED (locked before running).** This is the platform's
> single most important gatekeeper: it tests, on 10 years of free daily data,
> whether the volatility-selling thesis that justifies the entire option-chain
> collection effort is even structurally true.

---

## Why this experiment is the gatekeeper
The whole option-chain pipeline exists to harvest a **volatility risk premium**
(implied vol systematically exceeding subsequent realized vol). If that premium
is NOT structurally positive on NIFTY over a decade, we should stop building the
5-minute options infrastructure now. This validates (or kills) the core thesis
in one afternoon using daily data we can backfill for free.

## Scientific Question
Over 10 years, is `India VIX_t` (implied, ~30-day annualized vol) systematically
**greater** than the NIFTY realized volatility that follows over the next ~1
month? And if so, is the premium **persistent and mean-reverting** (i.e.
harvestable), or just a noisy average hiding ruinous negative tails?

## Data
- **^NSEI** (NIFTY 50) daily close, 10y from Yahoo (`yfinance`), cached to
  `data/external/`.
- **^INDIAVIX** (India VIX) daily close, 10y, same source.
- ~2450 trading days, 2016–2026 (spans COVID-2020 — deliberately, as the stress
  test for the short-vol tail).

## Definitions
- `VIX_t`: India VIX close on day t (annualized %, the market's implied 30-day
  vol).
- **Forward realized vol** over the VIX horizon (21 trading days ≈ 30 calendar):
  `RV_fwd_t = sqrt( (252 / 21) * Σ_{i=1..21} r_{t+i}² ) * 100`, where `r` are
  NIFTY daily log returns. Annualized %, same units as VIX.
- **VRP_t = VIX_t − RV_fwd_t** (vol points). Positive → implied exceeded
  realized → short-vol / premium would have paid.
- **VIX/RV ratio** = `VIX_t / RV_fwd_t`.

## Look-ahead note (honest)
`VRP_t` uses forward returns (t+1..t+21). This is intentional and correct: it is
the **ex-post payoff of selling vol at t** and holding one month. It is a
characterisation of the realized premium, not a peeking predictor. The final 21
days (no forward window) are dropped.

## Method (fixed)
1. Fraction of days with `VRP_t > 0`; mean & median VRP; mean VIX/RV ratio.
2. Significance that mean VRP > 0: one-sided t-test **and** sign test (robust to
   fat tails). Report both.
3. **Per-year** mean VRP and positive-fraction (structural persistence; expose
   2020 and any other stress years explicitly).
4. **Tail honesty:** the most negative VRP, the 1st/5th percentiles, and the
   fraction of days with VRP < −5 vol points (the short-vol blow-up risk).
5. **Mean-reversion / harvestability:** ADF test and OU half-life
   (`stat_tests.adf_test`, `half_life`) on the VRP series and on VIX itself.

## Decision Rules (LOCKED) — the Go/No-Go
- **GO — VRP STRUCTURALLY POSITIVE** if ALL: mean VRP > 0 with sign-test p<0.01,
  positive fraction > 60%, and positive mean VRP in ≥ 7 of 10 years. → the
  volatility-selling thesis is validated at the macro level; **continue the
  option-chain collection**, and the premium is real enough to build on.
- **NO-GO / RETHINK** if the premium is flat (fraction ≈ 50%, mean ≈ 0) or
  negative, or positive in < 6 years. → the thesis is not supported on NIFTF
  macro data; reconsider before investing more in the options pipeline.
- **CONDITIONAL GO** if the premium is positive on average but with a severe
  negative tail (e.g. large VRP<−5 fraction, deep percentile): GO, **but** the
  strategy must be a risk-managed, fractionally-sized short-vol book (per Exp 016
  tail work), never naive.

## Expected outcome (honest prior)
Equity-index VRP is one of the most robust premia in finance (S&P: VIX > realized
~85% of the time). NIFTY very likely shows a positive VRP too. The real question
is not *whether* it is positive on average but *how brutal the negative tail is*
(2020-style vol spikes where realized >> implied). I expect **CONDITIONAL GO**.

## Scope
Macro validation of the premium's existence and shape. **No trading rule, no
sizing, no strategy** — a GO means "the premium is real, keep building," not "put
on this trade."

---

## Scoring (filled AFTER running — 2423 days, 2016-07 → 2026-05, real Yahoo data)
- **VRP > 0 on 80.5% of days.** Mean **+2.47 vol pts**, median +3.30. Mean
  VIX/RV ratio **1.29×** (mean India VIX 16.5 vs mean forward realized 14.0).
- **Overwhelmingly significant:** one-sided t=17.6 (p≈1e-65); sign test
  1951/2423 positive (p≈8e-213).
- **Persistent across the decade:** positive mean VRP in **10 of 11** years.
  Every full year 2016–2025 positive (65–99% of days). Only the partial, recent
  2026 stub (97 days, forward window truncated) is flat at −0.04.
- **Brutal negative tail (as expected):** most negative VRP **−64.5 vol pts on
  2020-03-05** (COVID crash — realized detonated past implied); 1st pct −13.7,
  5th pct −6.0, **5.8% of days below −5**.
- **Mean-reverting / harvestable:** VRP is stationary (ADF p≈0.000) with an OU
  half-life of **~24 days** — roughly the VIX horizon itself. VIX also
  stationary.

## VERDICT: **CONDITIONAL GO** ✅
The volatility-selling thesis is **validated at the macro level**: India VIX has
structurally overpriced subsequent realized vol by ~1.3× for a decade, robustly
and persistently. **Keep collecting the option-chain data — the premium the
whole pipeline targets is real.** BUT the negative tail is severe (a single 2020
day at −65 vol pts), so any short-vol strategy built on this **must** be
risk-managed and fractionally sized (per the Exp 016 tail work) — never naive.

## Why this is trustworthy (unlike the Exp 012 micro-signal)
This is a **macro risk premium** — documented across decades and every major
equity index, economically motivated (insurance/variance-premium), and here
significant at p≈1e-65 across 10 years. That is a categorically different kind of
evidence from a data-mined micro-feature (`semivar_ratio`) that failed the Exp
023 snooping control. The VRP does not need a snooping correction: it was not
selected from a search, and its effect size is enormous and stable across
independent years.

## Immediate consequence
Go/No-Go = **GO.** The option-chain collection and the eventual Exp 027 (live
IV-vs-RV VRP on our own collected chains) are justified. The macro dry-run
confirms what the live-lens VRP check (gated at 20 option-days) is being built to
measure at the intraday/expiry level.

---
*Pre-registered 2026-07-01, executed same day as `scripts/exp028_macro_vrp.py`.*
