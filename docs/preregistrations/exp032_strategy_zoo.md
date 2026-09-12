# PRE-REGISTRATION — Experiment 032

**Multi-Strategy Zoo on Collected Option Chains — 7 Locked Candidates (PAPER / RESEARCH ONLY)**

> Status: **PRE-REGISTERED (locked before running).** Every entry rule, exit rule,
> threshold, cost assumption and decision bar below was fixed *before* the
> framework script was executed against the data. No parameter in this document
> was chosen by looking at a result. Because nothing is fitted, the sample is
> effectively out-of-sample with respect to these rules — but see the power
> analysis, which is the single most important section here.

---

## What this is (and is NOT)

A simultaneous, cost-realistic test of **7 structurally-motivated candidate
strategies** against 59 trading days of self-collected NIFTY option-chain
snapshots. Each candidate is a **single-variable change** from an established
prior finding (Exp 003/005/009/013/016/027/028/030/031), not a combinatorial
search over indicators.

It is **NOT** a search for the best strategy, **NOT** a parameter optimization,
and **NOT** a deployment decision. The maximum verdict any candidate can earn
from this experiment is **CANDIDATE FOR FORWARD TESTING**. By construction (see
Power), a "PROMOTE / deploy" verdict is *unreachable* from a 59-day sample and
is therefore excluded from the decision rules.

---

## Data

- **Option chains:** `data/option_chain/YYYY/MM/DD/*.parquet` — **59 trading
  days**, 2026-06-22 → 2026-09-11, 9,816 snapshot files.
  - ~170 snapshots/day, 09:15 → 15:29, ~1-minute to 5-minute cadence.
  - Two listed weekly expiries visible per day; **this experiment uses the NEAR
    expiry only** (single-instrument discipline).
  - **12 expiry days** in sample (weekly, Tuesdays): 06-23, 06-30, 07-07, 07-14,
    07-21, 07-28, 08-04, 08-11, 08-18, 08-25, 09-01, 09-08.
  - Schema per `nifty_quant/data/storage/parquet.py::_CHAIN_COLUMNS`. Note
    `implied_volatility` is **all-NaN by design**; IV is derived on read via
    `nifty_quant.analytics.black_scholes.implied_volatility` (r=0.065, q=0.012,
    matching `research/derive_iv.py` defaults).
  - Observed ATM quote quality: 100% two-sided, **median relative bid-ask spread
    0.24%** of mid. Real spreads are used for fills, not a guessed slippage.
- **India VIX:** `data/vix/YYYY/INDIAVIX_<date>.parquet` (column `india_vix`).
- **Excluded:** the 2026-06-22 session shows 328 distinct snapshot timestamps
  from 170 files (manual-collection artifact, double cadence). It is retained but
  flagged; all strategies key off the *first snapshot at/after the entry clock
  time*, which is robust to cadence irregularity.

### Data-integrity note (found and fixed before running)
Seven phantom nested folders (`2026/07/13/13` … `2026/07/21/21`) existed from an
`scp -r` into a pre-existing directory. Days 13–20 were byte-identical
duplicates; `21/21` contained one unique snapshot (`13_22.parquet`) which was
merged into its parent before the duplicates were deleted. **The true day count
is 59, not the 66 previously reported.** No data was lost.

---

## POWER ANALYSIS (read this before any result)

This is the governing constraint of the entire experiment and it is stated
*before* seeing results so it cannot be rationalized afterwards.

For a strategy trading once per day, the t-statistic on mean return relates to
annualized Sharpe as `t ≈ SR_ann × √(n/252)`.

| Candidate | Tradeable observations | √(n/252) | SR_ann needed for p<0.05 (1-sided) | SR_ann needed to survive FDR across 7 |
|---|---|---|---|---|
| S1 VRP v1.2 | ~47 (non-expiry days) | 0.432 | **3.81** | ~5.9 |
| S2 OI divergence | conditional subset (≤59) | ≤0.484 | **≥3.40** | ~5.2 |
| S3 IV squeeze | ~31 (after 20-day warmup) | 0.351 | **4.69** | ~7.2 |
| S4 PCR extreme | conditional subset (≤59) | ≤0.484 | **≥3.40** | ~5.2 |
| S5 Max-pain pinning | 12 expiry cycles | 0.218 | **7.54** | ~11.6 |
| S6 0-DTE straddle | 12 expiry days | 0.218 | **7.54** | ~11.6 |
| S7 Gap fade | conditional subset (≤59) | ≤0.484 | **≥3.40** | ~5.2 |

**Consequences, accepted in advance:**

1. **A true, excellent, genuinely tradeable edge of Sharpe 1.0–1.5 is
   statistically undetectable in this sample.** It will produce t ≈ 0.4–0.7 and
   be correctly reported as INCONCLUSIVE. Absence of significance here is
   **not** evidence against a strategy.
2. Any strategy that *does* clear p<0.05 in this sample implies an apparent
   Sharpe > 3.4, which is far more likely to be **luck, a look-ahead bug, or a
   cost-model error** than a real edge. A "significant" result is therefore
   treated as a **trigger for adversarial code review**, not as good news.
3. **The test is asymmetrically informative.** We have poor power to confirm a
   positive edge, but reasonable power to *reject* a strategy that is decisively
   negative after realistic costs (a large negative mean is detectable). So the
   realistic productive output of Exp 032 is **eliminating losers and locking
   rules**, not finding winners.
4. Sign-test power: detecting a win-rate edge needs ≥36/59 positive days (61%)
   for p<0.05 one-sided.

**Regime caveat:** 59 days spanning a single calm quarter, with no tail event
(no 2020-style shock, no sustained VIX spike). Short-vol strategies (S1, S5, S6)
are *structurally flattered* by exactly this kind of sample — their risk shows up
only in the tail that is absent here. Per Exp 016/030, the worst block is what
kills them. This sample cannot see it.

---

## Prior-finding alignment (why these 7, and honest priors)

| Candidate | Structural hypothesis | Prior experiment it changes by one variable | Honest prior |
|---|---|---|---|
| **S1** VRP Harvesting v1.2 | Variance risk premium is harvestable on our own chains at a *daily* horizon | Exp 030 v1.0 (21-day blocks, macro proxy) → change **horizon + real fills** | Positive mean, undetectable significance |
| **S2** OI Concentration Divergence | Call-OI build-up during a drop marks writer-defended resistance → reversal | Exp 009/029 (direction unpredictable) → change **conditioning variable to OI flow** | **REJECT expected** |
| **S3** IV Squeeze → RV expansion | Compressed ATM IV under-forecasts subsequent realized vol | Exp 003/013 (vol IS forecastable) → change **predictor to IV percentile** | Weakly negative (it is the mirror of S1) |
| **S4** PCR Extreme Reversal | Sentiment extremes in PCR(OI) mean-revert | Exp 005/009 (direction unpredictable) → change **conditioning variable to PCR** | **REJECT expected** |
| **S5** Max Pain Magnet | Writer hedging pins spot toward max-pain by expiry | Documented pinning effect; new to this platform | Inconclusive (n=12) |
| **S6** 0-DTE Straddle Mispricing | Expiry-day ATM straddles are systematically rich | Exp 027 0-DTE leg (descriptive) → change **to a traded rule with fills** | Positive mean, undetectable (n=12) |
| **S7** Overnight Gap Fade | Large overnight gaps over-react and partially fade | Exp 005/009/029 → change **conditioning variable to gap size** | **REJECT expected** |

S1 and S3 are deliberately **mirror images** (short vol on high-vs-forecast IV,
long vol on low IV percentile). They act as an internal consistency check: if
both appear profitable after costs, there is a bug in the cost or fill model.

---

## Instruments and fills (LOCKED)

Two instrument classes, chosen so each hypothesis is tested with the cheapest
faithful expression and so that **volatility bets and directional bets are never
conflated**:

- **Vol strategies (S1, S3, S5-trade, S6):** ATM straddle on the near expiry, 1
  lot. Strike = the strike nearest spot at the entry snapshot, then **held fixed**
  (the position does not re-strike as spot drifts).
  - Sell → receive **bid**. Buy → pay **ask**. Actual quoted spreads from data.
  - If either leg is not two-sided at entry or exit, the trade is **skipped** and
    recorded as such (no synthetic fill).
- **Directional strategies (S2, S4, S7):** index futures proxy transacted at
  spot, signed exposure of 1 unit of notional.
  - Cost: **1.0 bp of notional per side** slippage + fixed fees. Direction is
    tested on the underlying, not through an option, so option vega/theta cannot
    contaminate a directional result.

### Cost model (LOCKED, with sensitivity)
Indian F&O retail schedule. **These rates are assumptions and are flagged as
such**; they are module constants and the primary result is reported alongside a
sensitivity sweep.

- NIFTY lot size **75**.
- Brokerage **₹20 per order** (per leg, per side).
- **STT 0.1% of premium on the SELL side** of options only.
- Exchange transaction charge **0.03503% of premium**.
- SEBI turnover fee **0.0001%**.
- Stamp duty **0.003% on the BUY side**.
- **GST 18%** on (brokerage + exchange charge + SEBI fee).
- Futures proxy: **0.02% STT on sell**, **0.0019% exchange**, ₹20/order, GST 18%.
- **Sensitivity sweep (pre-registered):** spread multiplier **×0.5, ×1.0, ×1.5,
  ×2.0** applied to the quoted half-spread, reported for every candidate. A
  candidate whose sign flips between ×1.0 and ×2.0 is labelled
  **COST-FRAGILE** regardless of its p-value.

### Capital normalization
Straddle P&L per lot in ₹ is divided by a fixed **MARGIN_PER_LOT = ₹150,000**.
Sharpe, win rate, profit factor and expectancy are **invariant** to this constant
(it scales mean and standard deviation equally); only max-drawdown-% and CAGR
depend on it. Stated so the dependence is explicit.

---

## Strategy definitions (LOCKED — no parameter may be changed after this line)

All signals use only information available at or before the entry timestamp.
"Entry clock" means *the first snapshot at or after that wall-clock time*.

### S1 — VRP Harvesting v1.2
- **Universe:** near expiry, **DTE ≥ 1** (expiry days excluded → they are S6).
- **Signal (causal):** `rv_forecast` = annualized close-to-close realized vol of
  the prior **5** trading days' 15:25 spot (uses only completed prior days).
- **Entry:** 09:20. If `atm_iv_pct − rv_forecast_pct ≥ 2.0` → **SELL** 1 ATM
  straddle.
- **Exit:** 15:25 same day. No overnight hold, no stop.
- **H0:** mean net daily return ≤ 0. **H1:** > 0.
- **Test:** one-sided t-test with **Newey-West HAC** standard errors (lag 5), plus
  one-sided binomial sign test.

### S2 — OI Concentration Divergence
- **Universe:** near expiry, all days.
- **Signal:** from 09:20 to 14:00 — `spot_ret` and `d_call_oi`, `d_put_oi`
  (total near-expiry OI change over that window).
- **Entry:** 14:00. If `spot_ret ≤ −0.30%` **and** `d_call_oi > d_put_oi` →
  **LONG** futures proxy. If `spot_ret ≥ +0.30%` **and** `d_put_oi > d_call_oi` →
  **SHORT** futures proxy.
- **Exit:** 15:25 same day.
- **H0:** mean signed net return ≤ 0. **H1:** > 0.
- **Test:** one-sided t-test (HAC lag 5) + sign test.

### S3 — IV Squeeze → realized-vol expansion
- **Universe:** near expiry, **DTE ≥ 1**, after a **20-trading-day warmup**.
- **Signal:** percentile of today's 09:20 `atm_iv` within the trailing 20 days'
  09:20 `atm_iv` readings (causal, rolling).
- **Entry:** 09:20. If percentile **≤ 20th** → **BUY** 1 ATM straddle.
- **Exit:** 15:25 same day.
- **H0:** mean net daily return ≤ 0. **H1:** > 0.
- **Test:** one-sided t-test (HAC lag 5) + sign test.

### S4 — PCR Extreme Reversal
- **Universe:** near expiry, all days.
- **Signal:** `pcr_oi` on the near expiry at 09:20.
- **Entry:** 09:20. If `pcr_oi ≥ 1.20` → **LONG** futures proxy. If
  `pcr_oi ≤ 0.80` → **SHORT** futures proxy.
- **Exit:** 15:25 same day.
- **H0:** mean signed net return ≤ 0. **H1:** > 0.
- **Test:** one-sided t-test (HAC lag 5) + sign test.

### S5 — Max Pain Magnet
Two pre-registered components; the **statistic is primary**.
- **(a) Pinning statistic (primary):** on each of the 12 expiry days, compare
  `d_open = |spot(09:20) − max_pain(09:20)| / spot(09:20)` with
  `d_close = |spot(15:25) − max_pain(09:20)| / spot(09:20)`. Test whether
  `d_close < d_open` more often than a **same-volatility random-walk null**,
  generated by a stationary bootstrap of that day's own 5-minute returns
  (2,000 draws, block 10). **H0:** pinning contraction is no greater than the
  random-walk null. **H1:** it is greater.
- **(b) Conditional trade (secondary):** expiry day, 09:20, **only if**
  `d_open ≤ 1.0%` → SELL 0-DTE ATM straddle, exit 15:25. This differs from S6 by
  exactly one variable (the max-pain proximity condition).
- **Test (a):** bootstrap p-value. **Test (b):** one-sided t + sign test.

### S6 — Expiry-Day 0-DTE Straddle Mispricing
- **Universe:** the 12 expiry days; the expiring (0-DTE) near expiry.
- **Entry:** 09:20, unconditional. **SELL** 1 ATM straddle.
- **Exit:** 15:25 same day.
- **H0:** mean net daily return ≤ 0. **H1:** > 0.
- **Test:** one-sided t-test (HAC lag 5) + sign test.

### S7 — Overnight Gap Fade
- **Universe:** all days with a prior collected day.
- **Signal:** `gap = spot(09:20 today) / spot(15:25 prior day) − 1`.
- **Entry:** 09:20. If `gap ≥ +0.50%` → **SHORT** futures proxy. If
  `gap ≤ −0.50%` → **LONG** futures proxy.
- **Exit:** **10:15** same day (first hour).
- **H0:** mean signed net return ≤ 0. **H1:** > 0.
- **Test:** one-sided t-test (HAC lag 5) + sign test.

---

## Method (fixed)

1. Build a **causal feature panel** once from the 59 days and cache it
   (`data/derived/exp032_panel_snapshots.parquet`,
   `data/derived/exp032_panel_fixed_strike.parquet`). Per snapshot, near expiry:
   spot, ATM strike, ATM IV, PCR(OI), PCR(volume), max pain, total call/put OI,
   plus the fixed entry-strike leg quotes needed for exact exit fills.
2. **Validate the fast panel against the library path** on a sample day:
   panel `atm_iv`, `pcr_oi` and `max_pain` must match
   `live_lens.snapshot_metrics()` / `analytics.options.*` within tolerance.
   A mismatch aborts the run. (Guards against a fast-path bug producing fake
   edges.)
3. Evaluate all 7 strategies on the panel. Every signal is computed from data
   strictly at or before its entry timestamp; every trailing window is causal.
4. **Walk-forward / stability:** primary result is full-sample (legitimate,
   since nothing is fitted). Additionally report a **60/40 chronological split**
   (train = first 35 days, test = last 24 days) purely as a *stability
   diagnostic*, and per-month means. A candidate whose sign flips between the
   two halves is labelled **UNSTABLE**.
5. **Multiple-testing correction:** Benjamini-Hochberg **FDR at q=0.10** across
   all 7 primary p-values (`statsmodels multipletests(method="fdr_bh")`).
6. **Data-snooping correction:** Hansen's **SPA** + White's **Reality Check**
   via `analytics.data_snooping.reality_check_spa` on the aligned (59 × 7) daily
   net-return matrix, benchmark = zero (cash), block length 10, 5,000 draws.
   Required because 7 > 5 candidates are tested jointly.
7. Report **every** candidate including failures. No candidate may be dropped
   from the report after the fact.

---

## Metrics (all net of costs)

Per candidate: n trades, mean and median net return, win rate, **annualized
Sharpe**, Sortino, max drawdown, profit factor, expectancy, worst single trade,
HAC t-statistic, one-sided p-value, sign-test p-value, BH-adjusted q-value,
cost-sensitivity sign-flip point, train/test split means.

---

## Decision Rules (LOCKED)

Applied mechanically, in this order:

- **REJECT** — mean net return < 0 **and** (HAC t ≤ −1.0 **or** win rate < 45%).
  The sample has enough power to say a rule is a loser after costs.
- **COST-FRAGILE** — mean net > 0 at ×1.0 spread but < 0 at ×2.0 spread. Reported
  as a modifier alongside the primary label.
- **UNSTABLE** — sign of mean net return differs between train and test halves.
  Reported as a modifier.
- **CANDIDATE FOR FORWARD TESTING** — mean net > 0, survives BH at q=0.10,
  **and** SPA p < 0.10, **and** not COST-FRAGILE. *Maximum attainable verdict.*
  Mandatory follow-up: adversarial review for look-ahead before anything else.
- **INCONCLUSIVE — CONTINUE COLLECTING** — mean net > 0 but fails BH or SPA.
  Expected outcome for most candidates given the power analysis. Not a failure of
  the strategy; a failure of sample size.
- **ARCHIVE** — mean net ≈ 0 (|t| < 0.5) with no structural reason to persist.

**No candidate can be promoted to live trading by this experiment.** Even the
top label means "collect more data and forward-test the locked rule".

---

## Expected outcome (honest prior)

I expect: **S2, S4, S7 → REJECT or ARCHIVE** (they contradict Exp 005/009/029,
which are well-established on far larger samples). **S1 and S6 → positive mean,
INCONCLUSIVE** (the VRP is real per Exp 027/028, but 47 and 12 observations
cannot reach significance, and this calm sample flatters short vol). **S3 →
negative or flat**, being S1's mirror. **S5(a) → INCONCLUSIVE** at n=12.

If anything shows an apparent Sharpe > 3, my first hypothesis is a bug, not an
edge. I expect **zero** candidates to reach CANDIDATE FOR FORWARD TESTING, and
that is a successful outcome for this experiment: the rules become locked and
the 59 days become the first tranche of a genuine out-of-sample record.

---

## Scope

Read-only research on collected chains. Backtest with modelled fills from quoted
bid/ask. **No live orders. No paper orders. No deployment.** Real short-straddle
execution (margin calls, gamma path-dependency, assignment, spread widening in
stress) is materially worse than this model.

---

## Scoring (filled AFTER running — 59 trading days, 2026-06-22 → 2026-09-11)

### VERDICT: **NO CANDIDATE SURVIVES CORRECTION.** 2 rejected, 2 archived, 3 inconclusive. Exactly the pre-registered expectation.

Panel-validation gate: **PASS** (max |ΔPCR| = 0, max |ΔMaxPain| = 0,
max |ΔATM_IV| = 0.0000 vol pts vs `nifty_quant.analytics.options` on 2026-07-31).

### Primary results (net of full cost schedule, quoted spreads ×1.0)

| | strategy | n | mean%/trade | win% | Sharpe | maxDD% | PF | worst% | HAC t | p | q(BH) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | VRP Harvesting v1.2 | 28 | **+0.218** | 71 | 0.98 | −12.4 | 1.41 | −11.25 | 0.51 | 0.306 | 0.861 |
| S5 | Max Pain Magnet (traded) | 11 | **+0.282** | 73 | 0.54 | −9.6 | 1.25 | −9.56 | 0.33 | 0.369 | 0.861 |
| S6 | 0-DTE Straddle Mispricing | 11 | **+0.282** | 73 | 0.54 | −9.6 | 1.25 | −9.56 | 0.33 | 0.369 | 0.861 |
| S4 | PCR Extreme Reversal | 29 | −0.014 | 55 | −0.51 | −1.7 | 0.89 | −0.82 | −0.27 | 0.605 | 1.000 |
| S2 | OI Concentration Divergence | 20 | −0.031 | 50 | −1.65 | −0.9 | 0.59 | −0.49 | −0.59 | 0.723 | 1.000 |
| S7 | Overnight Gap Fade | 17 | −0.120 | 24 | −4.48 | −2.1 | 0.23 | −0.60 | −3.10 | 0.999 | 1.000 |
| S3 | IV Squeeze → RV expansion | 13 | −0.633 | 8 | −5.97 | −7.9 | 0.05 | −1.67 | −7.91 | 1.000 | 1.000 |

- **Benjamini-Hochberg FDR at q=0.10: nothing survives.** Smallest q = 0.861.
- **Hansen SPA p = 0.587**, White Reality Check p = 0.542, best candidate S1.
  **SPA does not reject "no candidate beats cash."** The best performer across 7
  tries is fully consistent with luck.
- **Cost sensitivity:** no candidate flips sign between ×0.5 and ×2.0 spreads, so
  none is flagged COST-FRAGILE. Straddle candidates lose only ~0.02–0.04
  pp/trade per half-spread step — quoted ATM spreads (0.24% of mid) are tight
  enough that spread is *not* the binding constraint here, unlike Exp 030 where
  a 3-vol-point assumption killed the edge.
- **Stability:** S1, S4, S5, S6 are all flagged **UNSTABLE** (sign flips between
  the first 35 and last 24 days). S1 runs −0.225% in the train half and +0.551%
  in the test half.

### Verdicts (mechanical application of the locked rules)

| strategy | verdict | modifiers |
|---|---|---|
| S1 VRP Harvesting v1.2 | INCONCLUSIVE — CONTINUE COLLECTING | UNSTABLE |
| S5 Max Pain Magnet | INCONCLUSIVE — CONTINUE COLLECTING | UNSTABLE |
| S6 0-DTE Straddle Mispricing | INCONCLUSIVE — CONTINUE COLLECTING | UNSTABLE |
| S4 PCR Extreme Reversal | ARCHIVE | UNSTABLE |
| S2 OI Concentration Divergence | ARCHIVE | |
| S7 Overnight Gap Fade | **REJECT** | |
| S3 IV Squeeze → RV expansion | **REJECT** | |

### Two bugs found and fixed during execution (recorded, not hidden)

1. **Phantom duplicate day folders.** Seven nested folders
   (`2026/07/13/13` … `21/21`) from an `scp -r` artifact inflated the day count
   to 66. True count is **59**. Days 13–20 were byte-identical; `21/21` held one
   unique snapshot, merged before deletion. No data lost.
2. **Outcome-correlated fill filter (material).** The first implementation
   required a positive *bid* on both legs at exit. On expiry days a call that
   goes worthless quotes 0.0 / 0.1, so 2026-07-07 and 2026-07-14 were silently
   discarded — an exclusion conditioned on a quantity correlated with the
   outcome. Corrected to the direction-aware rule (entry must be two-sided; at
   exit only a live *offer* is required, a zero bid being a legitimate worthless
   leg). **Realized effect: it had been *understating* short-vol performance** —
   both restored days were small wins, and S6's mean moved from −0.042% to
   +0.282%. Removing it was correct on principle regardless of sign.

### S5's pre-registered filter was non-binding — only 6 distinct rules were tested
`|spot − max_pain| / spot` at expiry open had **median 0.07% and maximum 0.26%**
across all 12 expiry days, against a locked threshold of 1.0%. Every day passed,
so **S5 is numerically identical to S6.** The threshold was chosen a priori
without inspecting the data (correct process) but turned out vacuous. The
substantive finding is that **max pain sits essentially on top of spot every
expiry morning and therefore carries no usable information** in this sample.
Consequently the FDR/SPA corrections were applied over 7 nominal but 6 distinct
hypotheses — conservative in the right direction.

**S5(a), the primary pinning test, is also poorly conditioned for the same
reason:** with spot already 0.07% from max pain at open, distance can almost only
grow. Observed contraction 1/11 cycles vs a random-walk expectation of 1.39
(z = −0.39, p = 0.653). **Pinning beyond a random walk is not established**, but
this test had close to no power by construction and should be redesigned (e.g.
conditioning on days where spot starts *far* from max pain) rather than treated
as evidence against pinning.

### Internal consistency check: PASSED
S1 (short vol on high IV-vs-forecast) and S3 (long vol on low IV percentile) are
deliberate mirror images. S1 returns **+0.218%**/trade at a 71% win rate while S3
returns **−0.633%** at an 8% win rate. Signs are opposite and magnitudes are
sensible (intraday long straddle bleeds theta plus spread). Had both appeared
profitable, the cost or fill model would have been broken. They did not.

### Look-ahead audit of the positive candidates: PASSED
Five adversarial checks on S1 and S6: no S1 trade lands on an expiry day (0/28);
independently recomputed trailing RV forecasts match to 1e-10 using strictly
prior closes; entry timestamps strictly precede exit timestamps; and all 11 S6
trades reconcile to the rupee against a hand-rolled P&L from raw quotes.
Correlation between S1's signal strength and realized return is +0.343 (the right
sign, on n=28).

### The number that matters most: S6's payoff shape
Reconstructed S6 trades: **10 wins of +0.64% to +4.86%, and one loss of −9.56%**
(2026-06-23, spot fell 291 points against a straddle that collected only 85.8).
One bad day is roughly **three times the sum of every gain**. Mean is positive
only because that day happened once in 11. This is precisely the Exp 016 / Exp 030
fat-left-tail signature — pennies in front of a steamroller — and an 11-cycle
sample cannot estimate the frequency of the steamroller.

## Honest conclusion

The experiment did what it was designed to do, and what it was designed to do was
**not** find a strategy.

- **The three directional candidates behaved as the platform's prior predicted.**
  S7 (gap fade) is decisively rejected at t = −3.10 with a 24% win rate; S2 and S4
  are indistinguishable from zero. This is a fourth independent replication of
  Exp 005 / 009 / 029: **NIFTY intraday direction is not predictable** from OI
  flow, PCR extremes, or overnight gaps. Conditioning on option-chain variables
  does not rescue it.
- **The volatility-side candidates all lean positive** (S1 +0.218%, S6 +0.282%,
  both ~71–73% win rates), directionally consistent with Exp 027/028's VRP
  finding. **None reaches significance, none survives FDR, and SPA cannot
  distinguish the best from luck.** This is the pre-registered expected outcome,
  not a disappointment: at n = 28 a true Sharpe of 1.0 yields t ≈ 0.5, and the
  observed t is 0.51. The result is *exactly as uninformative as predicted*.
- **Do not read S1's Sharpe 0.98 as an edge.** It sits on 28 trades in one calm
  quarter, flips sign between halves (UNSTABLE), and its worst single day is
  −11.25%. It is a *hypothesis worth forward-testing*, nothing more.
- **What was actually gained:** 7 rule sets are now locked in writing, a
  cost-realistic execution harness exists with real quoted-spread fills, two data
  bugs are fixed, and max pain has been shown to be uninformative at expiry open.
  From 2026-09-12 onward every collected day is a genuine out-of-sample test of
  these exact rules.

**Path forward.** Do not iterate on parameters — that is what made Exp 031 fail.
Keep collecting. Re-run this identical script unchanged at ~120 and ~250 trading
days; at 250 days S1 would need Sharpe ≈ 1.65 to clear p<0.05, which is a fair
test. Archive S2/S4 and drop S7 from future runs. Redesign S5 so its condition
can actually bind. **Nothing here justifies paper trading, let alone live
capital.**

## Reusable output
`scripts/strategy_test_framework.py` (panel cached at
`data/derived/exp032_panel.parquet`; results in
`reports/exp032_strategy_zoo_<stamp>.csv` and `reports/exp032_trades_<stamp>.csv`).

---
*Pre-registered 2026-09-12, executed same day as `scripts/strategy_test_framework.py`.*
