# PRE-REGISTRATION — Experiment 030

**VRP Harvesting Strategy v1.0 — 10-Year Macro Backtest (PAPER ONLY)**

> Status: **PRE-REGISTERED (locked before running).** Rules and decision
> thresholds fixed in advance, on purpose, *before* we have enough live
> option-chain data — so nothing is tuned to fit later data.

---

## What this is (and is NOT)
A rules-based backtest of harvesting the volatility risk premium that Exp 028
validated (India VIX > realized ~80% of days over 10y). It answers: **would a
simple, risk-managed short-volatility rule have produced an attractive,
survivable risk-adjusted return over 2016–2026, including the 2020 crash, after
costs?**

It is **NOT** a live strategy, **NOT** a deployment, and uses a **VRP-proxy
P&L**, not real option fills. A positive result earns *paper trading on live
collected chains* (Exp 027+), nothing more. Real short-straddle execution
(slippage, gamma path-dependency, assignment) will be **worse** than this proxy.

## Data
The Exp 028 cache: 10y daily **^NSEI** + **^INDIAVIX** (`data/external/
macro_vrp_daily.parquet`), 2016–2026.

## Strategy rules (v1.0 — LOCKED, deliberately simple, non-optimized)
- **Signal:** always short 1-month volatility (harvest the structural premium);
  no market-timing overlay in v1.0.
- **Horizon:** non-overlapping 21-trading-day blocks (~monthly rebalance) → ~115
  trades over 10y. Non-overlapping avoids fake sample inflation.
- **Per-trade P&L (vol-swap approximation, in vol points):**
  `pnl = VIX_start − RV_realized − cost`, where `RV_realized` is the annualized
  realized vol over the block and `cost` is a round-trip transaction cost.
- **Cost:** round-trip **1.5 vol points** (conservative-ish for a 1-month ATM
  straddle); report **sensitivity at 0 / 1.5 / 3.0** vol points.
- **Sizing (two variants, both fixed-rule, no fitting):**
  1. **Constant** notional (naive short vol).
  2. **Vol-targeted:** weight ∝ 1/VIX_start (sell *less* vol when vol is already
     high) — the standard tail-dampening rule motivated by Exp 016/028's fat
     left tail.
- **No stops** modelled in v1.0 (a 1-month variance position can't be cleanly
  stopped mid-month); tail control is via the sizing rule and is examined
  explicitly.

## Metrics (all after cost)
- Mean / median VRP capture per trade (vol pts) and t-stat; hit-rate.
- Annualized **Sharpe** and **Sortino** of the P&L series (scale-invariant).
- Cumulative-vol-point equity curve; **max drawdown** (in vol pts and as a % of
  a stated capital base under the vol-target sizing).
- **Worst single trade** and the **2020 block(s)** specifically.
- **Per-year** mean capture (stability).

## Hypotheses
- **H0:** after costs, short-vol harvesting has non-positive expected P&L / a
  non-survivable tail.
- **H1:** it has positive, reasonably stable, risk-adjusted P&L after costs, with
  a tail that vol-target sizing keeps survivable.

## Decision Rules (LOCKED)
- **VIABLE (→ proceed to live paper trading)** if ALL: net-of-cost mean capture
  > 0 with t-stat ≥ 2; annualized Sharpe ≥ 0.5; profitable in ≥ 7 of 10 years;
  **and** under vol-target sizing the worst single trade loss ≤ the prior
  cumulative gain (no ruin) AND max drawdown ≤ 40% of the stated capital base.
- **MARGINAL** if positive after cost but Sharpe 0.2–0.5, or it fails exactly one
  secondary bar → keep researching, do not paper-trade yet.
- **NOT VIABLE / REDESIGN** if mean capture ≤ 0 after cost, or a single block
  (e.g. 2020) exceeds all prior cumulative gains (ruin) even with vol-target
  sizing, or profits concentrate in < 5 years.

## Expected outcome (honest prior)
Given Exp 028, the *carry* is almost certainly positive after a 1.5-pt cost
(mean VRP was +2.5, so ~+1 net). The real question is **tail survivability**: a
2020-style block can wipe out a year of carry. I expect **MARGINAL-to-VIABLE
with vol-target sizing, NOT VIABLE naive** — i.e. the premium is real but only
harvestable with disciplined sizing. Anything rosier should be distrusted, and
even a VIABLE verdict is *paper-only*.

## Scope
Backtest of a VRP proxy on macro data. **No live orders. No real option fills.**

---

## Scoring (filled AFTER running — 116 non-overlapping 21-day blocks, 2016–2026)

### VERDICT: **MARGINAL** — do NOT paper-trade yet; keep researching.

- **The edge is real but thin, and brutally cost-sensitive:**

  | round-trip cost | mean net (vol pts) | hit-rate | Sharpe |
  |-----------------|--------------------|----------|--------|
  | 0.0 | +2.47 | 81% | 1.36 |
  | **1.5 (base)** | **+0.97** | **69%** | **0.53** |
  | 3.0 | −0.53 | 52% | **−0.29** |

  At a realistic 1.5-vol-pt cost the premium survives (Sharpe ~0.5); at 3.0 pts
  it **vanishes and goes negative**. Real ATM-straddle execution can easily cost
  2–3+ vol points, so the tradeable edge is far thinner than the gross premium
  (Sharpe 1.36) suggests.
- **Sizing (after 1.5-pt cost):** constant — Sharpe 0.53, t=1.7, worst −41.9,
  max DD −72.6; **vol-target (weight∝1/VIX)** — Sharpe 0.64, t=2.0, worst −29.1,
  max DD −48.5. Vol-target sizing materially improves the tail, confirming the
  Exp 016 thesis — but max drawdown (−48.5) still breaches the locked −40 budget.
- **Per-year:** positive in **8/11** years; negative 2020, 2024, and the partial
  2026 stub.
- **Tail / near-ruin:** worst block **2020-03-12** — sold VIX 41.2, realized blew
  to **81.6** → **−41.9 vol pts** in one month. It was survivable *only because*
  ~4 years of carry (+42.1) had accumulated first. Enter this strategy right
  before a 2020 and you're wiped. Classic "pennies in front of a steamroller."

### Why MARGINAL not VIABLE (against the locked rules)
Fails two bars: constant-size t=1.7 (< 2) and vol-target max DD −48.5 (< −40
budget). Positive after cost with vol-target Sharpe 0.64 → clears the MARGINAL
threshold, not VIABLE.

## Honest conclusion
This tempers the Exp 028 "GO" correctly: the VRP **exists** and is real, but the
**tradeable** version after realistic costs is a modest, cost-fragile,
fat-tailed carry — Sharpe ~0.5–0.6 at best, with a 2020-type block able to erase
years of gains. It is **not** a push-button money machine. Path forward before
any paper trading: (v1.1) add a volatility-regime filter (stand aside when VIX is
spiking / term structure inverts), tighter cost modelling, and a hard tail hedge
(long wings) — then re-test. Even a future VIABLE verdict is **paper-only**.

---
*Pre-registered 2026-07-03, executed same day as `scripts/exp030_vrp_strategy.py`.*
