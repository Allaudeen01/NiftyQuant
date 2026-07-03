# PRE-REGISTRATION — Experiment 031

**VRP Harvesting Strategy v1.1 — Regime Filter + Tail Hedge + Kill-Switch (PAPER)**

> Status: **PRE-REGISTERED (locked before running).** Thresholds fixed a-priori
> (taken as given, NOT grid-searched) before running.

---

## Honest overfitting caveat (read first)
v1.1 adds four risk rules to the SAME 10-year window where we already know 2020
hurt v1.0. Even with a-priori thresholds, the *specification* is chosen with
hindsight (we know a vol spike is the enemy). Therefore:
- **An in-sample Sharpe improvement here is PROVISIONAL, not proof.**
- Thresholds are the ones proposed, used verbatim — **no searching/tuning**. If I
  had to search them, the result would be discarded as data-mined.
- The decisive validation remains **forward** (Exp 027, live chains). This
  experiment only asks: "do sane, pre-committed risk rules improve the historical
  risk profile without destroying the carry?"

## Rules (v1.1 = v1.0 + four LOCKED overlays)
Baseline v1.0: non-overlapping 21-day short-vol blocks, P&L = VIX_start −
RV_realized − cost, vol-target sizing (weight ∝ 1/VIX). Overlays:

1. **Regime filter (stand aside unless ALL hold — all causal, info at entry):**
   - `VIX_start < 22` (avoid extreme vol levels), AND
   - `VIX_start ≤ VIX_{start−5d}` (VIX falling or flat, not rising), AND
   - trailing 21-day realized vol ≤ **1.2×** its own trailing 126-day average
     (RV not spiking). If any fails → **no trade** that block (P&L 0, no cost).
2. **Tail hedge (loss-cap proxy):** when `VIX_start > 18`, cap the block loss at
   **−15 vol pts** (proxy for buying OTM wings) at a hedge cost of **1.5 vol
   pts**. (On macro data we cannot price real puts; this is a conservative
   stand-in.)
3. **Kill-switch (on the % equity curve):** if peak-to-trough drawdown > **20%**
   → halve position size; if > **30%** → size 0 (halt) until equity recovers
   above the −20% line.
4. **Cost:** base **1.5 vol pts** round-trip; report sensitivity at 0 / 1.5 /
   3.0. (Real per-trade bid-ask from collected chains applies only to the live
   version, not this macro proxy.)

Interpretation convention (for %/drawdown): **1 vol pt of P&L = 1% of capital**
at mean vol-target weight (an explicit, modest leverage choice — the same one
used in v1.0's scoring, so v1.0↔v1.1 drawdowns are comparable).

## Data
Exp 028 cache (`data/external/macro_vrp_daily.parquet`), 10y daily ^NSEI + VIX.

## Metrics (after cost) & comparison
Report v1.1 and re-report v1.0 (vol-target) on the identical blocks: mean net
capture, hit-rate, annualized Sharpe & Sortino, max drawdown %, worst block,
per-year mean, and **# trades taken vs skipped** (the filter reduces n → weaker
t-stats, an honest cost).

## Decision Rules (LOCKED)
- **CLEARS THE BAR (→ pre-register live paper, wait for Exp 027)** if ALL:
  annualized Sharpe ≥ **0.7**, max drawdown ≤ **30%**, positive in ≥ **7/10**
  years, worst block survivable (≤ prior cumulative gain), **and** it does this
  while still taking a meaningful number of trades (≥ 50% of eligible blocks —
  guards against a filter that only "works" by never trading).
- **IMPROVED-BUT-SHORT** if it beats v1.0 on Sharpe and drawdown but misses one
  bar → v1.2 iteration; no paper trading.
- **NO IMPROVEMENT** if Sharpe/drawdown are not better than v1.0 → the overlays
  don't help; rethink.
- **Overriding rule:** even CLEARS-THE-BAR is **provisional/in-sample**; live
  paper trading waits on Exp 027 (20+ collected days) regardless.

## Expected outcome (honest prior)
The filter should cut the 2020 damage (it stands aside when VIX rises/‌is high)
and the kill-switch caps drawdown, so Sharpe and max-DD should improve — but
partly *because* we designed around the known crash. I expect
**IMPROVED-BUT-SHORT to CLEARS-THE-BAR in-sample**, with the honest asterisk that
forward data is the real judge. Watch for the filter improving Sharpe merely by
skipping most trades (guarded by the ≥50%-participation rule).

## Scope
Backtest of risk overlays on a VRP proxy. **No live orders. Paper only.**

---

## Scoring (filled AFTER running — 116 blocks, 2016–2026)

### VERDICT: **NO IMPROVEMENT** — v1.1 is strictly WORSE than v1.0. Do not adopt.

Head-to-head at base cost (1.5 vol pts; 1 vol pt = 1% capital):

| strategy | Sharpe | maxDD% | CAGR% | worst% | trades |
|----------|--------|--------|-------|--------|--------|
| v1.0 (vol-target) | **0.64** | −42.8 | **+9.3** | −29.1 | 116 |
| v1.1 (overlays)   | **0.09** | −29.5 | **−0.1** | −29.1 | 42 |

- The regime filter admits only **42/116 blocks (36%)** — it fails the ≥50%
  participation guard outright, and it **destroyed the edge**: Sharpe 0.64 → 0.09,
  CAGR +9.3% → −0.1%.

### Why it backfired (the real lesson)
**Standing aside when VIX is high/rising is exactly backwards for VRP.** The
volatility risk premium is *largest* when VIX is elevated — that's the
compensation for bearing crash risk. The regime filter systematically skips the
fat-premium, high-VIX trades and keeps only calm, **thin-premium** low-VIX
trades — where the fixed 1.5-pt cost eats most of the carry (cost sensitivity
confirms: gross Sharpe 1.21 at 0 cost collapses to 0.09 at 1.5, because the
surviving trades have little premium to begin with). The intuition "avoid
volatile regimes" is a classic VRP mistake, and the backtest caught it.

- The kill-switch/hedge *did* trim max drawdown (−42.8% → −29.5%), but at the
  cost of nearly all return — a bad trade. And the worst single block is
  **unchanged (−29.1%)**: the tail hedge keys on *entry* VIX > 18, so it does
  **not** protect the dangerous calm→crash block (enter at VIX 16, realized
  spikes) — precisely the case you most need covered.
- 2020 still negative (−6.45%); the filter reacts too late to a fast spike.

### Consequence for the roadmap
- **Reject the "regime filter to avoid vol" approach.** For VRP the edge lives in
  the high-vol entries; the goal is to *survive* them (tail hedge + sizing), not
  *avoid* them.
- A sensible v1.2 would go the other way: keep participation high, make the tail
  hedge trigger on *forecast/position risk* (not just entry VIX) so it covers
  calm→crash, and lean on sizing rather than avoidance. But **do not keep adding
  rules to this same 10y window** — that is how overfitting happens. The
  decisive test is forward (Exp 027). v1.0 with disciplined sizing remains the
  better in-sample baseline for now, and it is still only MARGINAL.

---
*Pre-registered 2026-07-03, executed same day as `scripts/exp031_vrp_strategy_v1_1.py`.*
