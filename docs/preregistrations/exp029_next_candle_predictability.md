# PRE-REGISTRATION — Experiment 029

**Does the Last 5-Minute Candle Predict the Next Candle's Direction & Magnitude?**

> Status: **PRE-REGISTERED (locked before running).** This directly tests the
> user's question: "can I see a candle pattern and predict the next candle up/down
> by 10–20 points?" Decision rules fixed before results are seen.

---

## Scientific Question
Using the shape of the current 5-minute candle (and a few recent ones), can we
predict, out-of-sample and after costs:
1. the **direction** of the next 5-minute candle (up/down), and
2. its **magnitude** (how many points it moves)?

## Object of study
NIFTY 5-minute candles, regular session (09:15–15:30), 2024–2026 (~490 days,
~74 bars/day → ~36,000 candles). The prediction target is the **next** bar's
close-to-close return, formed only from information available at the close of the
current bar (no look-ahead).

## Predictor features (from the current candle + recent context — all causal)
- `ret` — current bar return (sign & size)
- `body` = (close−open)/open; `range` = (high−low)/open
- `body_frac` = |close−open| / (high−low) (candle "conviction")
- `upper_wick`, `lower_wick` (normalized)
- `ret_lag1`, `ret_lag2` — prior two bar returns (momentum/reversal)
- `dist_vwap` — distance of close from the session VWAP so far
- `minute_of_day` — intraday seasonality control

## Models
- **Direction:** logistic regression (next return > 0?) and, as a stricter check,
  a gradient-boosted tree — both walk-forward. Baseline = always predict the
  majority class / unconditional up-rate.
- **Magnitude:** OLS and gradient-boosted regressor for next |return|, vs the
  naive baseline "next move ≈ recent average move" (a volatility forecast).

## Method (fixed, no look-ahead)
- **Expanding walk-forward**, 4 folds, first 50% train. Features standardized on
  train only. Predict each test bar using only prior bars.
- **Direction metrics:** out-of-sample accuracy and ROC-AUC vs baseline; a
  directional trading proxy P&L **after costs** (see below).
- **Magnitude metrics:** OOS R² and QLIKE vs the recent-average-move baseline.
- **Costs:** NIFTY ≈ 24,000. Round-trip cost assumed **1.5 bps** (~0.015%,
  conservative for liquid futures incl. spread+slippage+fees ≈ 3.6 index points).
  A directional signal must beat this to matter. A "10-point" move ≈ 0.042%.

## Decision Rules (LOCKED)
- **DIRECTION PREDICTABLE** only if ALL: OOS accuracy > 53% (vs ~50%) with the
  binomial test p<0.01, ROC-AUC > 0.55, sign stable across all 4 folds, **and**
  the after-cost directional P&L is positive in ≥3/4 folds. Otherwise
  **DIRECTION NOT PREDICTABLE.**
- **MAGNITUDE PREDICTABLE (size, not direction)** if next-|return| OOS R² > 0 in
  ≥3/4 folds over the recent-average baseline. (Expected YES — this is just
  volatility clustering; it is NOT a direction edge.)
- **NET VERDICT — CAN YOU PREDICT THE NEXT CANDLE?** = "yes" only if DIRECTION is
  predictable AND survives costs. If only MAGNITUDE is predictable, the honest
  answer is **"you can predict the SIZE of the move, not the direction."**

## Expected outcome (honest prior — stated before running)
Based on Exp 001–009 (directional branch closed, intraday returns ≈ random walk)
and Exp 023 (single-feature significance evaporates under scrutiny), the expected
result is: **DIRECTION NOT PREDICTABLE** (accuracy ≈ 50–51%, negative after
costs), **MAGNITUDE PREDICTABLE** (volatility clusters). I.e. you can forecast how
big the next candle is, not which way it goes. If direction *does* clear the bar,
it must then face an Exp 023 snooping check before any trust.

## Scope
Falsification test of intraday candle-pattern predictability. **No trading rule
is deployed regardless of outcome** — a positive result would only earn a
dedicated follow-up, not a live trade.

---

## Scoring (filled AFTER running — 35,581 NIFTY 5m candles, 2024-06 → 2026-06)
Up-rate 50.3%; mean |move| **11.4 index points (0.047%)**; round-trip cost ~3.6
points (1.5 bps).

- **DIRECTION — NOT PREDICTABLE.** Walk-forward OOS accuracy **50.8%** (baseline
  50.3%), mean **AUC 0.508** (0.5 = coin flip). After costs the directional
  trading proxy **lost money in 4/4 folds** (thousands of index points bled to
  spread/slippage). A candle's shape carries essentially no information about the
  next candle's direction.
- **MAGNITUDE — PREDICTABLE (size, not direction).** Next-|move| OOS R² positive
  in **3/4 folds** (up to +0.06) over the recent-average baseline — i.e.
  volatility clusters, so you can forecast *how big* the next candle is. This is
  the same forecastable-vol result as Exp 003/013, not a direction edge.

## NET VERDICT: **You can predict the SIZE of the next candle, not its direction.**
Direction is a coin flip that loses after costs. The "up/down by 10–20 points"
prediction the question asked for is **not achievable** on this data — confirmed
empirically on 35k candles, consistent with Exp 001–009 (closed directional
branch) and Exp 023 (single-signal significance evaporates under scrutiny).

## The takeaway (why this matters)
This is the whole reason the platform points at volatility, not direction. You
cannot know *which way* the next candle goes, but you *can* know it will likely be
small or large — and the market systematically **overpays** for that size
(Exp 028 macro VRP, GO). The edge is selling overpriced move-size with tight risk
control, never guessing direction.

---
*Pre-registered 2026-07-01, executed same day as `scripts/exp029_next_candle.py`.*
