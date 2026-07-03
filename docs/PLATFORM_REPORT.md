# NiftyQuant — End-to-End Platform Report

*How the system works, every experiment run to date and its verdict, and the
technical skills behind the build.*

Compiled: 2026-07-03. Scale at time of writing: ~35 research/utility scripts,
30 test modules / **248 tests**, 10 pre-registered experiments, 9 days of live
option-chain data and counting.

---

## Part 1 — What this platform is (in one paragraph)

NiftyQuant is a **read-only quantitative research platform** for the Indian
index/derivatives market (NIFTY). It records market data (daily/5-minute
candles, live option chains, India VIX), runs **pre-registered, falsifiable
experiments** with walk-forward out-of-sample validation and multiple-testing
control, and reaches **honest verdicts** — including "rejected" and
"insufficient data." It places **no orders**. Its purpose is to find, and just
as importantly to *disprove*, tradeable edges without self-deception.

---

## Part 2 — How it works, end to end

### 2.1 The data layer (`nifty_quant/data/`)
- **Providers** (`providers/angelone.py`, `groww.py`, `base.py`): thin adapters
  over broker APIs. Angel One SmartAPI is the live source (TOTP auth, instrument
  master, historical candles, live LTP, option chains, India VIX). Trading is
  hard-disabled; the provider is read-only by construction.
- **Models** (`models.py`): typed domain objects — `OptionChain`, `OptionQuote`,
  `OptionType` — so the rest of the system speaks options, not raw dicts.
- **Storage** (`storage/parquet.py`): a partitioned Parquet warehouse. Candles
  live under `data/candles/<freq>/<year>/`; option snapshots under
  `data/option_chain/YYYY/MM/DD/HH_MM.parquet` (one append-only file per poll);
  India VIX under `data/vix/`. Writes are atomic (`.tmp` then `replace`) so a
  crash mid-write never corrupts the warehouse.
- **Quality** (`data/quality.py`, `scripts/option_quality_report.py`):
  completeness, cadence/gaps, field sanity, and a GOOD/REVIEW verdict per day.

### 2.2 The daily collection loop (`scripts/collect_market_data.py`)
The engine that runs every trading day. It polls the live feed on a fixed
cadence during 09:15–15:30 IST and writes per-minute option-chain snapshots +
India VIX. It is hardened for unattended operation:
- closed-market guard (weekends / NSE holidays), sandbox `--test` mode;
- retry-with-backoff on rate-limit *and* network errors, per-request throttle;
- crash-proof poll loop (one bad poll never ends the day);
- structured JSON file logging (`logs/collector_<date>.log`);
- tz-naive-IST timestamps (so option data joins cleanly to the candle warehouse);
- Windows keep-awake (`SetThreadExecutionState`) so the box can't sleep;
- `--monitor` for the live lens (below).

Why this exists: intraday option snapshots **cannot be backfilled**. Recording
them now is the one irreversible, high-value action.

### 2.3 The live lens (`nifty_quant/research/live_lens.py`)
With `--monitor`, each poll prints a descriptive situational line (PCR, ATM
straddle & implied move, ATM IV, max pain, spot, VIX) and appends observations.
It also loads the **growing history** of collected days and runs **sample-gated
pre-registered checks** (VRP needs 20 days; 0-DTE straddle needs 8 expiry days),
reporting `ACCUMULATING (n/need)` until the threshold is met. It never claims a
pattern on a small sample and never emits a trade signal.

### 2.4 The analytics toolkit (`nifty_quant/analytics/`)
- `black_scholes.py` — pricing / implied-vol back-solve.
- `options.py` — PCR, max pain, ATM IV, chain utilities.
- `indicators.py` — standard technical indicators (RSI, EMAs, etc.) for the
  benchmark strategy zoo.
- `stat_tests.py` — Hurst, ADF, OU half-life, Lo-MacKinlay variance ratio,
  Kelly / fractional-Kelly, **Hill estimator**, **GPD peaks-over-threshold**.
- `vol_estimators.py` — Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang.
- `data_snooping.py` — stationary bootstrap, **White Reality Check**,
  **Hansen SPA**.

### 2.5 Backtest & feed (`nifty_quant/backtest/`, `feed/`)
An event-driven backtest engine (`engine.py`) with a simulated broker
(slippage + fees), portfolio, risk engine, and a strategy zoo
(`strategies/`). A replay feed (`feed/replay.py`) streams the warehouse through
strategies deterministically; a paper feed supports live dry-runs. This is how
the directional-strategy experiments (Exp 001–009) were run and closed.

### 2.6 The research engine (`nifty_quant/research/`, `scripts/expNNN_*.py`)
Every experiment is a pre-registration (`docs/preregistrations/expNNN_*.md`)
with **locked decision rules**, plus a runnable script. Shared machinery:
expanding walk-forward folds, HAC/Newey-West errors, block bootstrap,
Benjamini-Hochberg FDR, and (now) the SPA snooping control. Verdicts are one of
SUPPORTED / REJECTED / REDUNDANT / INCONCLUSIVE / INSUFFICIENT DATA.

### 2.7 Validation (`nifty_quant/validation/`)
Performance metrics (Sharpe, drawdown, etc.), drift detection, and alerting —
the layer that would guard a strategy in production once one exists.

---

## Part 3 — The daily routine

1. Before 09:15 IST: `python scripts/collect_market_data.py --num-expiries 2
   --strike-band-pct 6 --poll 120 --request-pause 1.2 --monitor`
2. It collects unattended and auto-exits after 15:30.
3. After close: `python scripts/option_quality_report.py --date <today>` →
   confirm GOOD.
4. Periodically: as day-count crosses thresholds, run/extend offline experiments.
5. Commit code + pre-registrations (data stays local, gitignored).

---

## Part 4 — Every experiment and its verdict

The campaign's honest arc: **direction is not forecastable; volatility is; and
the one robust edge is the volatility risk premium.**

### Phase 1 — Directional branch (Exp 001–009): CLOSED
Price-only / technical strategies over the 5m warehouse with costs, via the
backtest zoo and studies of gaps, intraday autocorrelation, and volatility
decomposition (`scripts/gap_study.py`, `intraday_autocorr.py`, `vol_decomp.py`,
`vol_clock.py`, `run_*`). **Result: no strategy beat buy-and-hold after costs;
intraday returns are ≈ a random walk.** The directional branch was closed.

### Phase 2 — Volatility structure

| Exp | Question | Verdict |
|-----|----------|---------|
| **003** RV persistence | Is daily RV forecastable? | **SUPPORTED** — volatility clusters; HAR-RV works |
| **010** Volume–volatility | Does volume inform vol? | **BLOCKED** — NIFTY cash index has zero volume |
| **011** Leverage × session | Is the leverage effect time-of-day dependent? | **REJECTED** — effect is uniform across the day |
| **012** Semivariance ratio | Does down/up semivariance add to HAR? | **SUPPORTED → later DOWNGRADED** (see Exp 023) |
| **013** RV forecast horse-race | HAR vs EWMA/GARCH/EGARCH? | HAR **best** (QLIKE + Diebold-Mariano); others significantly worse |
| **014** Jump detection | How much of vol is jumps; are they clustered? | ~90%+ **continuous**; jumps rare, **not clustered**, no forecast value |
| **016** Tail risk (EVT) | How fat are NIFTY tails? | **Fat** (excess kurtosis +2.08, Hill α≈3); Gaussian ES understates ~1.2× |
| **017** Regime HMM | Are there real, persistent regimes? | **2 regimes** (calm/turbulent, ~11-day dwell) but **no alpha** over HAR |
| **024** OHLC estimators | Does Yang-Zhang/range beat 5m RV? | **RV5M best**; Parkinson/GK tied; naive close-to-close +31% worse |

### Phase 3 — Rigor & external data

| Exp | Question | Verdict |
|-----|----------|---------|
| **018** Global spillover | Does overnight US predict NIFTY? | Open-gap **coupling ~21%** but **NOT tradeable**; no post-open edge |
| **023** Reality Check / SPA | Do our best findings survive data-snooping? | **No** — `semivar_ratio` fails SPA (p=0.22); **Exp 012 downgraded 60→35** |
| **029** Next-candle | Can a candle pattern predict the next candle? | **Direction NOT predictable** (50.8%, loses after costs); **size** is |

### Phase 4 — The thesis test

| Exp | Question | Verdict |
|-----|----------|---------|
| **028** Macro VRP (10y) | Is the volatility risk premium real on NIFTY? | **GO** — India VIX > realized 80.5% of days over 10y, mean +2.5 vol pts, VIX/RV 1.29×, p≈1e-65, positive 10/11 years; severe COVID tail → size fractionally |

**Net position:** one genuine, robust edge — the **volatility risk premium** —
validated at the macro level and now being measured live on our own collected
option chains (Exp 027, gated at ~20 collected days). Everything directional was
correctly rejected; one micro-signal that looked good was talked back down by the
snooping control. That is the platform working as designed.

---

## Part 5 — Technical skills used to build this

**Software engineering**
- Python (modular package architecture, typed domain models, dependency
  injection for providers), event-driven design, graceful signal handling,
  Windows OS integration (`SetThreadExecutionState`), TOTP/2FA auth flow,
  retry-with-backoff, API rate-limiting, structured JSON logging, reproducibility
  tooling (`repro.py`, lineage).

**Data engineering**
- Partitioned columnar (Parquet/PyArrow) time-series warehouse; append-only,
  crash-safe atomic writes; per-minute snapshot design (no read-modify-write);
  data-quality reporting; external-data ingestion & caching (`yfinance`);
  timezone discipline for cross-dataset joins.

**Quantitative finance**
- Realized volatility & **HAR-RV**; semivariance; **bipower variation / jump**
  decomposition; **GARCH/EGARCH**, **EWMA**; OHLC vol estimators
  (**Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang**); **Black-Scholes** &
  implied vol; option analytics (PCR, max pain, ATM straddle, implied move);
  **volatility risk premium**; **HMM** regime modelling; position sizing via
  **Kelly / fractional-Kelly**.

**Statistics & research methodology**
- **Pre-registration** with locked decision rules; **expanding walk-forward**
  out-of-sample validation; **QLIKE / RMSE / MAE** loss; **Diebold-Mariano**
  forecast tests; **HAC/Newey-West** robust errors; **block / stationary
  bootstrap**; multiple-testing control via **Benjamini-Hochberg FDR** and
  **White Reality Check + Hansen SPA**; **Extreme Value Theory** (Hill, GPD/POT);
  stationarity & mean-reversion (**ADF**, **Hurst**, **Ornstein-Uhlenbeck
  half-life**, **Lo-MacKinlay variance ratio**); rigorous **look-ahead
  avoidance** (filtered vs smoothed states, strict-backward joins, lagged
  features).

**Machine learning**
- `scikit-learn` (logistic regression, gradient-boosted trees), `hmmlearn`
  (Gaussian HMM) — used carefully and always walk-forward, with explicit
  rejection of high-parameter models (e.g. LSTM) on small samples.

**Testing & quality**
- `pytest` with **248 tests** across 30 modules; primitives validated against
  **known synthetic processes** (random walk, AR(1), Pareto/Student-t tails,
  simulated Brownian OHLC); provider fakes; deterministic replay tests.

**Broker / market integration**
- Angel One SmartAPI (auth, instrument master, historical + live data, option
  chains, India VIX); Groww adapter; NSE session/holiday calendar handling.

---

## Part 6 — Honest limitations

- **Sample size:** ~2 years of intraday candles and only ~9 days of option
  chains. The decisive tests (live VRP, Exp 027) are still forward-only.
- **One market, one instrument:** NIFTY index. No cross-sectional breadth.
- **No live trading:** by design. Execution, capacity, and slippage at size are
  untested (deferred experiments 021/024-style risk work).
- **The edge is modest and risky:** the VRP is real but carries a brutal
  left tail (2020: −65 vol points). Any deployment must be fractionally sized.
- **Solo-scale infrastructure:** strong research discipline (~7/10 as infra),
  but not remotely the data/latency/breadth of an institutional quant desk.

The platform's true product is not a strategy — it is **trustworthy verdicts**.
It has proven a real macro edge (VRP) and disproven many tempting illusions
(candle prediction, chart patterns, jump timing, a data-mined vol feature). That
honesty is the asset.
