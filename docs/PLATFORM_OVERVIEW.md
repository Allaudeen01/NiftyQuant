# NiftyQuant — How the Platform Works

*A complete, honest walkthrough of what this system does, how each piece fits
together, and what it can and cannot deliver.*

Last updated: 2026-07-01

---

## 1. What this is (and what it is not)

NiftyQuant is a **systematic quantitative research platform** for the Indian
index/derivatives market (NIFTY). Its job is to let one person run **rigorous,
falsifiable, pre-registered experiments** on market data and reach *honest*
verdicts — including "rejected" and "insufficient data" — without fooling
themselves.

**It IS:**
- A read-only market-data recorder (spot, option chains, India VIX).
- A statistics + options analytics toolkit.
- A disciplined experiment framework with pre-registration and FDR control.
- A live "situational awareness" lens that grows into pattern checks as data
  accumulates.

**It is NOT:**
- A live trading bot. It places **no orders**, ever.
- A "press start and it finds a money-printing strategy" loop. That is
  data-dredging and is deliberately refused (see §7).
- A directional edge/holy grail. The evidence so far says price direction is
  not forecastable after costs; the only forecastable dimension is
  **volatility**.

---

## 2. The core philosophy

Every claim must survive the same gauntlet real quant research uses:

1. **Pre-registration** — hypothesis, variables, method, and decision rules are
   locked in a `docs/preregistrations/expNNN_*.md` file *before* the experiment
   runs. Thresholds cannot be moved after seeing results.
2. **Walk-forward / out-of-sample** — models are scored on data they were not
   fit on (expanding folds), not on the full in-sample fit.
3. **Multiple-testing control** — any screen over many candidates uses
   Benjamini-Hochberg FDR correction, and survivors get a *fresh*
   pre-registered confirmation test.
4. **Honest verdicts** — SUPPORTED / REJECTED / REDUNDANT / INSUFFICIENT DATA.
   A negative result is a real result and is logged as such.
5. **No look-ahead** — predictors are lagged; a bug that leaked future
   information was found and fixed during the feature search.

The point of the discipline is to **prevent self-deception**. Most "patterns"
found on two years of one index are noise; the framework's job is to make that
obvious rather than hide it.

---

## 3. The daily loop (data collection)

This is the part that runs every trading day. It is the platform's engine
because **intraday option snapshots cannot be backfilled** — if you don't record
them today, that day is gone forever.

### 3.1 The collector — `scripts/collect_market_data.py`

Read-only. Polls the live Angel One feed on a fixed cadence during the regular
session (09:15–15:30 IST) and writes:

- **Option-chain snapshots** → `data/option_chain/YYYY/MM/DD/HH_MM.parquet`
  (one append-only file per poll; nearest N expiries share the file).
- **India VIX series** → `data/vix/YYYY/INDIAVIX_<date>.parquet`.

Steady-state command:

```
python scripts/collect_market_data.py --num-expiries 2 --strike-band-pct 6 --poll 120 --request-pause 1.2 --monitor
```

Hardening built in over several days of live use:
- **Closed-market guard** — refuses to run on weekends / NSE holidays.
- **Sandbox mode** (`--test`) — writes to `data_test/`, never production.
- **Rate-limit respect** — `--request-pause` throttle plus spacing between
  expiry requests; retry-with-backoff on both rate-limit *and* network errors.
- **Crash-proof poll loop** — one bad poll (or a write error) logs and continues
  instead of ending the day.
- **File logging** — structured JSON to `logs/collector_<date>.log`.
- **tz-naive IST timestamps** — consistent with the daily-candle warehouse so
  option data joins cleanly to realized volatility later.
- **Windows keep-awake** — `SetThreadExecutionState` stops the machine
  suspending mid-session (display may still sleep; the process won't).
- **`--monitor`** — turns on the live lens (see §5).

### 3.2 Data-quality report — `scripts/option_quality_report.py`

Run after each session: `python scripts/option_quality_report.py --date YYYY-MM-DD`

Reports snapshot count, first/last time, completeness (both expiries present),
cadence and max gap, field sanity (OI>0, two-sided quotes, crossed quotes,
spot range, strikes/snapshot), VIX coverage, and a **GOOD / REVIEW** verdict.
IV is 0% *by design* — it is derived later from price via Black-Scholes, not
stored raw.

### 3.3 Live spot correctness

The provider's `get_spot()` reads the **live index LTP** via `getMarketData`
(with a daily-candle fallback), not the lagged daily-candle close. Verified
within ~0.1% of official NSE prints; residual is explained by opening-auction
timing and NSE's 30-minute-average close.

---

## 4. The research engine (offline experiments)

This is where accumulated data becomes knowledge. Experiments are numbered and
each has a pre-registration + a runnable script.

### 4.1 Statistics toolkit — `nifty_quant/analytics/stat_tests.py`

Pure, textbook, first-principles primitives (validated against synthetic cases):
- `hurst_exponent` — trending (>0.5) vs mean-reverting (<0.5) vs random walk.
- `adf_test` — Augmented Dickey-Fuller stationarity.
- `half_life` — Ornstein-Uhlenbeck mean-reversion half-life (a principled,
  non-curve-fit holding-period estimate).
- `variance_ratio` — Lo-MacKinlay heteroskedasticity-robust VR test.
- `kelly_fraction` / `kelly_gaussian` / `fractional_kelly` — position sizing,
  with half-Kelly as the sane default (full Kelly is far too aggressive under
  estimation error and fat tails).

### 4.2 Bounded feature search — `scripts/volatility_feature_search.py`

A **one-shot** (NOT a loop), FDR-corrected screen of ~15 volatility-structure
features as increments to a HAR-RV baseline. It caught and fixed a look-ahead
bug (candidates weren't lagged). Survivors: `semivar_ratio`, `intraday_range`.
A survivor is only a *candidate* — it must then pass a fresh pre-registered test.

### 4.3 Experiment status so far

| Exp | Topic | Verdict |
|-----|-------|---------|
| 001–009 | Price-only / directional branch | **CLOSED** — no edge beats buy & hold after costs; intraday direction unpredictable |
| 010 | Volume–Volatility | **BLOCKED** — NIFTY cash index has zero volume; needs backfilled NIFTY *futures* |
| 011 | Leverage × Session (`scripts/leverage_session.py`) | **REJECTED** session-dependence — leverage effect is uniform across the day |
| 012 | Semivariance ratio (`scripts/exp012_semivariance.py`) | **SUPPORTED** — see below |

**Exp 012 (the current best result):** the downside/upside realized
semivariance ratio adds significant, out-of-sample, stable incremental power for
next-day realized volatility, *beyond* both HAR-RV and the leverage effect. In
the full model the leverage dummy goes insignificant (t=−0.57) while
`semivar_ratio` stays significant (t=+2.88) — i.e. it appears to **subsume** the
coarse leverage effect. **But** the effect is small (~1.3% incremental OOS R²),
one fold was negative, and it came from a search on the *same* 2-year window, so
Evidence Score is a modest 60/100. The honest next step is **forward
validation** on newly collected data — which is exactly what the daily loop is
building.

Critically: a SUPPORTED result here means "add this term to the volatility
*forecasting* model and study further." **It is not a trading strategy.**

---

## 5. The live lens — `nifty_quant/research/live_lens.py`

Enabled with `--monitor`. Two jobs:

**(a) Situational awareness, every poll.** A one-line descriptive readout:

```
LENS  13:42 NIFTY 23,910 | VIX 14.1 (+1.6) | PCR 0.78 | straddle 88 (impl ±0.37%) | max-pain 24,000
```

Metrics: PCR(OI), ATM straddle premium & implied move, ATM IV (back-solved),
max pain, spot, India VIX and its intraday change. All observations are also
appended to `logs/observations_<date>.jsonl`.

**(b) Sample-gated pattern checks that grow with the data.** A
`HistoricalContext` loads every collected day, computes a per-day summary, and
positions today's live readings by percentile against history. It runs
pre-registered checks that **refuse to report a result until enough data
exists**:
- **VRP** (VIX-implied vs realized daily move) — needs 20 days.
- **0-DTE straddle implied-vs-realized** (expiry days only) — needs 8 expiry
  days.

Until the threshold is met a check prints `ACCUMULATING (n/need)`. This is the
mechanism by which the lens "grows into" pattern discovery **without
manufacturing false positives** on tiny samples. It never emits a trade signal.

---

## 6. Deployment

A full Linux/EC2 kit exists (`scripts/collect_daily.sh`, `sync_to_s3.sh`,
`healthcheck.sh`, `DEPLOY.md`, LF line-endings via `.gitattributes`) and is
portable to any Linux box (Oracle Always-Free / Raspberry Pi / VPS).

**Current choice:** running on the **local Windows PC** with keep-awake enabled
and the "never sleep" power setting, because the available AWS credits can't be
used for EC2 this month. Data (`data/`, `data_test/`, `reports/`, `logs/`) is
gitignored — only code and pre-registrations are committed.

---

## 7. What the platform deliberately will NOT do

- **No "loop until it finds a strategy."** Continuously mining two years of one
  index until something looks profitable is textbook data-dredging; it
  guarantees false positives. Every screen here is bounded, FDR-corrected, and
  one-shot, and survivors face a fresh pre-registered test.
- **No moving the goalposts.** Decision rules are locked before results are seen.
- **No claiming patterns on small samples.** The lens stays ACCUMULATING until
  its declared minimum.
- **No live orders.** Read-only by construction.

---

## 8. Honest expectations

- As solo research infrastructure: roughly **7/10** — disciplined, reproducible,
  and it prevents self-deception, which is most of the battle.
- Versus world-best quant shops: **1–2/10** — they have tick data across
  thousands of instruments, teams, co-location, and years of curated history.
- **Most likely realistic outcome:** a small, risk-managed **volatility** edge
  (something in the variance-risk-premium / semivariance family), *not* a
  directional holy grail. The value is in knowing *precisely* how strong (or
  weak) any edge is, with error bars, rather than trading on a hunch.

---

## 9. The daily routine, end to end

1. **Before 09:15** — start the collector with `--monitor`.
2. **During the session** — it polls every ~120s, writes snapshots + VIX, and
   prints the live lens line. Runs unattended; survives blips.
3. **After 15:30** — it exits automatically on weekdays after close.
4. **Post-session** — run `option_quality_report.py --date <today>`; confirm
   GOOD.
5. **Periodically** — as day count crosses thresholds, the lens checks flip from
   ACCUMULATING to a real (confidence-qualified) reading; run/extend offline
   experiments (e.g. forward-validate Exp 012) on the growing sample.
6. **Commit** code/prereg changes as work is done (data stays local/gitignored).

That's the whole machine: **record faithfully, analyze honestly, and let the
evidence — not hope — decide.**
