# NiftyQuant — Research Roadmap (Experiments 013+)

*Triaged from a proposed experiment list, 2026-07-01. Honest priorities,
dependencies, and explicit "not now / not this way" calls.*

Numbering continues from the completed set. **Exp 012 (semivariance ratio) is
SUPPORTED and already committed** — the proposed list is renumbered from 013.

Every experiment follows the house rules: pre-register decision rules before
running, use expanding walk-forward, correct for multiple testing, and accept
"rejected / insufficient" as valid outcomes.

---

## Priority order

**Now (feasible on existing daily + 5m candle warehouse):**
1. **013 RV forecast comparison** — ✅ DONE. HAR CONFIRMED BEST (lowest QLIKE;
   EWMA/GARCH/EGARCH significantly worse, RW tied). *LSTM excluded.*
2. **014 Jump detection** — ✅ DONE. ~8.4% mean relative jump; jumps NOT
   clustered (serially independent); jump split adds NO forecasting value over
   HAR. NIFTY vol is predominantly continuous.
3. **016 Tail risk** — ✅ DONE. Clearly fat-tailed (excess kurtosis +2.08, JB
   p≈3.6e-20, Hill α≈3, ~5× too many 3σ days); tails near-symmetric on robust
   evidence; Gaussian ES understates 99.5% tail loss by ~1.2×. Added
   `hill_estimator` + `pot_gpd_fit` to stat_tests (+3 tests). Feeds sizing.
4. **023 Reality Check / SPA** — ✅ DONE. Best RV model (HAR) does NOT beat the
   naive RW under SPA (p=0.18); best screen feature (`semivar_ratio`) does NOT
   survive snooping (SPA p=0.22). **Exp 012 downgraded to weak/unconfirmed
   (Evidence 60→35).** Added `data_snooping.py` (reality_check_spa) + 5 tests.

**Next (feasible, medium value or partially done):**
5. **015 Vol-clustering half-life** — mostly exists (`half_life` in
   `stat_tests.py`, `rv_persistence.py`); formalize on RV shocks with a decay
   curve.
6. **017 Regime switching (HMM)** — ✅ DONE. BIC picks 2 states (calm ann-vol
   6.5% / turbulent 11.0%), highly persistent (~11.5-day dwell). But adds NO RV
   forecast value over HAR and NO directional edge — a real, interpretable
   re-description of vol clustering, useful as a risk-context overlay only.
7. **020 Feature importance** — Random Forest over lagged features, reported as
   *descriptive structure only*. HIGH overfit risk on ~470 daily obs; framed as
   "does any signal plausibly exist," never as strategy discovery.
8. **019 Intraday seasonality (beyond vol)** — per-5m-bucket momentum/reversal.
   Note: NIFTY cash has zero volume (Exp 010) and no spread, so the volume/
   spread parts need futures or option data first.

**Blocked on data:**
9. **018 Volatility spillover** — ✅ DONE (data sourced via `yfinance`). Open-gap
   coupling to overnight global confirmed (~21% of gap variance, S&P t=+2.69) but
   NOT tradeable; NO post-open edge; NO RV-forecast value over HAR. "Coupling
   exists, edge does not." External data cached to `data/external/`.
10. **017(proxy) Order-flow proxy** — candle body/wick/close-location. Low
    expected value; parked.

**The prize, forward-only:**
11. **027 Volatility Risk Premium (IV vs RV)** — the reason option chains are
    being collected. Needs ~20+ collected option-days (currently ~6); the live
    lens already gates this. This is the endgame, not a today task.

**Deferred until a candidate STRATEGY exists (premature now):**
- **021 Transaction-cost sensitivity** — alpha-vs-cost breakeven curve. Only
  meaningful for a real trading rule.
- **024 Monte Carlo robustness** — bootstrap/trade-shuffle survival probability.
- **025 Risk metrics** — Sharpe/Sortino/Calmar/MDD/Ulcer/Omega/Tail. Sharpe and
  drawdown already exist in `backtest/metrics.py`; extend when there's a curve.
- **026 Strategy capacity** — slippage/market-impact at size. Near-irrelevant at
  solo-book size until crores; lowest priority.

---

## Explicit "not now / not this way" calls (brutal honesty)

- **LSTM / deep nets for daily RV: rejected for this sample.** ~470 daily
  observations cannot support a model with thousands of parameters — it memorizes
  noise, flatters in-sample, and fails out-of-sample. Published horse-races
  generally show HAR beating neural nets on daily RV. Revisit only with far more
  data and heavy regularization.
- **Walk-forward (proposed 022) is already standard**, not a missing experiment.
  It is built into 003/006A/007/012 and will be in every new one.
- **Feature importance (020) is not strategy discovery.** It is a descriptive
  check for whether structure plausibly exists, reported with the overfit caveat.
- **VRP (027) will not be claimed on a small sample.** Gated at 20+ option-days.

---

## Dependencies to install when the relevant experiment starts
- `arch` — GARCH / EGARCH (Exp 013). ✅ installed (8.0.0).
- `hmmlearn` — Gaussian HMM (Exp 017). ✅ installed (0.3.3).
- `statsmodels`, `scikit-learn` — already installed.

---

## Shared infrastructure to reuse (don't rebuild)
- Daily RV from 5m candles: `scripts/rv_persistence.py`, `scripts/regime_strategy.py`.
- HAR features + correct 1-day lagging: `scripts/volatility_feature_search.py`.
- Stat primitives (Hurst/ADF/half-life/VR/Kelly): `nifty_quant/analytics/stat_tests.py`.
- Risk metrics (Sharpe/drawdown): `nifty_quant/backtest/metrics.py`,
  `nifty_quant/validation/performance.py`.
- Option analytics (PCR/max-pain/ATM-IV): `nifty_quant/analytics/options.py`.
