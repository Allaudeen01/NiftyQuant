# PRE-REGISTRATION — Experiment 023

**Data-Snooping Control: White's Reality Check & Hansen's SPA**

> Status: **PRE-REGISTERED (locked before running).** Method and decision rules
> fixed in advance.

---

## Why this experiment exists
This program has run many tests, and Exp 012 (`semivar_ratio`) explicitly
originated from a **15-candidate feature search**. FDR/Benjamini-Hochberg was
applied there, but BH controls the false-discovery rate over *independent-ish*
p-values; it does **not** fully account for the dependence between overlapping
model forecasts. The proper family-wise, dependence-aware control is a
**bootstrap reality check**. This experiment adds that control and applies it
where it matters most.

## Scientific Question
After honestly accounting for the number of models/features searched, does the
**best** performer's outperformance survive — or is it consistent with luck
(data snooping)?

## Methods (standard: White 2000 "Reality Check"; Hansen 2005 "SPA")
- **Performance measure** `f_{k,t}`: benchmark loss − model-k loss on period `t`
  (higher = model better than benchmark).
- **Stationary bootstrap** (Politis-Romano 1994) with mean block length `L` to
  preserve serial dependence; `B` resamples.
- **White's Reality Check** statistic `V = max_k √T · f̄_k`; p-value from the
  bootstrap null (`g_k = f̄_k` recentering for all k).
- **Hansen's SPA** (consistent variant `SPA_c`): studentized `√T·f̄_k/ω_k`, with
  poor models recentered out via the `−√(2 log log T)` threshold; `ω_k` from the
  bootstrap. SPA is more powerful (not diluted by hopeless models).
- Implemented as reusable primitives in
  `nifty_quant/analytics/data_snooping.py`, validated on synthetic null/
  alternative cases in tests. `L=10`, `B=5000` unless noted.

## Applications (fixed in advance)
1. **RV forecast horse-race (Exp 013 set).** Benchmark = random walk (RW).
   Models = {EWMA, HAR, GARCH, EGARCH}. `f_{k,t}` = QLIKE_RW − QLIKE_k on the
   247 common OOS days. Question: does the best RV forecaster beat the naive
   benchmark after accounting for trying four models?
2. **The 15-feature screen (Exp 012 provenance stress-test).** Benchmark =
   HAR-only next-day log-RV forecast. Models = the 15 HAR+candidate models from
   `volatility_feature_search.py`. `f_{k,t}` = (baseline squared error) −
   (augmented squared error) on a pooled expanding one-step walk-forward OOS
   set. Question: **does the best feature (expected: `semivar_ratio`) survive a
   data-snooping-corrected test**, or was it a lucky draw from 15?

## Hypotheses
- **H0 (each application):** the best model does not outperform the benchmark
  once the search over K models is accounted for (max_k E[f_k] ≤ 0).
- **H1:** at least one model genuinely outperforms (reality-check / SPA p < 0.05).

## Decision Rules (LOCKED)
- **PASSES SNOOPING CONTROL** if SPA_c p-value < 0.05 (primary) **and** White RC
  p-value < 0.10 (secondary, RC is more conservative). → the best performer's
  edge is not attributable to the multiplicity of the search.
- **FAILS / SNOOPING-EXPLAINED** if SPA_c p ≥ 0.10. → the apparent edge is
  consistent with luck given the search; downgrade confidence accordingly.
- **BORDERLINE** if 0.05 ≤ SPA_c p < 0.10. → report as weak/indicative.

### Pre-committed consequence for Exp 012
- If Application 2 shows `semivar_ratio` **passes**, Exp 012's Evidence Score is
  reaffirmed (the search multiplicity does not explain it).
- If it **fails**, I will explicitly downgrade Exp 012's confidence in its
  scoring note (honest, no goalpost-moving). Either way the result is recorded.

## Expected outcomes (honest prior)
- App 1: HAR very likely beats RW under SPA (Exp 013 showed HAR clearly best; RW
  was only "not significantly different" the other way). Plausible PASS.
- App 2: genuinely uncertain. `semivar_ratio` had a small effect (~1.3% incr R²)
  and came from a 15-way search; it may well be **BORDERLINE or FAIL** under a
  strict snooping control. That would be an important, humbling result and is
  exactly why this test is run.

## Scope
Inference/validation methodology. **No trading rule, no strategy.**

---

## Scoring (filled AFTER running — stationary bootstrap L=10, B=5000)

### Application 1 — RV forecast horse-race (247 OOS days), benchmark = RW
- Per-model mean QLIKE advantage over RW: HAR +0.0168 (t=+1.06), EWMA −0.011,
  EGARCH −0.230 (t=−5.56), GARCH −0.258 (t=−4.92).
- Best = HAR. **White RC p = 0.556; Hansen SPA p = 0.183.**
- **Verdict: FAILS / snooping-explained.** HAR is clearly the best of the four
  models (and decisively beats EWMA/GARCH/EGARCH per Exp 013), but its edge over
  the *trivial random-walk* RV forecast is **not** statistically distinguishable
  from luck once the 4-model search is accounted for. Honest reading: HAR is the
  right workhorse *among models*, but "HAR beats tomorrow=today" is not a
  significant claim on this sample.

### Application 2 — the 15-feature HAR screen (236 pooled OOS days), benchmark = HAR-only
- Best feature = **`semivar_ratio`** (mean SE-reduction +2.6e-3, single-model
  t=+2.00) — it IS the top of the 15, consistent with the original screen.
- **White RC p = 0.110; Hansen SPA p = 0.223.**
- **Verdict: FAILS / snooping-explained.** `semivar_ratio`'s naive single-model
  t=+2.00 looks "significant," but once you account for having searched **15**
  correlated features, the max-statistic bar is higher and it does **not** clear
  it. Its edge over HAR is consistent with a lucky draw from the search.

## Consequence for Exp 012 (pre-committed — executed)
Per the locked rule, Exp 012's confidence is **downgraded**. `semivar_ratio`
passed its dedicated 5-rule confirmation, but (a) that confirmation used the
*same* 2-year window as the screen (never truly out-of-sample), and (b) a proper
data-snooping control on the pooled OOS does **not** clear it. A downgrade note
has been appended to `exp012_semivariance_ratio.md`. Net status: **weak,
unconfirmed candidate — do not treat as established; the decisive test is FORWARD
out-of-sample data.**

## Broader lesson recorded
Single-variable t-stats (even ~2–3) systematically overstate significance when
the variable was the survivor of a multi-candidate search. Every future
"survivor of a screen" must clear a bootstrap SPA control, not just BH-FDR,
before earning any Evidence Score above ~40. `nifty_quant/analytics/data_snooping.py`
(reality_check_spa) is now the standard tool for this.

---
*Pre-registered 2026-07-01, executed same day as `scripts/exp023_reality_check.py`
with `nifty_quant/analytics/data_snooping.py`.*
