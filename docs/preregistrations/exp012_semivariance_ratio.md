# PRE-REGISTRATION — Experiment 012

**Downside/Upside Semivariance Ratio as an Incremental Next-Day RV Predictor**

> Status: **PRE-REGISTERED (locked before running).** This document fixes the
> hypothesis, variables, method, and decision rules in advance. Results are NOT
> yet known. Do not alter the decision thresholds after seeing results.

---

## Provenance (honest context)
This candidate did **not** arise from an independent hypothesis. It surfaced
from a **bounded 15-candidate exploratory screen**
(`scripts/volatility_feature_search.py`), where `semivar_ratio` was one of two
survivors (mean walk-forward incremental OOS R² ≈ 0.019, positive in 3/4 folds,
BH-adjusted p ≈ 0.010). Because it came from a search, the honest prior is an
**elevated false-discovery risk** even after in-screen FDR correction. This
dedicated, pre-registered experiment exists precisely to confirm or reject it on
fresh, fixed decision rules — the same discipline applied to Exp 001–011.

## Scientific Question
Does the ratio of downside to upside realized semivariance on day *t* add
statistically significant, out-of-sample, stable predictive power for day *t+1*
realized volatility, **beyond** (a) the HAR-RV baseline and (b) the already-
established leverage effect (Exp 006A)?

## One New Variable vs Experiment 003
`semivar_ratio` (lagged one day). Everything else — target, HAR terms, log-space,
HAC errors, walk-forward protocol — is identical to Exp 003/006A/007.

## Definitions
- 5-minute log returns within the regular session (09:15–15:30), no overnight.
- Downside semivariance: `RS_down = Σ r_i²` over bars with `r_i < 0`.
- Upside semivariance: `RS_up = Σ r_i²` over bars with `r_i > 0`.
- **`semivar_ratio` = RS_down / RS_up** (day *t*), then **shifted +1 day**.
- Target: `y = log RV_{t}` (next-day RV relative to information through *t−1*).
- HAR baseline predictors: `har_d, har_w, har_m` (lagged 1/5/22-day mean log RV).

## Hypotheses
- **H0:** After controlling for HAR-RV **and** the day's return-sign leverage
  term, `semivar_ratio` adds no incremental predictive power
  (β = 0; incremental OOS R² ≤ 0).
- **H1:** `semivar_ratio` adds significant, positive, stable incremental
  predictive power **beyond both** the HAR baseline and the leverage term.

## The critical control (the make-or-break test)
Semivariance ratio is mechanically related to the sign of returns, so it may
simply **re-express the leverage effect** already found in Exp 006A. Therefore
the decisive model is:

```
log RV_next ~ har_d + har_w + har_m + neg_return_dummy + semivar_ratio
```

`semivar_ratio` must add value **over the leverage-augmented baseline**, not just
over plain HAR. If it only beats plain HAR but not the leverage-augmented model,
the verdict is "redundant with the known leverage effect," NOT a new finding.

## Method (fixed)
1. HAC (Newey-West, maxlags=10) regression of the full model on the full sample;
   report `semivar_ratio` coefficient, t-stat, p-value.
2. Incremental OOS R² via **4-fold expanding walk-forward**: full model vs the
   **leverage-augmented baseline** (not plain HAR), per fold.
3. Per-year coefficient sign and significance (2024/2025/2026).
4. Block(day) bootstrap 95% CI for the coefficient (≥1000 resamples).
5. Report sample size at each stage.

## Decision Rules (LOCKED — do not change after seeing results)
**SUPPORTED** only if ALL hold:
- HAC t-stat significant (|t| ≥ 2.5) in the full leverage-augmented model, AND
- bootstrap 95% CI for the coefficient excludes zero, AND
- incremental OOS R² (over the leverage-augmented baseline) > 0 in **≥ 3 of 4**
  walk-forward folds, AND
- coefficient sign is **stable across all three years**, AND
- mean incremental OOS R² > **0.01** (economic-relevance floor).

**REDUNDANT** (interesting but not new): beats plain HAR but NOT the
leverage-augmented baseline → it is a restatement of the Exp 006A leverage
effect. Log as such; do not promote.

**REJECTED:** fails significance, sign flips across years/folds, incremental OOS
R² ≤ 0 over the leverage-augmented baseline, or CI includes zero.

**INSUFFICIENT DATA:** any fold too small for a stable estimate (report, don't
force a verdict).

## Minimum Sample
~470 daily observations (2 years). Adequate for HAR-style regression with HAC;
folds are thin, so per-fold results carry wide error bars and the year-stability
+ bootstrap requirements are the real gate.

## Expected Failure Modes
- **Redundancy with the leverage effect** (highest likelihood) — the control
  model is designed to catch exactly this.
- **Screening false positive** — it came from a 15-way search; fresh rules and
  the leverage control are the defense.
- **Small / unstable** — ~0.02 incremental R² is modest; may not clear the folds
  or year-stability bars.

## Scoring (filled AFTER running — commit 893afc6 design, executed on 472 daily obs)
- Status: **SUPPORTED** (passed all five locked rules).
- Key numbers: semivar_ratio HAC t=+2.88 (p=0.004); bootstrap 95% CI [+0.022,+0.133]
  (excludes 0); incremental OOS R² over the LEVERAGE-augmented baseline = +0.0133
  mean, positive in 3/4 folds; sign positive in all three years (2024/25/26).
- **Central discovery:** in the full model the leverage dummy `neg_lag` becomes
  INSIGNIFICANT (t=−0.57) while `semivar_ratio` stays significant (t=+2.88). The
  continuous downside/upside variance ratio appears to **subsume** the coarse
  Exp 006A leverage effect, not merely restate it. It is *how downside-skewed the
  intraday variance was*, more than *whether the day was down*, that adds
  forecasting power for next-day RV.
- Evidence Score: 60/100 (passed a pre-registered, leverage-controlled,
  walk-forward screen — but small effect ~1.3% incr R², one negative fold, uneven
  per-year strength, and it originated from a 15-way search on the SAME sample).
- Confidence Score: 65/100.
- **Honest limitation:** screen-then-confirm on the same 2-year window is not
  fresh out-of-sample. The decisive validation is FORWARD — does it hold on new
  months / the option-chain-era data being collected now?
- Future work: (1) forward/out-of-sample validation on newly collected data;
  (2) formalise as an HAR-X volatility model with the semivariance term;
  (3) re-examine the Exp 006A leverage result as a coarse proxy for this.
- Scope reminder: this is a VOLATILITY-FORECASTING improvement, NOT a strategy.

## Explicitly out of scope
No trading rule, no entry/exit, no sizing. This is a structural forecasting test.
A SUPPORTED result means "add `semivar_ratio` to the volatility model and study
further" — it is **not** a strategy.

---

## ⚠️ CONFIDENCE DOWNGRADE (2026-07-01, from Exp 023 — data-snooping control)
Exp 023 applied White's Reality Check and Hansen's SPA to the original 15-feature
screen that generated this candidate. On a 236-day pooled expanding walk-forward,
`semivar_ratio` was the **best** of the 15 features (single-model t=+2.00) but
did **NOT survive** the data-snooping control: **White RC p=0.110, Hansen SPA
p=0.223** (both ≥ 0.05). Once the multiplicity of the 15-way search is accounted
for, its edge is consistent with luck.

This does not erase Exp 012's dedicated confirmation (HAC t=+2.88, bootstrap CI,
walk-forward), but that confirmation used the **same 2-year window** as the
screen, so it was never truly out-of-sample — a limitation this document already
flagged. Combined with the SPA failure, the honest status is now:

> **Exp 012 status: WEAK, UNCONFIRMED CANDIDATE.** Evidence Score revised
> **60 → 35/100**; Confidence **65 → 40/100**. Do not treat `semivar_ratio` as an
> established predictor. The decisive test is **forward** out-of-sample data
> (new months), not any further analysis of the 2024–26 sample.

The original scoring above is preserved unedited for the record; this note
supersedes its confidence claims.

---
*Pre-registered 2026-06-30. Provenance: bounded volatility-feature search
(commit ecba014). To be executed as a separate, subsequent step.*
