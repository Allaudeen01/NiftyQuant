# PRE-REGISTRATION — Experiment 017

**Regime Switching via a Gaussian Hidden Markov Model**

> Status: **PRE-REGISTERED (locked before running).** Method and decision rules
> fixed in advance.

---

## Scientific Questions
1. Does an unsupervised **Gaussian HMM** on daily NIFTY features discover
   distinct, interpretable regimes (e.g. calm vs turbulent, bull vs bear)?
2. Are the regimes **persistent** (real states, not noise re-labelling)?
3. Does the number of states preferred by **BIC** exceed 1 (i.e. is a
   regime-switching model justified over a single Gaussian)?
4. **The honest alpha test:** using only *causal* (expanding-window, filtered)
   regime information, does the regime add forward-predictive power for
   (a) next-day realized volatility beyond HAR, or (b) next-day return sign?

## Object of study
Daily NIFTY data 2024–2026 (~490 days) from the 5m warehouse: daily
close-to-close log return `ret` and log realized volatility `log_rv`
(intraday, overnight excluded). Both standardized for the HMM.

## Why this is an upgrade, not a rebuild
Exp 004 already labels regimes by HAR-forecast percentile (LOW/NORMAL/HIGH) — a
*supervised, hand-cut* volatility bucket. The HMM is *unsupervised and joint*
(return + vol), infers the number of states and the transition dynamics from the
data, and provides regime **persistence/duration** — things the percentile cut
cannot.

## Method (fixed)
- **Characterisation (Q1–Q3):** full-sample `GaussianHMM(full cov, n_iter≥200)`
  for K=2 and K=3 (fixed seed, several inits, keep best log-likelihood). Report
  per-state mean return & mean RV, the transition matrix, self-persistence, and
  expected duration `1/(1−p_ii)`. Choose K by **BIC** vs a K=1 single Gaussian.
- **Causal forward test (Q4) — the make-or-break, look-ahead-safe part:** a
  full-sample Viterbi/smoothed path is contaminated by future data, so it is
  **NOT** used for prediction. Instead: refit the HMM on an expanding window
  every 21 trading days; each day `t`, compute the **filtered** posterior
  `P(high-vol state | data through t)` (high-vol state identified within each
  refit by the larger mean `log_rv`, so it is label-invariant). Then:
  - (a) RV: `log RV_{t+1} ~ har_d + har_w + har_m + p_highvol_t` — HAC t-stat and
    4-fold walk-forward incremental OOS R² vs HAR.
  - (b) direction: mean next-day return and sign hit-rate by causal high/low-vol
    state; two-sample test.

## Decision Rules (LOCKED)
- **REGIMES REAL** (Q1–Q3) if: BIC prefers K≥2 over K=1, **and** at least one
  state pair differs materially in mean RV (ratio ≥ 1.5×), **and** mean expected
  duration of states ≥ 3 days (persistence).
- **Q4a — REGIME ADDS RV FORECAST VALUE** only if causal `p_highvol` HAC
  |t| ≥ 2.5 **and** mean incremental OOS R² > 0 in ≥3/4 folds. Otherwise
  **NO INCREMENTAL VALUE over HAR** (expected — HAR already encodes vol
  persistence).
- **Q4b — REGIME PREDICTS DIRECTION** only if next-day mean return differs across
  states at p<0.01 with a consistent sign. Otherwise **NO DIRECTIONAL EDGE**
  (expected, consistent with Exp 001–009).

## Expected outcome (honest prior)
The HMM will almost certainly find a persistent **calm/low-vol** state and a
**turbulent/high-vol** state — but that is just a re-description of the
volatility clustering already established in Exp 003. The likely verdicts:
REGIMES REAL = yes (descriptively), Q4a = NO incremental value over HAR, Q4b = NO
directional edge. That would make the HMM a useful **risk-context / labeling**
tool, not a source of alpha. Anything stronger would be surprising and would face
the data-snooping scrutiny of Exp 023.

## Caveats
HMMs overfit readily and find "regimes" in pure noise; BIC and the causal
(filtered, expanding-refit) protocol are the guards. Label-switching across
refits is handled by identifying states by their RV level, not by index.

## Scope
Characterisation + one causal forecasting test. **No trading rule, no strategy.**

---

## Scoring (filled AFTER running — 495 daily obs, features: daily return + log-RV)
- **Q1–Q3 — REGIMES REAL: yes (descriptively).** BIC prefers **K=2** (2631)
  over K=1 (2831) and K=3 (2640). The two states:
  - **CALM:** 48% of days, ann-vol **6.5%**, mean return **+0.018%/day**.
  - **TURBULENT:** 52% of days, ann-vol **11.0%**, mean return **−0.009%/day**.
  Both highly persistent (self-transition 0.91, **expected duration ~11.5 days**),
  RV ratio 1.69×. So the regimes are genuine, persistent, and interpretable —
  and show the classic "drift up when calm, drift down when turbulent" texture.
- **Q4a — RV forecast: NO INCREMENTAL VALUE over HAR.** Causal (filtered,
  expanding-refit) `p_highvol`: HAC t=+1.45 (ns); walk-forward mean incremental
  OOS R² = **−0.054** (1/4 folds positive). HAR already encodes the vol
  persistence the regime represents.
- **Q4b — Direction: NO DIRECTIONAL EDGE.** Next-day mean return high-vol
  +0.053% vs low-vol −0.085%, Welch p=0.091 (not significant at the locked 0.01
  bar). Consistent with Exp 001–009.

## Interpretation (honest)
The HMM cleanly recovers exactly what Exp 003 already established: NIFTY has a
persistent low-vol state and a persistent high-vol state (volatility
clustering). It is a **real, interpretable, unsupervised re-description** of that
clustering — genuinely useful as a **risk-context / regime-labeling** tool (e.g.
sizing down in the turbulent state, expected ~11-day dwell time) — but it adds
**no forecasting alpha** over HAR and gives **no directional edge**. This is a
"confirms known structure, no new edge" result, in line with the whole campaign.
Its practical value is contextual (risk overlay), not predictive.

---
*Pre-registered 2026-07-01, executed same day as `scripts/exp017_regime_hmm.py`.*
