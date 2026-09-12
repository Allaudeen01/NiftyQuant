# PRE-REGISTRATION — Experiment 034

**Option Chain Information Content Study — Does option-chain state carry incremental out-of-sample predictive information about near-term NIFTY movement? (RESEARCH ONLY)**

> Status: **DRAFT REV-2 — NOT YET LOCKED.** Nothing in this document has been
> run. It becomes binding only when committed. After that commit, no
> parameter, target, feature, threshold or criterion below may be changed; any
> new idea belongs to EXP035+.

---

## Relationship to EXP033 (hard boundary)

EXP033 is **permanently frozen** (tag `EXP033_FINAL_FROZEN`, freeze commit
`123f345`, experiment commit `5a1d3ba`). EXP034 does not modify, rerun,
regenerate or overwrite any EXP033 file, and writes exclusively to
`data/derived/exp034_*`, `reports/exp034/` and `docs/preregistrations/exp034_*`.
EXP034 builds its own feature panel from raw `data/option_chain`; it does not
consume EXP033's derived artifacts.

### What EXP033 established — and what it left open
EXP033 tested 6 pre-declared *discrete* option-chain events × 3 triple-barrier
configs, scored through a calibrated LightGBM model behind a trading gate, net
of costs: 9 REJECT, 9 INSUFFICIENT_DATA, 0 candidates. It established that
(a) those discretized events traded that way lose money after costs, and
(b) at this sample size all three LightGBM models had **worse log loss than a
constant base-rate predictor**.

It did **not** establish that option-chain data carries no information. Four
confounders sat between "information" and "verdict": hard thresholding of
continuous signals into binary events, a trade gate, transaction costs, and an
OOS window of only ~19 days. EXP034 removes all four and asks the prior
question directly.

### Defect inherited from EXP033, corrected here (disclosed, not repaired there)
The stored `volume` field is **cumulative daily traded volume** — verified
empirically: strictly monotonically non-decreasing across all 172 snapshots of
2026-09-11 for the ATM call, with zero negative first differences (whereas
`open_interest` shows 73 negative differences, correctly behaving as a level).

EXP033's E1 "volume spike" family applied a trailing z-score to the **raw
cumulative level**, while its own docstring described a "2-min volume
increment". A z-score of a monotonically increasing series against a trailing
window is dominated by intraday drift; it largely measured *"cumulative volume
has grown since the open"*, not *"volume is unusual right now"*.

EXP033 is frozen and is **not** corrected. EXP034 therefore defines every
volume feature on the **first difference** of cumulative volume (see F01–F03).
This is stated here so the distinction between the two experiments is on the
record rather than discovered later.

---

## What this is (and is NOT)

A measurement of **information content**, not a strategy search. There are no
entry/exit rules, no take-profit/stop thresholds, no win rates, and no
transaction costs anywhere in EXP034. Economic significance is explicitly out
of scope and deferred to EXP035+.

### Experiment naming (LOCKED)
- **EXP034-A** — the 58-usable-day exploratory run defined by this document.
- **EXP034-C** — the forward-only confirmatory replication, same locked
  pipeline, triggered per the rule below.

"EXP034" without a suffix refers to the preregistration and codebase shared by
both. The maximum verdict attainable by **EXP034-A alone** is
**EXPLORATORY-POSITIVE**, which authorizes nothing except running EXP034-C.

### Scientific question
> Conditional on price history, intraday time, and India VIX, does near-expiry
> NIFTY option-chain state carry **incremental out-of-sample predictive
> information** about the *next* 2- and 10-minute spot movement — in sign, and
> in magnitude?

---

## Sample-reuse decision (LOCKED)

This is the **third** experiment run against the same 59 collected days
(EXP032, EXP033, now EXP034-A). That is a program-level data-snooping exposure
that no within-experiment correction can undo. Accordingly:

- EXP034-A is designated **EXPLORATORY BY CONSTRUCTION**. No result from it
  may ever be described as confirmed, validated, or tradable, regardless of
  how strong it looks.
- **Permitted conclusion vocabulary for EXP034-A is closed to exactly three
  labels:** `EXPLORATORY EVIDENCE OF INCREMENTAL INFORMATION`,
  `NO EXPLORATORY EVIDENCE OF INCREMENTAL INFORMATION`, or
  `INCONCLUSIVE — UNDERPOWERED`. The words *proves*, *confirms*, *validates*,
  *establishes*, and *tradable* may not be applied to an EXP034-A result in
  any report, commit message, or summary.
- A **confirmatory replication is pre-committed here**: this identical
  pipeline, unchanged, run on **forward days only** (collected after
  2026-09-11) once total usable days reach **≥ 100**, published as its own
  experiment id (**EXP034-C**), per the EXP032 freeze policy that re-running
  locked rules on new data is a separate out-of-sample experiment.
- Rejected alternatives, recorded so the choice is auditable: a 12-day lockbox
  (too small to confirm anything — it would spend irreplaceable data on a test
  with no power), and simply proceeding without acknowledgement (contradicts
  the discipline that downgraded Exp012 after SPA).

### EXP034-C trigger and eligibility (LOCKED)
- **Trigger:** total usable days ≥ 100, where "usable" means a day passing the
  *same* data-quality screen defined below — no other definition.
- EXP034-C is run **once**, on the forward block only. It is **not** re-run
  as days accumulate, and may not be run early on a partial forward sample.
- No feature may be added, removed or substituted; no model may be swapped;
  no threshold may be retuned. EXP034-C executes the byte-identical pipeline
  recorded in the EXP034 freeze manifest.

### EXP034-C data-compatibility rule (LOCKED)
The confirmatory replication may use only observations produced by a
data-generating process compatible with the schema and timing assumptions
preregistered here. The current process is designated **`dgp-v1`**, defined by:

| Property | `dgp-v1` semantics |
|---|---|
| Snapshot schema | `_SNAP_COLUMNS` as stored by `collect_market_data.py::write_snapshot` |
| Timestamp semantics | one `ts` stamped **before** any network call; all rows of a poll share it |
| Quote-to-spot ordering | quotes fetched **first**, `get_spot()` called **last**, 30s spot cache |
| Near-expiry selection | minimum `expiry` present in the snapshot |
| Spot source semantics | live index LTP, with daily-close fallback on LTP failure |
| India VIX timing | fetched **before** the chains, stamped with the same `ts` |
| Option field definitions | `volume` cumulative daily; `open_interest` a level; `oi_change` unpopulated; `implied_volatility` null |
| Storage format | one parquet file per poll under `YYYY/MM/DD/HH_MM.parquet` |

Each forward-collected day must pass the same per-day data-quality screen
**and** be certified `dgp-v1` before inclusion. A collector or provider code
change does not automatically disqualify forward data; but if it alters
**information timing, field meaning, or feature observability**, the affected
days are a new `dgp-v2` and **cannot be pooled into EXP034-C** — they would
require a separately numbered experiment. All post-lock collector or provider
changes must be versioned and logged in `docs/data_versions.md` with the
effective date. The compatibility decision for every forward day is made
**before** any EXP034-C model output is inspected.

> **Trade-off stated now, before locking.** Two `dgp-v1` properties documented
> below are arguably defects: the pre-stamped timestamp, and the far-expiry
> quote-newer-than-spot ordering. Repairing either is good for the platform
> but would start `dgp-v2` and reset the EXP034-C clock, since pre- and
> post-fix days could not be pooled. Choosing to fix the collector is
> therefore a decision to delay EXP034-C, and should be made deliberately.

---

## Data

- **Source:** `data/option_chain/YYYY/MM/DD/HH_MM.parquet`, **59 collected
  days**, 2026-06-22 → 2026-09-11, ~168 snapshots/day, median cadence ~2m11s.
- **Usable: 58 days.** 2026-06-26 is excluded — its `spot` field is constant
  (std = 0) for all 127 snapshots of the session.
  **Root cause:** `AngelOneProvider.get_spot()` falls back to
  `series.candles[-1].close` (the last *daily* close) when `_live_index_ltp()`
  returns `None`; a session-long LTP failure pins spot to a constant.
- **Per-day data-quality screen (LOCKED, applied identically in A and C):** a
  day is *usable* only if it has ≥ 100 distinct snapshots, per-day spot
  standard deviation > 0.5 points, ≥ 20 distinct strikes, both CE and PE
  present at the ATM strike in ≥ 95% of snapshots, and zero crossed quotes
  (ask < bid) at the ATM strike.
- **India VIX:** `data/vix/<year>/INDIAVIX_<date>.parquet`, joined by
  **backward as-of merge only**.
- **Near expiry only** (see timing finding below).

### Verified intra-poll timing (governs the whole design)
`collect_market_data.py::poll_once` stamps one `ts` **before** any network
call, then fetches VIX → near expiry → next expiry. Inside
`angelone.py::get_option_chain`, the option quotes are fetched first and
`spot = self.get_spot(...)` is called **last**, behind a 30-second cache.
Therefore, within one recorded snapshot:

| Series | Timing vs recorded `spot` | Leakage direction |
|---|---|---|
| Near-expiry quotes | a few seconds **older** | safe (biases *against* finding chain value) |
| Next-expiry quotes | a few seconds **newer** (reuses cached spot) | **unsafe** — encodes already-realized spot movement |
| India VIX | fetched first, so **older** | safe |

**Locked consequences:** (1) EXP034 uses **near-expiry data only**; (2) the
prediction window is offset by one full snapshot, so all intra-poll timing
slop is in the past before the label window opens.

---

## Definition of t+1 (LOCKED)

For every snapshot **t**, **t+1 means the next chronologically available valid
near-expiry option-chain snapshot after t.** It does **not** mean
`timestamp(t) + 2 minutes`. If one or more expected polling intervals are
missing, the next available valid snapshot is used.

All prediction targets and label windows begin from information observed at
that next valid snapshot, ensuring the feature snapshot at t is strictly in
the past relative to the prediction window.

The actual elapsed time between t and t+1 is retained and reported as a
data-quality diagnostic. Every emitted observation **must** record:

- `feature_timestamp` — the snapshot supplying all features (t)
- `prediction_start_timestamp` — the snapshot opening the label window (t+1)
- `prediction_end_timestamp` — the snapshot closing the label window (t+1+h)
- `elapsed_seconds_t_to_t1` — realized seconds between t and t+1

Observations where `elapsed_seconds_t_to_t1` exceeds **300 s** are dropped
(causal, outcome-independent). The distribution of `elapsed_seconds_t_to_t1`
is reported per fold.

---

## Prediction targets (LOCKED)

Decision time is snapshot **t**: every feature uses information at or before t.
The label window opens at t+1 as defined above. All labels are strictly
within-day; index arithmetic below is over *valid snapshots*, not clock time.

| ID | Target | Definition | Type | MODEL_0 naive baseline |
|---|---|---|---|---|
| **T1a** | signed return, h=1 | `spot[t+2]/spot[t+1] − 1` | regression | 0 (no-change / random walk) |
| **T1b** | signed return, h=5 | `spot[t+6]/spot[t+1] − 1` | regression | 0 |
| **T2a** | absolute return, h=1 | `abs(T1a)` | regression | causal rolling mean of abs-return per 30-min ToD bucket |
| **T2b** | absolute return, h=5 | `abs(T1b)` | regression | same |

**Secondary (not primary hypotheses):** T3a/T3b — binary "above-normal move",
`1{abs(r) > causal ToD-bucket median}`, baseline = base rate. Reported for
calibration evidence only; a T3 result can never establish a primary claim.
T3 is a coarsening of T2 and is deliberately kept out of the primary family.

**Exclusions (causal, outcome-independent):** observations whose label window
would cross 15:29 are **dropped, not truncated**; no overnight windows;
observations with `elapsed_seconds_t_to_t1 > 300` dropped; observations whose
label window contains any gap > 300 s dropped.

---

## Model ladder and exact model matrices (LOCKED)

| Model | Information set | Matrix columns |
|---|---|---|
| **MODEL_0** | Naive, per-target (table above) | n/a |
| **MODEL_1** | Price + time only | **30** |
| **MODEL_2** | MODEL_1 + India VIX | **35** |
| **MODEL_3** | MODEL_2 + option-chain features | **51** |

**MODEL_1 columns (30, enumerated):** `ret_1`, `ret_2`, `ret_5`, `ret_10`,
`rv_intraday_expanding`, `rv_20`, `twap_dist_pct`, `dist_from_high_pct`,
`dist_from_low_pct`, `or_state` (ordinal −1/0/+1), `or_forming_flag`,
`tod_bucket_dummy_02` … `tod_bucket_dummy_13` (12 dummies; 13 × 30-min buckets
spanning 09:15–15:29, bucket 01 is the reference), `dow_tue`, `dow_wed`,
`dow_thu`, `dow_fri` (Monday reference), `dte_cal`, `is_expiry_day`,
`atm_rolled`.

**MODEL_2 adds (5):** `vix`, `vix_chg`, `vix_mom`, `vix_accel`,
`vix_pctile_causal`.

**MODEL_3 adds (16):** the 15 option-chain features F01–F15 below, plus the
auxiliary indicator `iv_available`.

**Estimator (LOCKED): ridge regression** (logistic ridge for the secondary
T3). Rationale: the models are exactly nested, which makes the Clark-West test
valid; ridge is stable under the strong collinearity these features exhibit;
and EXP033's direct evidence is that a flexible learner underperformed a
constant base rate on this sample.

**LightGBM is a robustness check only.** Pre-declared and binding: a GBM
result may be reported alongside, but **can never convert a primary ridge null
into a positive finding**, and a null ridge result may not trigger an
exploratory switch to a nonlinear model within this experiment. Testing
nonlinear models is a separately declared experiment.

### Preprocessing and hyperparameter discipline (LOCKED)

All preprocessing parameters are fit **exclusively on the training portion of
each walk-forward fold**. This includes, where applicable: feature scaling,
centering, variance estimation, imputation statistics, clipping thresholds,
and any transformation requiring fitted parameters.

No preprocessing statistic may be computed using validation, test, future, or
full-sample data.

For each fold, in this order:
1. Fit preprocessing on the training observations only.
2. Freeze all fitted preprocessing parameters.
3. Apply the frozen transformation to the corresponding OOS observations.
4. Fit the estimator using only transformed training data.
5. Generate predictions on transformed OOS data **without refitting**.

The implementation must not use global normalization, global z-scores,
full-sample means, full-sample standard deviations, or future-derived
imputation statistics.

**Ridge alpha selection (LOCKED):** alpha is selected strictly inside the
training window via a **chronological inner split** — inner-train = the first
80% of training days, inner-validation = the last 20% — over the fixed grid
`alpha ∈ {0.1, 1, 10, 100, 1000}`. The model is then refit on the full
training window at the selected alpha. **The OOS test block is never used to
select alpha, and never influences any fitted parameter.**

---

## Option-chain feature registry (LOCKED — exactly 15 features, 5 groups)

Near expiry only. Relative strike level `k ∈ {−2,−1,0,+1,+2}` denotes the
strike at ATM+k; absolute strikes are never used. `ε` is a small constant
guarding division. `dv_X(k,t) = max(0, vol_cum_X(k,t) − vol_cum_X(k,t−1))` is
the **per-interval** traded volume (first difference of the cumulative stored
field; the `max(0, ·)` guards a provider counter reset).
`doi_X(k,t) = oi_X(k,t) − oi_X(k,t−1)`. `mid_X(k,t)` is `(bid+ask)/2` when both
are positive, else `last_price`.

**Count reconciliation (audit performed before locking):** the REV-1 draft's
prose claimed 14 by bundling the two distinct IV-skew quantities under one
label, while its JSON enumerated 15. Each of the 15 was audited for
duplication, derivation from another entry, or metadata status. **None is a
duplicate and none is derived from another**; F12 (a same-strike put-call IV
spread) and F13 (a cross-strike smile slope) measure different things. The
correct universe is therefore **15**, and 15 is now stated everywhere.

| ID | Name | Exact formula | Lookback | Normalization | Missing-value rule | Group |
|---|---|---|---|---|---|---|
| **F01** | `ce_pe_volume_imbalance_atm1` | `(Σ_{k∈{−1,0,1}} dv_CE(k,t) − Σ dv_PE(k,t)) / (Σ dv_CE + Σ dv_PE + ε)` | 1 snapshot | self-normalizing, range [−1,1] | drop obs if t−1 absent | G1 |
| **F02** | `volume_acceleration` | `TV(t) / (mean_{j=1..5} TV(t−j) + ε) − 1`, where `TV(t)=Σ_{k∈{−2..2}}(dv_CE+dv_PE)` | 6 snapshots | self-normalizing ratio | drop obs if <6 prior snapshots that day | G1 |
| **F03** | `volume_anomaly_tod` | `(TV(t) − μ_ToD) / (σ_ToD + ε)` | prior days | μ,σ over the same 30-min bucket on **prior days only**, expanding, ≥5 prior days | drop obs if <5 prior days available | G1 |
| **F04** | `net_oi_change_imbalance_atm2` | `(Σ_{k∈{−2..2}} doi_CE(k,t) − Σ doi_PE(k,t)) / (Σ|doi_CE| + Σ|doi_PE| + ε)` | 1 snapshot | self-normalizing, [−1,1] | drop obs if t−1 absent | G2 |
| **F05** | `oi_concentration_atm1` | `Σ_{k∈{−1,0,1}}(oi_CE+oi_PE) / Σ_{all strikes}(oi_CE+oi_PE)` | 0 | ratio in [0,1] | drop obs if denominator = 0 | G2 |
| **F06** | `oi_migration` | `(S*(t) − S*(t−1)) / 50`, `S*` = strike maximizing `oi_CE+oi_PE` over all collected strikes | 1 snapshot | strike steps | drop obs if t−1 absent | G2 |
| **F07** | `pcr_oi_change` | `PCR(t) − PCR(t−1)`, `PCR = Σ_all oi_PE / Σ_all oi_CE` | 1 snapshot | difference of a ratio | drop obs if t−1 absent or denominator = 0 | G2 |
| **F08** | `ce_pe_premium_momentum_diff` | `r_CE − r_PE`, `r_X = mid_X(0,t)/mid_X(0,t−1) − 1`, **fixed-strike**: the strike that is ATM at t, evaluated at both t and t−1 | 1 snapshot | return difference | median-impute (train-fitted) if that strike absent at t−1 | G3 |
| **F09** | `atm_straddle_return` | `S(t)/S(t−1) − 1`, `S(t)=mid_CE(0,t)+mid_PE(0,t)`, same fixed-strike rule | 1 snapshot | return | median-impute (train-fitted) if absent at t−1 | G3 |
| **F10** | `atm_iv` | mean of `iv_CE(0,t)`, `iv_PE(0,t)` over those that solve | 0 | vol points (fraction) | median-impute (train-fitted); `iv_available=0` | G4 |
| **F11** | `atm_iv_change` | `atm_iv(t) − atm_iv(t−1)`, fixed-strike | 1 snapshot | difference | median-impute (train-fitted); `iv_available=0` | G4 |
| **F12** | `iv_skew_ce_minus_pe` | `iv_CE(0,t) − iv_PE(0,t)` | 0 | difference | median-impute (train-fitted); `iv_available=0` | G4 |
| **F13** | `iv_skew_slope` | `(mean(iv_CE(+1,t), iv_PE(+1,t)) − mean(iv_CE(−1,t), iv_PE(−1,t))) / 2` | 0 | vol points per strike step | median-impute (train-fitted); `iv_available=0` | G4 |
| **F14** | `dist_to_max_call_oi_norm` | `(K*_CE(t) − spot(t)) / (S(t) + ε)`, `K*_CE` = max-`oi_CE` strike above spot, `S(t)` = ATM straddle mid | 0 | normalized by implied move | drop obs if no strike above spot | G5 |
| **F15** | `dist_to_max_put_oi_norm` | `(spot(t) − K*_PE(t)) / (S(t) + ε)`, `K*_PE` = max-`oi_PE` strike below spot | 0 | normalized by implied move | drop obs if no strike below spot | G5 |

**Auxiliary indicator (not one of the 15):** `iv_available` = 1 if F10–F13 all
solved at t, else 0. `included_in_model_matrix: true`; declared metadata for
count-reconciliation purposes.

All 15 are `included_in_model_matrix: true`. Ablation groups: G1 = {F01,F02,F03},
G2 = {F04,F05,F06,F07}, G3 = {F08,F09}, G4 = {F10,F11,F12,F13},
G5 = {F14,F15}.

**Group rationales.** *G1 Flow:* informed or aggressive directional flow often
prints in options first; option volume has a strong intraday U-shape, so a
prior-day ToD baseline is mandatory rather than cosmetic. *G2 Positioning:*
writer/dealer inventory and changes in where that risk sits constrain or
amplify spot moves; OI levels are non-stationary, so changes are used.
*G3 Premium:* premiums can lead spot when informed flow hits options first;
F08 (difference) and F09 (sum) are the direction and magnitude rotations of the
same CE/PE return pair. *G4 Vol surface:* IV encodes expected magnitude and
skew encodes expected asymmetry — a priori the most plausible source of
incremental value, aligned with the platform's standing finding that
volatility is forecastable. *G5 Structure:* large OI concentrations act as
magnets or barriers; normalizing by the ATM straddle makes the distance
regime-comparable.

**Known weakness disclosed in advance:** for European index options, put-call
parity implies `iv_CE(0,t) ≈ iv_PE(0,t)`, so **F12 is expected to be small and
partly an artifact of the fixed r = 0.065 / q = 0.012 forward assumption**
rather than pure market information. It is retained because removing a feature
after drafting — on a judgement call, before any result — would still be
altering the preregistered universe. Its weakness is recorded so a null or a
spuriously large F12 importance is interpreted correctly.

**IV handling:** IV is derived (never stored) via
`analytics.black_scholes.implied_volatility` at r=0.065, q=0.012, restricted to
`k ∈ {−1,0,+1}` where coverage is high (~79% overall). Missingness is **not
random** (far-OTM, illiquid), so it is handled by **train-fitted median
imputation plus the `iv_available` indicator** — never by silent
full-sample imputation.

`oi_change` as stored is unusable (0% nonzero on every day) and is never used;
OI change is derived by differencing `open_interest` per (strike, option_type).

---

## Walk-forward design (LOCKED)

Expanding window, **minimum 30 trading days** to fit, test blocks of **5
trading days**, step 5 → ~6 folds covering days 31–58 = **28 out-of-sample
trading days**, versus EXP033's 19. The model is refit at each fold boundary
and then frozen for that fold's test block.

- **Purge:** any training observation whose label window overlaps the test
  block is removed.
- **Embargo:** h snapshots either side of each boundary, on top of the
  structural t+1 offset.
- Labels never cross a day boundary, so no fold boundary can straddle one.
- Folds are assembled in **trading-day space**. (`generate_windows()` is
  calendar-day based; EXP033 established this sample runs at ~0.72 trading
  days per calendar day, and EXP034 does its fold arithmetic in trading days
  directly to avoid repeating that error.)
- Every metric is reported **per fold** and **per trading day**, never only
  pooled.

---

## Evaluation and inference (LOCKED)

**Regression:** out-of-sample R² against MODEL_0 (`1 − SSE_model/SSE_naive`),
RMSE, and Pearson & Spearman information coefficients — pooled, per fold, and
per day.

**Classification (secondary T3):** log loss, Brier score, reliability by
decile; ROC-AUC secondary only.

**Incremental-value test:** the **Clark-West adjusted-MSPE** statistic,
one-sided. Standard Diebold-Mariano is invalid for nested models under the
null and is not used. `stat_tests.py` contains no DM/CW today, so this is new
EXP034-owned code.

**Uncertainty:** primary inference is a **stationary block bootstrap over
whole trading days** (reusing `analytics.data_snooping.stationary_bootstrap_indices`),
which respects intraday dependence. HAC/Newey-West at lag ≥ h is reported as a
cross-check. A **non-overlapping replication** (sampling every h-th snapshot)
is a pre-registered robustness check.

**Timing-artifact control:** the no-gap variant (label base at t instead of
t+1) is computed and reported **as a strictly diagnostic quantity**. It can
only ever *invalidate* a positive finding, never create or rescue one: if an
effect is present with the no-gap variant but absent with the locked t+1
definition, the effect is declared a timing artifact.

---

## Multiple-testing universe (LOCKED)

**Primary family: 4 hypotheses** — for each of T1a, T1b, T2a, T2b:
*H1: MODEL_3 achieves better OOS predictive accuracy than MODEL_2* (one-sided).
Benjamini-Hochberg FDR at **q = 0.10** across these 4. BH validity here rests
on positive regression dependence among correlated targets, which is stated
rather than assumed silently.

**Beating MODEL_1 but not MODEL_2 is not primary evidence of incremental
option-chain information**, and may not be reported as such: option IV, skew
and activity plausibly re-encode recent realized volatility, which MODEL_2
already contains.

**Secondary family (FDR-corrected separately, never able to establish a
primary claim):** T3a/T3b; MODEL_2 vs MODEL_1; MODEL_3 vs MODEL_1;
expiry vs non-expiry stratification; the LightGBM robustness check.

Feature-group ablations (G1–G5) and permutation importance run **only if** a
primary hypothesis succeeds, and only within the same OOS folds — never
in-sample.

---

## Pre-run consistency check (LOCKED — must pass before any execution)

Before any EXP034 run, a preflight validation compares the locked
configuration against the implementation's actual inputs and **exits non-zero
on any mismatch**, aborting the experiment. At minimum it verifies:

- chain feature count == 15, and every feature ID F01–F15 present exactly once;
- auxiliary indicator count == 1 (`iv_available`);
- realized model-matrix column counts == MODEL_1 30 / MODEL_2 35 / MODEL_3 51,
  after excluding explicitly declared metadata columns;
- MODEL_1 ⊂ MODEL_2 ⊂ MODEL_3 (strict nesting actually holds in the built
  matrices, since Clark-West validity depends on it);
- primary target count == 4 and target IDs match;
- primary hypothesis count == 4;
- ablation-group membership matches the registry exactly and partitions F01–F15;
- the audit columns `feature_timestamp`, `prediction_start_timestamp`,
  `prediction_end_timestamp`, `elapsed_seconds_t_to_t1` are present and
  `prediction_start_timestamp > feature_timestamp` for every row;
- the config-file hash matches the value recorded at lock time.

This mirrors the EXP033 freeze tooling, which exists precisely to catch the
class of drift that produced the 14-vs-15 discrepancy in REV-1.

---

## Power analysis (procedure LOCKED; numbers recorded before any OOS scoring)

Minimum detectable effect is computed for each target from the **initial
30-day training block only**, and written into a locked addendum **before any
out-of-sample fold is scored**.

Prior expectation, stated in advance: with ~58 days, day-clustered standard
errors on an information coefficient are roughly 0.02–0.04, so only
|IC| ≳ 0.06–0.11 is detectable, while realistic intraday option-flow ICs are
0.02–0.05. **T1 (signed return) is therefore pre-declared as very likely
underpowered.** T2 (magnitude) is where a genuine positive is plausible,
consistent with the platform's standing finding that volatility — not
direction — is forecastable.

### Interpretation rule (LOCKED)
**"No evidence of incremental information" and "evidence of no incremental
information" are different conclusions and must never be conflated.** At this
sample size a T1 null means the former. A null may be reported as *evidence of
absence* only if the pre-computed minimum detectable effect is **smaller** than
the effect size that would be economically or scientifically meaningful —
which, per the numbers above, is not expected to hold for T1.

**Confound stated in advance:** IV and VIX features may largely re-encode
recent realized volatility. This is precisely why MODEL_3-vs-**MODEL_2** is
the primary test.

---

## Success criteria (LOCKED)

### EXP034-A (exploratory) — all five must hold, per target
1. Mean Δ OOS-R² (MODEL_3 − MODEL_2) > 0, pooled across the OOS period;
2. Clark-West one-sided p < 0.05 under the day-block bootstrap;
3. Survives BH-FDR at q = 0.10 across the 4 primary hypotheses;
4. The improvement is positive in **≥ 2/3 of folds** (stability, not one fold);
5. The effect survives the locked t+1 timing definition (i.e. is not
   dependent on the diagnostic no-gap variant).

Verdict if all five hold: **EXPLORATORY EVIDENCE OF INCREMENTAL INFORMATION**
for target X. This authorizes exactly one thing: running EXP034-C. It does not
authorize strategy work, paper trading, or capital, and may not be described
using the prohibited vocabulary listed earlier.

### Confirmation (EXP034-C) — required for any non-exploratory claim
A finding is **CONFIRMED** only if EXP034-A's five criteria hold **and**
EXP034-C independently reproduces the **same sign** of improvement on the
forward-only sample, on the same target, with the same locked pipeline. Any
other EXP034-C outcome leaves the finding exploratory.

## Failure criteria (LOCKED)

Any of: Δ OOS-R² ≤ 0; Clark-West p ≥ 0.05; fails FDR; unstable across folds;
or the effect depends on the no-gap diagnostic → **NO EXPLORATORY EVIDENCE OF
INCREMENTAL INFORMATION.**

**Third band, pre-defined:** point estimate positive but the confidence
interval spans zero *and* the estimate is below the pre-computed minimum
detectable effect → **INCONCLUSIVE — UNDERPOWERED.** Reported as such, never
as evidence of absence.

**The negative result is an acceptable and expected outcome.** No search
continues after it.

---

## Anti-snooping guards (LOCKED)

- No threshold, feature, target, horizon or model may be added, removed or
  tuned after the first OOS fold is scored.
- A weak result is never grounds for adding features.
- A losing target is never dropped; all four primaries are reported.
- The LightGBM check cannot rescue a ridge null, and a ridge null may not
  trigger a nonlinear-model switch inside this experiment.
- Feature importance is computed only after a primary success, only OOS.
- Any genuinely new idea arising during the run belongs to EXP035+.

## Scope

Read-only research on collected chains plus EXP034's own derived panel. No
costs, no fills, no orders, no paper trading, no deployment. EXP033 remains
untouched and verifiable via `python scripts/exp033_freeze.py --verify`.

---
*Drafted 2026-09-12 (REV-2). Not locked until committed.*
