# EXP035 — Design Archive

## STATUS: DESIGN TERMINATED BEFORE PREREGISTRATION

**REASON:** No candidate event family is both measurement-valid and
statistically feasible using the current NIFTY-only dataset.

EXP035 never reached preregistration. It was stopped at the design-audit stage,
by design, because the audits showed the dataset could not support the
hypothesis family.

---

## What EXP035 was NOT

**EXP035 is not a failed trading strategy.** No trading strategy was ever
built, proposed, or evaluated. Specifically, across Phase 0 and Phase 0.5:

- **No future outcome was inspected** — no future spot return, target hit, stop
  hit, P&L, win rate, Sharpe, information coefficient, or outcome correlation.
- **No model was fitted** — no Ridge, no LightGBM, no estimator of any kind.
- **No backtest was run.**
- **No predictive hypothesis was tested.** Not one. The work stopped before any
  hypothesis could be stated formally enough to test.
- **Nothing was labelled using future prices.**

Every number produced by EXP035 is a property of predictors, data quality,
collection timing, or event counts. The termination decision was reached purely
from measurement validity and statistical feasibility, never from performance.

This distinction matters for the research record: a design that is abandoned
*before* looking at outcomes carries no selection bias and consumes no
inferential budget. EXP035 leaves the dataset exactly as unexamined as it found
it.

---

## Candidate research question (never preregistered)

> *Does abnormal localized incremental option activity at a particular strike
> precede a measurable short-horizon NIFTY response?*

---

## Phase 0 — Hypothesis and data capability audit

Report: `reports/exp035/phase0_report.md`
Script: `scripts/exp035_phase0_audit.py`

Screened eight candidate quantity families for causal measurability. Findings:

- **Supported:** incremental volume (on absolute-contract identity), incremental
  OI, CE/PE imbalance at a strike, strike concentration, strike migration,
  quoted spread, contemporaneous premium response.
- **Not supported:** cross-sectional z as an anomaly detector (bounded at
  √10 = 3.162 over an 11-strike band, structurally ATM-dominated, fires on
  66.8% of snapshots); quote-change indicator (degenerate, ~98.5%); CE/PE
  imbalance as a band-level event (98.1%); strike-localized premium response as
  independent information (ρ = 0.958 across strikes); pooling near and next
  expiry (~69× liquidity gap).
- **Key structural finding:** "localized" volume is largely *not* localized —
  ΔVolume correlates across adjacent strikes and CE/PE at ρ = 0.86–0.93, a
  common market-activity factor.

Five of seven candidate event families were degenerate. Two survived:
**concentration** and **migration**.

### Methodology error found and corrected in Phase 0

The first audit execution reported 5.3% negative incremental volume and broke
every bounded derived measure. Cause: gathering at relative-strike indices
*before* differencing, so an ATM roll (10.71% of snapshots) subtracted two
different contracts. Corrected ordering reproduced EXP034's independently
established 0.0102% negative rate.

Classified as **an implementation error discovered before any outcome
inspection** — not a data defect and not a market finding. It is now
generalized as a standing rule in `RESEARCH_POLICY.md` §2 (contract-identity
differencing).

---

## Phase 0.5 — Event power and stability audit

Report: `reports/exp035/phase0_5_event_power_stability.md`
Script: `scripts/exp035_phase0_5_audit.py`

### CONCENTRATION — DESIGN NOT VIABLE FOR CURRENT NIFTY-ONLY SAMPLE

The measurement is valid: bounded, artifact-free, causally available, robust
once the contract-identity rule is applied. It fails on two other grounds.

**Threshold stability.** The frequency curve is a cliff, not a curve — event
count collapses ~36× between θ=0.25 (3,433 events) and θ=0.35 (95). No plateau
exists on which to place a stable definition.

**Effective sample size.** Events pile onto a few sessions: at θ=0.40, 65 events
occur on 8 days (8.1 per event-day, 91.2% within 5 snapshots of each other).
Treating the event-day as the independent unit gives 6–36 effective clusters
depending on threshold.

**The required calendar duration is not an actionable solution.** Because
same-day events are dependent, the effective sample is capped near one cluster
per trading day, which floors the minimum detectable effect at
`(1.6449+0.8416)/√(N−3)` — **0.332 at today's 59 days**, against an externally
pre-specified plausible range of IC 0.02–0.05. Reaching 0.05 needs ~2,476
event-days ≈ **16 years** at θ=0.30. No threshold choice moves this floor.

### MIGRATION — NOT SUPPORTED BY CURRENT DATA

**Sequential collection timing cannot be ruled out as a dependency using the
recorded metadata.**

Structural exposure established from code: `option_instruments()` returns
`sorted(out, key=(strike, option_type))` and `_fetch_market_data()` chunks that
list into 50-token batches with a 1.2 s throttle. Contract order is therefore
strike-ascending and the within-snapshot observation skew is **monotonic in
strike** — 3.6–4.8 s across the full ladder (168–210 contracts per expiry).
Per-contract observation timing is **not recorded**: every row of a poll shares
one `snapshot_ts`, and batch membership depends on scrip-master ordering that is
neither stored nor reconstructible.

Indirect diagnostics neither convicted nor exonerated the measure:

- Synchronized-band control (ATM±1) showed a 71× collapse, but is **confounded**
   — that band has 3 strikes, so `|Δcom| ≥ 1` is mechanically harder there.
- Batch-count dependence pointed the right way (2.72% vs 0.67%) but rested on
  **one day**; Spearman(rate, n_batches) = +0.202 at n=59.
- The **direction test contradicted** the artifact model: monotonic low→high
  skew predicts positive Δcom, but observed migration is 61% negative. The
  simple mechanism does not explain the data, and the asymmetry is unexplained.

Per the governing instruction, inability to exclude the dependence using
recorded metadata is itself disqualifying. **No repair was attempted**, and
none should be attempted inside EXP035.

Power independently fails: 27 effective clusters at the most favourable
threshold → MDE 0.508, ~10× the top of the plausible range.

---

## Preserved artifacts

| Artifact | Path |
|---|---|
| Phase 0 report | `reports/exp035/phase0_report.md` |
| Phase 0 audit script | `scripts/exp035_phase0_audit.py` |
| Phase 0 capability table | `reports/exp035/phase0_capability_audit.csv` |
| Phase 0 redundancy matrix | `reports/exp035/phase0_redundancy.csv` |
| Phase 0 ATM-roll / timing table | `reports/exp035/phase0_atm_roll.csv` |
| Phase 0.5 report | `reports/exp035/phase0_5_event_power_stability.md` |
| Phase 0.5 audit script | `scripts/exp035_phase0_5_audit.py` |
| Phase 0.5 frequency curves | `reports/exp035/phase0_5_event_frequency.csv` |
| Phase 0.5 migration artifact table | `reports/exp035/phase0_5_migration_artifact.csv` |
| Phase 0.5 collection metadata | `reports/exp035/phase0_5_collection_meta.csv` |
| This archive | `docs/exp035_design_archive.md` |

EXP032, EXP033 and EXP034-A were untouched throughout; their freeze
verifications pass and their tags are intact.

---

## What EXP035 contributed

Although terminated, the design work produced durable assets:

1. **A standing methodology rule** (`RESEARCH_POLICY.md` §2) that will prevent a
   silent class of error in every future contract-level difference.
2. **A quantified structural limit** on this dataset: any day-clustered event
   study has MDE(IC) ≥ 0.332 at 59 days, and ≥ 0.05 only past ~2,500 days. This
   bounds what *any* successor event study can hope to detect, not just EXP035.
3. **A concrete, code-level characterization** of the sequential-collection
   skew, which is now the primary driver of the data-collection redesign
   (`docs/data_collection_redesign.md`).
4. A demonstration that stopping before outcome inspection is practical, and
   costs nothing inferentially.

---
*Archived without preregistration. No outcome analysis was ever performed.*
