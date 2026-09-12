# EXP034 — Errata and Pre-Committed Sensitivity Diagnostics

**Non-binding addendum.** The binding preregistration is
`docs/preregistrations/exp034_option_information_content.md` +
`docs/preregistrations/exp034_config.json`, locked at commit
**`18a178e94dcb2df985c304d3a037de50407acbba`**. That pair is **not edited** by
this file. This document records (a) factual errata in the locked prose, and
(b) sensitivity diagnostics declared **before any feature, target value, model
or result existed** — i.e. before the first OOS fold was scored, as the locked
anti-snooping guards require.

Everything below was written during Phase 1 (data pipeline only). At the time
of writing, no EXP034 feature had been computed, no target value had been
computed, and no model had been fitted.

---

## Erratum 1 — usable-day count is 57, not the 58 stated in the prose

**Locked prose says:** "Usable: 58 days."
**The locked screen returns:** 57.

There is no contradiction to resolve in the binding rules. The locked config
defines usable as *"a day passing the identical per-day data-quality screen
defined in `data.per_day_quality_screen` — no other definition."* The screen is
therefore operative, and the prose figure of 58 was a pre-screen estimate
written before the screen had been executed: it accounted only for the
2026-06-26 spot exclusion.

The screen excludes two days:

| Day | Failing check | Observed |
|---|---|---|
| 2026-06-26 | `pass_spot_std` | spot std = 0.00 (frozen-spot session; `get_spot()` daily-close fallback) |
| 2026-07-07 | `pass_min_snapshots` | 22 snapshots vs the locked minimum of 100 |

2026-07-07 fails by a wide margin (22 vs 100, spanning a 321-minute collection
gap) and was already excluded from EXP033 on the same grounds, so no judgement
call is involved.

**Operative consequences.**
- The EXP034-A analysis sample is **57 usable days**.
- The **EXP034-C trigger** (≥ 100 usable days) is measured by this same screen.
  At 57 today, it requires **43 further qualifying days**, not 42.

**Action taken:** none to the locked files. Recorded here only.

---

## Erratum 2 — no lower bound on the t→t+1 gap

The locked rule caps the t→t+1 separation only from above (300 s). It sets no
lower bound, so abnormally tight consecutive pairs are retained.

Measured on the **near-expiry stream EXP034 actually consumes**, the picture is
narrower than the raw file counts suggest. Only **2026-08-25** runs at double
cadence (median gap 62.0 s); every other day sits at ~131–136 s, and **all 172
sub-30 s rows come from that single day**. 2026-06-22 — flagged as a
double-cadence day in EXP033's audit of the *raw, both-expiry* file set — has a
near-expiry median gap of **136.4 s**, i.e. entirely normal, and contributes
**zero** sub-30 s rows. Its collection artifact does not reach EXP034's
analysis universe at all.

### Correction to an earlier mis-statement of the mechanism
It was initially suggested that short gaps risk "already-realized movement"
entering the features. **That framing was wrong and is retracted.** Features at
snapshot *t* are always observed *before* spot at *t+1*, whatever the gap size,
because within a poll the quotes are fetched first and `get_spot()` last. With
near-expiry-only selection plus the t+1 label base, the ordering-based leakage
channel is closed **structurally**, independent of gap length.

### The actual mechanism: shared microstructure noise
The real contaminant is different, and is a correlation artifact rather than a
leak. At a separation of a few seconds, `spot_t` and `spot_{t+1}` are
effectively the *same* observation. A microstructure noise shock then inflates
a price-history feature (which has `spot_t` in its numerator) while
simultaneously deflating the label (which has `spot_{t+1}` in its denominator),
inducing spurious **reversal** correlation.

Two properties bound its importance for the locked primary test:
1. It is a property of the shared price features, so it contaminates MODEL_1,
   MODEL_2 and MODEL_3 alike and **largely differences away** in the
   incremental MODEL_3 − MODEL_2 comparison, which is the primary hypothesis.
2. It is confined to a small minority of rows on two days out of 57, and cannot
   by itself produce an effect that must also be stable across ≥ 2/3 of folds.

### A second, distinct issue on the same days: horizon heterogeneity
2026-08-25's **median** gap is 62.3 s against 132.5 s on every other day — the
entire session ran at double cadence. Because horizons are defined in
**snapshots**, `h = 5` spans roughly 5 minutes on such a day rather than ~10.
The locked preregistration already anticipates this ("index arithmetic below is
over *valid snapshots*, not clock time", and it mandates reporting the elapsed
distribution per fold), so this is an accepted design property, not a
violation. It is nonetheless worth isolating.

### Observed distribution (retained rows)

Out of 18,920 retained rows:

| t→t+1 gap | rows | share | contributing days |
|---|---|---|---|
| < 5 s | 32 | 0.17% | 2026-08-25 |
| < 15 s | 83 | 0.44% | 2026-08-25 |
| < 30 s | 172 | 0.91% | 2026-08-25 |
| < 60 s | 357 | 1.89% | 2026-06-22, 2026-08-25 |
| < 100 s | 636 | 3.36% | 2026-06-22, 2026-08-25 |
| < 120 s | 646 | 3.41% | 2026-06-22, 2026-08-25 |

Day-median gaps: 2026-08-25 at **62.0 s**, next-lowest day at **131.2 s** — a
clean separation with nothing in between.

### Decision
**The locked preregistration is not amended.** Its value lies in not moving,
and a ~1% edge case on two of 57 days is a poor reason to mutate a binding
document. The concern is instead addressed by **pre-committed sensitivity
diagnostics**, declared here before any result exists.

### Declared diagnostics (metadata only; they never change retention)

| Column | Definition | Rationale |
|---|---|---|
| `short_gap_flag` | `elapsed_seconds_t_to_t1 < 30` | Intra-poll slop is ~2–5 s and microstructure noise decorrelates within seconds; 30 s is a ~6–10× margin over the slop and far below the ~132 s normal cadence. |
| `double_cadence_day_flag` | day median gap `< 100 s` | Selects exactly one day, 2026-08-25 (62.0 s), against a next-lowest of 131.2 s. Any threshold in ~[65 s, 130 s] selects the identical single day, so the choice is immaterial. Flags 724 rows (3.83%). |
| `day_median_gap_seconds` | continuous, per day | Stored continuously so **any** later gap threshold can be evaluated without re-deriving the panel. |

**Binding rule on these diagnostics — identical to the rule already governing
the no-gap variant:** they may only ever **invalidate** a positive primary
finding, never create or rescue one. Concretely, the locked primary analysis
runs on the full retained sample; the flagged-subsample results are reported
alongside. If a primary finding holds on the full sample but disappears once
flagged rows are removed, the finding is declared a timing artifact. If a
primary finding is absent on the full sample but appears only on some
subsample, **that subsample result is discarded**, not promoted.

---

## Recorded for the platform record

The EXP033 cumulative-volume defect (its E1 family z-scored the raw cumulative
volume level while its docstring described a 2-minute increment) is documented
in the EXP034 preregistration as motivation for EXP034's first-difference
definitions. EXP033 itself remains frozen and unrepaired, per
`docs/RESEARCH_POLICY.md`.

---
*Written during EXP034 Phase 1, before any feature, target value, model or
result existed.*
