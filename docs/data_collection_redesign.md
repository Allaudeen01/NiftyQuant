# Data Collection Redesign — Proposal (READ-ONLY DESIGN DOCUMENT)

**Status: PROPOSAL ONLY. No collector code is changed by this document.**
No successor experiment is created. Implementation requires explicit approval.

Motivation: three consecutive experiments (EXP033, EXP034-A, EXP035) each hit
limits that trace to *collection design* rather than to market structure or to
analysis method. This document proposes what to change, why, in what order, and
at what cost.

---

## 0. The governing constraint before anything else

Any change that alters **information timing, field meaning, or feature
observability** starts a new `dgp-v2` under `RESEARCH_POLICY.md` §4. Forward
days under `dgp-v2` **cannot be pooled** with `dgp-v1` days.

**Concrete consequence:** the pre-committed EXP034-C replication requires 100
usable `dgp-v1` days and stands at 57. Any change in §§1–5 below resets that
counter to zero. That is not a reason to avoid the change — a bad
data-generating process should not be preserved to reach a day count sooner —
but it must be a deliberate decision, taken once, with eyes open.

**Recommended sequencing follows from this:** make *all* breaking changes in a
single cutover, on a chosen date, rather than drip-feeding them. One reset, not
five.

---

## 1. Underlying spot timing

**Current.** `get_option_chain()` fetches all option quotes first, then calls
`get_spot()` **last**, behind a 30-second cache. The recorded `snapshot_ts` is
stamped *before* any network call. Net effect: near-expiry quotes are a few
seconds *older* than the recorded spot (the safe direction); the far expiry
reuses the cached spot, making its quotes *newer* than spot (the unsafe
direction); and everything is stamped with a time that precedes it all.

**Problems.** (a) The near/far asymmetry forced EXP034 to restrict to near
expiry purely for timing hygiene. (b) `snapshot_ts` is not the observation time
of anything. (c) The daily-close fallback in `get_spot()` silently pinned spot
to a constant for an entire session (2026-06-26), which only surfaced as a
downstream audit finding.

**Proposal.**
- Fetch spot **first and last**, recording both (`spot_pre`, `spot_post`) with
  their own timestamps. The pair bounds the underlying's movement across the
  poll and makes staleness self-evident.
- Record `spot_source` explicitly (`live_ltp` vs `daily_close_fallback`) so a
  fallback day is visible at collection time, not three experiments later.
- Drop the spot cache within a poll, or reduce its TTL below the poll's own
  duration, so the two reads are genuinely independent.

---

## 2. Per-contract timestamps

**Current.** Every row of a poll shares one `snapshot_ts`. A 224-row poll file
carries exactly **one** distinct timestamp. Per-contract observation time is
not recorded and is not reconstructible.

**Problems.** This single gap is what made EXP035's migration family
*unfalsifiable*: the sequential-collection dependency could be neither measured
nor excluded. It also caps what any future cross-sectional analysis can claim.

**Proposal (highest value change).**
- Record `observed_ts` **per contract row** — the wall-clock time the batch
  containing that contract returned.
- Record `batch_index` and `batch_size` per row.
- Retain `snapshot_ts` unchanged as the poll identifier, so the join key and
  file layout are untouched.

This is additive at the row level and turns an unmeasurable confound into a
measurable, controllable covariate.

---

## 3. Within-snapshot synchronization

**Current.** `_fetch_market_data()` chunks tokens in batches of 50 with a
`_throttle()` (1.2 s) between batches. With 168–210 contracts per expiry, that
is 4–5 batches and **3.6–4.8 s** of spread inside a single "snapshot".

**Problems.** A "snapshot" is not a snapshot. Any cross-sectional comparison —
which strike is most active, where activity is concentrated, whether it moved —
compares observations taken seconds apart.

**Proposal.**
- Reduce the fetched universe to a configured strike band **before** fetching,
  rather than fetching the full ladder and trimming afterwards. The collector
  already trims post-hoc; trimming pre-fetch cuts contracts roughly in half and
  removes 1–2 batches outright.
- Where the API permits, raise batch size or issue batches concurrently under
  the rate limit, so spread is bounded by latency rather than by
  `n_batches × pause`.
- Record the realized spread per poll (§2 makes this possible) and expose it as
  a first-class quality metric (see the dashboard).

---

## 4. Sequential collection skew

**Current.** `option_instruments()` ends with
`sorted(out, key=lambda o: (o.strike, o.option_type.value))`. Token order is
therefore **strike-ascending**, so the batch skew is **monotonic in strike** —
low strikes always observed first, high strikes always last, every poll, every
day.

**Problems.** A systematic, direction-carrying bias aligned exactly with the
axis a strike-localized study measures along. Random skew would add noise;
monotonic skew adds *structure* that can imitate a finding.

**Proposal.**
- **Randomize token order per poll** (seeded and recorded), so skew becomes
  noise rather than a strike-aligned gradient. This is a one-line change with
  disproportionate scientific value.
- Record the realized order (or its seed) so the skew is reconstructible after
  the fact.
- Prefer interleaving strikes across batches over contiguous strike blocks.

Note these are complementary, not alternatives to §3: randomization removes the
*bias*, reduced spread removes the *magnitude*.

---

## 5. Snapshot frequency

**Current.** ~132 s median cadence (`--poll 120` plus ~12 s of work). Observed
p01 33 s, p99 144 s, and one 19,305 s outage.

**Problems.** A ~2-minute grid is coarse relative to "short-horizon response".
EXP033's triple-barrier work could not resolve 5-point barriers within an
interval; EXP034 had to define horizons in snapshots rather than clock time;
EXP035 found that a burst can straddle the batch window.

**Proposal.**
- Target a 30–60 s cadence for the near expiry, which is the only expiry any
  recent experiment has used. Fetching a narrower band (§3) buys most of the
  budget for this.
- Consider a two-tier schedule: near expiry at high frequency, next expiry at
  the current rate, since nothing currently depends on next-expiry granularity.
- Record the intended cadence in the file so the realized-vs-intended gap is
  auditable.

---

## 6. Near-expiry identification

**Current.** Near expiry is inferred downstream as `min(expiry)` present in the
file. It is not recorded at collection time.

**Problems.** Inference is correct today but fragile: it depends on what the
collector happened to fetch, and it silently changes meaning on expiry day when
the front contract expires intraday. EXP035 had to verify per-snapshot that the
minimum expiry was constant within a session.

**Proposal.**
- Record `expiry_rank` (0 = near, 1 = next) per row at collection time.
- Record `dte_calendar` and an `is_expiry_day` flag per poll.
- Record the expiry list the collector *intended* to fetch, so a partial fetch
  is distinguishable from a genuinely shorter ladder.

---

## 7. ATM identification

**Current.** ATM is derived downstream as the strike nearest spot. With spot
recorded after the quotes, the ATM assignment inherits the spot's timing.

**Problems.** ATM rolls on **10.71%** of snapshots, and every relative-strike
label changes identity at a roll. This is the exact mechanism behind the
Phase 0 error now codified in `RESEARCH_POLICY.md` §2.

**Proposal.**
- Record `atm_strike` per poll at collection time, computed from a recorded
  spot with a recorded timestamp, so downstream consumers share one definition
  rather than each re-deriving it.
- Record an `atm_rolled` flag per poll.
- Keep relative-strike mapping strictly a *downstream* operation — never store
  relative labels as if they were identity.

---

## 8. Contract identity tracking

**Current.** Identity is implicit in `(expiry, strike, option_type)`. The Angel
`token` is fetched and used, then discarded before storage.

**Problems.** Implicit identity is what allows the relative-strike differencing
error to compile and run. Storing explicit identity makes the correct operation
the natural one.

**Proposal.**
- Store the broker `token` and `trading_symbol` per row. They are already in
  hand at fetch time and thrown away.
- Emit a stable `contract_uid = f"{underlying}:{expiry}:{strike}:{option_type}"`.
- With `token` stored, any contract redefinition or master refresh becomes
  detectable rather than silent.

---

## 9. Raw versus derived storage

**Current.** Raw quotes are stored; `implied_volatility` is stored as all-NaN
by design and derived on read; `oi_change` exists in the schema but is **never
populated** (0% nonzero on every day) and must be derived by differencing.

**Problems.** A permanently-null column and a permanently-zero column are traps
— EXP033 and EXP034 each had to discover independently that `oi_change` is
unusable. Meanwhile the genuinely expensive derivation (IV) is recomputed by
every consumer.

**Proposal.**
- **Drop `oi_change` from the schema**, or populate it correctly. A column that
  is always zero is worse than an absent one.
- Keep IV derived rather than stored (the current choice is right — it keeps
  pricing assumptions in the analysis layer), but publish a **cached derived
  store** with the assumptions recorded, so consumers share one computation.
- Maintain the strict raw/derived separation: raw files remain append-only and
  never back-filled; every derived artifact records the code version that made
  it.

---

## 10. Data-quality certification

**Current.** `option_quality_report.py` gives a per-day GOOD/REVIEW verdict.
Certification of `dgp` semantics is done retrospectively, per experiment, by
each experiment's own code.

**Problems.** Three experiments each re-implemented day screening, and each
found something the previous had not — the frozen-spot day, the cumulative
volume semantics, the 22-snapshot day, the strike-monotonic skew. The checks
lived in analysis code rather than beside the data.

**Proposal.**
- Emit a **per-day certification record at collection time**, not
  retrospectively: snapshot count vs intended, realized cadence, within-poll
  spread, spot source and staleness, two-sided quote rate, crossed quotes,
  contract count, batch count, expiry ranks fetched.
- Version it as `dgp_version` in the record itself, so a day carries its own
  provenance.
- Maintain `docs/data_versions.md` as the changelog §4 already requires.
- Expose all of it in the diagnostics dashboard so drift is visible the day it
  happens rather than at the next audit.

---

## Prioritization

| # | Change | Value | Breaks `dgp-v1`? |
|---|---|---|---|
| **2** | Per-contract timestamps + batch index | **Highest** — converts an unmeasurable confound into a covariate | No (additive) — but see note |
| **4** | Randomize token order | **Highest** — removes a structured, direction-carrying bias for one line | **Yes** |
| 3 | Pre-fetch band trim, reduce spread | High — halves contracts, removes batches | **Yes** |
| 1 | Spot pre/post + source flag | High — kills a whole class of silent failure | **Yes** |
| 8 | Store token / symbol / uid | Moderate — makes correct operations natural | No (additive) |
| 5 | Faster cadence | Moderate–high, but costs API budget | **Yes** |
| 6,7 | Record expiry rank, ATM, roll flag | Moderate — removes downstream re-derivation | No (additive) |
| 9 | Drop dead `oi_change`, cache IV | Moderate — removes known traps | Schema change |
| 10 | Collection-time certification | Moderate — shifts discovery earlier | No (additive) |

**Recommended plan.** Bundle **1, 3, 4, 5** (all breaking) into a single
`dgp-v2` cutover on a chosen date, and ship **2, 6, 7, 8, 9, 10** alongside it
since they are mostly additive and there is no reason to pay the reset twice.
Log the cutover in `docs/data_versions.md` with its effective date.

**What this does not fix.** Better collection does not change the arithmetic in
`reports/exp035/phase0_5_event_power_stability.md`: a day-clustered event study
still faces MDE(IC) ≥ 0.332 at 59 days. Collection quality determines whether a
measurement is *trustworthy*; it does not determine whether the sample is *large
enough*. Those are separate problems and only one of them is addressed here.

---
*Proposal only. No collector change is authorized by this document.*
