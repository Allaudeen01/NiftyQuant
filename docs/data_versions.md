# Data-Generating Process Changelog

Required by `RESEARCH_POLICY.md` §4. Every change to collection semantics that
alters **information timing, field meaning, or feature observability** gets a
new version and an entry here with its effective date.

## GOVERNING INVARIANT

> **No data collected under dgp-v1 may ever be silently pooled with dgp-v2 for
> a confirmatory experiment.**

Enforced structurally, not procedurally:

1. every dgp-v2 row carries `dgp_version`;
2. dgp-v2 writes to a **separate root** — `data/option_chain_v2/` — so a glob
   of the v1 root cannot pick it up;
3. `nifty_quant.data.dgp.assert_single_version()` raises on a mixed frame.

**Cost, stated explicitly.** The pre-committed EXP034-C replication requires
**100 usable dgp-v1 days** and stands at **57**. dgp-v2 days do **not** count
toward it. Cutting over resets that counter; rollback recovers the collector
but not the counter.

---

## dgp-v1 — from 2026-06-22, currently ACTIVE (default)

The process that produced the existing 59-day archive under
`data/option_chain/`.

| Property | Behaviour |
|---|---|
| Timestamp | one `snapshot_ts` per poll, stamped **before** any network call |
| Per-contract timing | **not recorded** — every row of a poll shares one timestamp |
| Fetch order | strike-ascending (`sorted(strike, option_type)`), batched 50 with 1.2 s pause → skew **monotonic in strike** |
| Strike band | full ladder fetched, then trimmed post-hoc |
| Spot | read **last**, inside the chain fetch, behind a 30 s cache; silent daily-close fallback |
| Expiries | fetched sequentially, 1.5 s apart; near expiry inferred downstream as `min(expiry)` |
| Retry | whole-chain; one failed batch discards and refetches all, after 4–12 s backoff, still stamped with the original time |
| Partial poll | written with whatever succeeded; not marked |
| Storage | `data/option_chain/YYYY/MM/DD/HH_MM.parquet` (minute granularity) |
| Dead columns | `oi_change` (0 % nonzero on every day), `implied_volatility` (always NaN) |

**Known limitations, all documented elsewhere:** frozen-spot session
2026-06-26 (daily-close fallback, undetected until EXP033); cumulative-volume
semantics mistaken for increments (EXP033 E1); within-expiry skew 2.4 s,
full-poll 6.3 s; minute-granularity filenames colliding at sub-minute cadence
(2026-06-22, 328 timestamps in 170 files).

---

## dgp-v2 — IMPLEMENTED, NOT YET ACTIVE (flag `--dgp-v2`, default OFF)

Effective date: **not yet cut over.** No dgp-v2 data has been collected.

| # | Change | v1 → v2 |
|---|---|---|
| 1 | Per-contract timestamps | none → `observed_ts`, `batch_index`, `fetch_rank`, `fetch_attempt` per row |
| 2 | Spot | read last, cached, silent fallback → observed **before and after** the poll, cache bypassed, `spot_source` recorded |
| 3 | Fetch order | strike-ascending → seeded shuffle (`order_seed` recorded); rate-limit neutral |
| 4 | Batch architecture | full ladder fetched then trimmed → **pre-fetch** band filter (roughly halves contracts and removes 1–2 batches). Concurrency deliberately **not** adopted |
| 5 | Identity | timestamp-as-identity → explicit `poll_id`, `snapshot_id`; filenames resolve to seconds |
| 6 | Expiry metadata | inferred downstream → `expiry_rank`, `expiries_intended` recorded at collection |
| 7 | Contract identity | implicit `(expiry,strike,type)`; token discarded → `contract_uid`, `token`, `trading_symbol` stored |
| 8 | Schema | `oi_change` and `implied_volatility` dropped; v2 columns added |
| 9 | Version | implicit → `dgp_version`, `collector_sha`, `provider_sha` per row |
| 10 | Retry | whole-chain, timestamps invisible → **batch-level**; successful batches keep their true timestamps; `fetch_attempt` in the primary key so a retry cannot silently overwrite |
| 11 | Backward compat | — | separate root; frozen v1 readers untouched |
| 13 | Certification | retrospective, in analysis code → `certify_poll()` at collection time |

**Not adopted:** request concurrency. It is the only proposal that materially
changes rate-limit pressure — `_throttle()` serializes all SDK calls, so
concurrency would require replacing it with a token-bucket limiter. Pre-fetch
band trimming already captures most of the spread reduction at zero
rate-limit risk.

### Validation status

`python scripts/dgp_v2_selftest.py` — **33/33 pass**, offline against a stub
provider, in a throwaway directory. Covers: schema conformance, per-contract
timing, spot bracketing, order decorrelation, band pre-filtering, identity
uniqueness, expiry ranking, dead-column absence, batch-failure retry,
spot-unavailable abort, fallback surfacing, the mixed-version guard, and
sub-minute filename resolution.

The suite includes a deliberate **negative control**: with `--no-shuffle` it
asserts the strike/fetch-order correlation returns to ρ > 0.95, reproducing the
v1 defect on demand and proving the shuffle is what removes it.

### Cutover procedure (not yet executed)

1. Implement behind `--dgp-v2`, default off. ✅ **done**
2. Run both collectors in parallel ≥3 sessions — **costs double API budget;
   verify against the rate limit first.**
3. Validate: modelled-vs-measured skew agreement on the first v2 day;
   certification replay; all three freeze verifications.
4. Pick a cutover date, log it here, flip the default, stop v1 writes.
5. Re-verify all freezes; confirm EXP034-C's filter counts **0** v2 days.

### Rollback

Flip the flag off. The v1 root was never written by v2, so v1 resumes cleanly.
**Days collected under v2 are not retro-convertible to v1** — for those days
the EXP034-C reset is permanent regardless of rollback.
