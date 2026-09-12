"""Data-generating-process versioning for the collector.

`dgp-v1` is the historical process: one `snapshot_ts` shared by every row of a
poll, strike-sorted sequential fetching, spot read last behind a 30s cache, and
no per-contract observation timing.

`dgp-v2` records what v1 could not: per-contract observation timestamps, batch
index, fetch order, explicit poll identity, spot observed before AND after the
chain fetch with its source, and poll completion status.

GOVERNING INVARIANT
-------------------
No data collected under dgp-v1 may ever be silently pooled with dgp-v2 for a
confirmatory experiment. Enforcement is structural, not procedural:

  1. every v2 row carries `dgp_version`;
  2. v2 writes to a SEPARATE root (`data/option_chain_v2/`), so a naive glob of
     the v1 root cannot pick it up;
  3. `assert_single_version()` refuses a mixed frame.

This matters concretely: EXP034-C requires 100 usable **dgp-v1** days and stands
at 57. Cutover resets that counter; v2 days do not count toward it.
"""

from __future__ import annotations

import itertools
import time
from datetime import datetime

import pandas as pd

DGP_V1 = 1
DGP_V2 = 2

V1_ROOT = "option_chain"
V2_ROOT = "option_chain_v2"

# v2 schema. Note what is ABSENT by deliberate choice:
#   * `oi_change`          -- 0% nonzero on every v1 day; a permanently-zero
#                             column is worse than an absent one.
#   * `implied_volatility` -- always NaN in v1; IV derivation belongs in the
#                             analysis layer where its r/q assumptions live.
V2_COLUMNS = [
    # identity
    "dgp_version", "poll_id", "snapshot_id", "contract_uid",
    "underlying", "expiry", "expiry_rank", "strike", "option_type",
    "token", "trading_symbol",
    # timing
    "poll_started_ts", "observed_ts", "batch_index", "fetch_rank", "fetch_attempt",
    # underlying, observed either side of the chain fetch
    "spot_pre", "spot_pre_ts", "spot_post", "spot_post_ts", "spot_source",
    # quote payload
    "last_price", "bid", "ask", "volume", "open_interest",
    # poll-level provenance
    "poll_status", "expiries_intended", "order_seed",
    "collector_sha", "provider_sha", "context",
]

# Primary key. `fetch_attempt` is part of the key so a retry can never silently
# overwrite a successful earlier observation via drop_duplicates(keep="last").
V2_KEYS = ["poll_id", "expiry", "strike", "option_type", "fetch_attempt"]

POLL_STATUS_COMPLETE = "complete"
POLL_STATUS_PARTIAL = "partial"
POLL_STATUS_FAILED = "failed"

SPOT_SOURCE_LIVE = "live_ltp"
SPOT_SOURCE_FALLBACK = "daily_close_fallback"
SPOT_SOURCE_UNAVAILABLE = "unavailable"

_counter = itertools.count()


def new_poll_id(ts: datetime | None = None) -> str:
    """Monotonic, human-readable poll identifier.

    Wall-clock alone is not safe as an identifier (an NTP step can move it
    backwards), so a process-local counter is appended to guarantee
    monotonicity within a session.
    """
    t = pd.Timestamp(ts) if ts is not None else pd.Timestamp(datetime.now())
    return f"{t.strftime('%Y%m%dT%H%M%S')}.{t.microsecond // 1000:03d}-{next(_counter):06d}"


def contract_uid(underlying: str, expiry, strike: float, option_type: str) -> str:
    """Canonical contract identity. Never key on broker token alone -- tokens
    can be reused across expiries."""
    return f"{underlying}:{pd.Timestamp(expiry).date().isoformat()}:{strike:g}:{option_type}"


def snapshot_dir(data_dir: str, ts, version: int = DGP_V2) -> str:
    root = V2_ROOT if version == DGP_V2 else V1_ROOT
    t = pd.Timestamp(ts)
    return f"{data_dir}/{root}/{t.year:04d}/{t.month:02d}/{t.day:02d}"


def snapshot_filename(ts) -> str:
    """Second-granularity filename.

    v1 used HH_MM.parquet, which collides once polling is faster than once per
    minute -- exactly how 2026-06-22 accumulated 328 timestamps across 170
    files. v2 targets 30-60s cadence, so the filename must resolve seconds.
    """
    t = pd.Timestamp(ts)
    return f"{t.hour:02d}_{t.minute:02d}_{t.second:02d}.parquet"


def assert_single_version(df: pd.DataFrame) -> int:
    """Refuse a frame that mixes data-generating processes.

    A v1 frame has no `dgp_version` column at all; a v2 frame has it on every
    row. Anything else is a pooling error and must fail loudly.
    """
    if "dgp_version" not in df.columns:
        return DGP_V1
    versions = set(pd.unique(df["dgp_version"].dropna()))
    if len(versions) != 1:
        raise ValueError(
            f"refusing to operate on mixed data-generating processes: {sorted(versions)}. "
            "dgp-v1 and dgp-v2 data may never be pooled for a confirmatory experiment."
        )
    return int(versions.pop())


def monotonic_anchor() -> tuple[float, float]:
    """Anchor wall-clock to a monotonic clock once per poll.

    Per-contract offsets are measured on `time.monotonic()` so that an NTP step
    mid-poll cannot make observation times appear to move backwards; the anchor
    converts them back to wall-clock for storage.
    """
    return time.time(), time.monotonic()


def observed_wall(anchor: tuple[float, float], mono: float) -> pd.Timestamp:
    wall0, mono0 = anchor
    return pd.Timestamp(wall0 + (mono - mono0), unit="s")
