"""dgp-v2 poll + write path.

Kept in its own module so `scripts/collect_market_data.py` needs only a thin
flag-gated branch, and so the dgp-v1 path it already contains stays byte-for-byte
the behaviour that produced the existing 59-day archive.

Nothing here ever writes to the dgp-v1 root. See `nifty_quant/data/dgp.py`.
"""

from __future__ import annotations

import json
import random
import subprocess
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from nifty_quant.data import dgp
from nifty_quant.log import get_logger

_log = get_logger("collect.v2")

_SHA_CACHE: dict[str, str | None] = {}


def _git_sha(path: str) -> str | None:
    """Content SHA of a source file, for provenance (§9)."""
    if path in _SHA_CACHE:
        return _SHA_CACHE[path]
    try:
        out = subprocess.run(["git", "hash-object", path], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        out = None
    _SHA_CACHE[path] = out
    return out


def poll_once_v2(provider, data_dir: str, underlying: str, expiries: list[date],
                 band_pct: float, *, shuffle: bool = True,
                 expiry_gap: float = 1.5, retries: int = 2) -> dict:
    """One dgp-v2 poll.

    Order of operations, and why each matters:

      1. spot observed BEFORE the chain fetch      -> bounds the poll's start
      2. per-expiry chain fetch, timed per batch   -> real observation times
      3. spot observed AFTER the chain fetch       -> bounds the poll's end
      4. batch-level retry only for failed batches -> successful batches keep
                                                      their true timestamps

    Under dgp-v1 (1) did not exist, (2) shared one pre-computed timestamp, and
    a retry in (4) refetched *everything* after a 4-12s backoff while still
    stamping the original time.
    """
    anchor = dgp.monotonic_anchor()
    poll_started = pd.Timestamp(anchor[0], unit="s")
    poll_id = dgp.new_poll_id(poll_started)
    order_seed = random.randrange(2 ** 31) if shuffle else None

    spot_pre, spot_pre_mono, spot_source = provider.get_spot_observed(underlying)
    if spot_pre is None:
        _log.event("poll_aborted_no_spot", level=40, poll_id=poll_id)
        return {"poll_id": poll_id, "poll_status": dgp.POLL_STATUS_FAILED,
                "rows": 0, "path": None, "spot_source": spot_source}

    band = (spot_pre * (1 - band_pct / 100.0), spot_pre * (1 + band_pct / 100.0)) \
        if band_pct > 0 else None

    all_quotes: list[tuple[int, object]] = []
    meta_by_expiry: dict[str, dict] = {}
    missing: list[str] = []

    for rank, expiry in enumerate(sorted(expiries)):
        if rank > 0:
            import time as _t
            _t.sleep(expiry_gap)
        quotes, meta = [], {}
        for attempt in range(retries):
            try:
                quotes, meta = provider.get_option_chain_timed(
                    underlying, expiry, strike_band=band,
                    shuffle_seed=(None if order_seed is None else order_seed + rank),
                    fetch_attempt=attempt)
            except Exception as exc:  # noqa: BLE001
                _log.event("chain_v2_failed", level=40, expiry=expiry.isoformat(),
                           attempt=attempt, error=str(exc)[:120])
                continue
            if not meta.get("failed_batches"):
                break
            _log.event("chain_v2_partial_batches", level=30,
                       expiry=expiry.isoformat(), attempt=attempt,
                       failed=meta["failed_batches"])
        if not quotes:
            missing.append(expiry.isoformat())
            continue
        meta_by_expiry[expiry.isoformat()] = meta
        all_quotes.extend((rank, q) for q in quotes)

    spot_post, spot_post_mono, spot_source_post = provider.get_spot_observed(underlying)

    if not all_quotes:
        _log.event("poll_failed_no_quotes", level=40, poll_id=poll_id)
        return {"poll_id": poll_id, "poll_status": dgp.POLL_STATUS_FAILED,
                "rows": 0, "path": None, "spot_source": spot_source}

    status = (dgp.POLL_STATUS_COMPLETE if not missing else dgp.POLL_STATUS_PARTIAL)
    rows = _build_rows(
        all_quotes, anchor=anchor, poll_id=poll_id, poll_started=poll_started,
        underlying=underlying, spot_pre=spot_pre, spot_pre_mono=spot_pre_mono,
        spot_post=spot_post, spot_post_mono=spot_post_mono,
        spot_source=spot_source, status=status,
        expiries_intended=[e.isoformat() for e in sorted(expiries)],
        order_seed=order_seed)

    path = write_snapshot_v2(data_dir, poll_started, rows)
    _log.event("poll_v2_written", poll_id=poll_id, rows=len(rows), status=status,
               path=str(path), missing=missing, spot_source=spot_source)
    return {"poll_id": poll_id, "poll_status": status, "rows": len(rows),
            "path": path, "spot_source": spot_source,
            "missing_expiries": missing, "meta": meta_by_expiry,
            "order_seed": order_seed}


def _build_rows(indexed_quotes, *, anchor, poll_id, poll_started, underlying,
                spot_pre, spot_pre_mono, spot_post, spot_post_mono,
                spot_source, status, expiries_intended, order_seed) -> list[dict]:
    collector_sha = _git_sha("scripts/collect_market_data.py")
    provider_sha = _git_sha("nifty_quant/data/providers/angelone.py")
    intended = ",".join(expiries_intended)
    out: list[dict] = []
    for rank, q in indexed_quotes:
        observed = (dgp.observed_wall(anchor, q.observed_ts)
                    if q.observed_ts is not None else pd.NaT)
        out.append({
            "dgp_version": dgp.DGP_V2,
            "poll_id": poll_id,
            "snapshot_id": f"{poll_id}:{rank}",
            "contract_uid": dgp.contract_uid(underlying, q.expiry, q.strike,
                                             q.option_type.value),
            "underlying": underlying,
            "expiry": pd.Timestamp(q.expiry),
            "expiry_rank": rank,
            "strike": float(q.strike),
            "option_type": q.option_type.value,
            "token": q.token,
            "trading_symbol": q.trading_symbol,
            "poll_started_ts": poll_started,
            "observed_ts": observed,
            "batch_index": q.batch_index,
            "fetch_rank": q.fetch_rank,
            "fetch_attempt": q.fetch_attempt,
            "spot_pre": spot_pre,
            "spot_pre_ts": dgp.observed_wall(anchor, spot_pre_mono),
            "spot_post": spot_post,
            "spot_post_ts": (dgp.observed_wall(anchor, spot_post_mono)
                             if spot_post is not None else pd.NaT),
            "spot_source": spot_source,
            "last_price": float(q.last_price),
            "bid": float(q.bid),
            "ask": float(q.ask),
            "volume": float(q.volume),
            "open_interest": float(q.open_interest),
            "poll_status": status,
            "expiries_intended": intended,
            "order_seed": order_seed,
            "collector_sha": collector_sha,
            "provider_sha": provider_sha,
            "context": json.dumps({}),
        })
    return out


def write_snapshot_v2(data_dir: str, poll_started, rows: list[dict]) -> Path:
    """Write one poll to the dgp-v2 root. NEVER touches the dgp-v1 root."""
    df = pd.DataFrame(rows, columns=dgp.V2_COLUMNS)
    folder = Path(dgp.snapshot_dir(data_dir, poll_started, dgp.DGP_V2))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / dgp.snapshot_filename(poll_started)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = df.drop_duplicates(subset=dgp.V2_KEYS, keep="last")
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)
    return path


def certify_poll(rows: list[dict]) -> dict:
    """Collection-time certificate for one poll (§13).

    Computed here, beside the data, rather than retrospectively in analysis
    code -- which is how dgp-v1 defects (the frozen spot feed, the dead
    oi_change column, the strike-monotonic skew) each stayed hidden until an
    audit several experiments later.
    """
    if not rows:
        return {"poll_status": dgp.POLL_STATUS_FAILED, "n_rows": 0}
    df = pd.DataFrame(rows)
    obs = pd.to_datetime(df["observed_ts"], errors="coerce")
    per_expiry_spread = (obs.groupby(df["expiry_rank"]).max()
                         - obs.groupby(df["expiry_rank"]).min()).dt.total_seconds()
    two_sided = float(((df["bid"] > 0) & (df["ask"] > 0)).mean())
    crossed = float(((df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] < df["bid"])).mean())
    return {
        "poll_id": df["poll_id"].iloc[0],
        "poll_status": df["poll_status"].iloc[0],
        "dgp_version": int(df["dgp_version"].iloc[0]),
        "n_rows": int(len(df)),
        "n_expiries": int(df["expiry_rank"].nunique()),
        "n_batches": int(df["batch_index"].nunique()),
        "measured_within_expiry_spread_s": float(per_expiry_spread.max()),
        "measured_full_poll_spread_s": float((obs.max() - obs.min()).total_seconds()),
        "spot_source": df["spot_source"].iloc[0],
        "spot_pre": float(df["spot_pre"].iloc[0]),
        "spot_post": (float(df["spot_post"].iloc[0])
                      if pd.notna(df["spot_post"].iloc[0]) else None),
        "two_sided_pct": round(100 * two_sided, 3),
        "crossed_pct": round(100 * crossed, 4),
        "order_seed": df["order_seed"].iloc[0],
        "n_retry_rows": int((df["fetch_attempt"] > 0).sum()),
    }
