"""Market-hours data collector: India VIX + NIFTY option-chain snapshots.

READ-ONLY. Places NO orders. Polls the live Angel One feed every ``--poll``
seconds during the regular session and writes:

  * option-chain snapshots -> data/option_chain/<YYYY>/<MM>/<DD>/<HH_MM>.parquet
    (one append-only file PER poll; India VIX is captured into each row's
    ``context``; all collected expiries for that minute share the one file)
  * an India VIX time series  -> data/vix/<YEAR>/INDIAVIX_<DATE>.parquet

Per-minute partitioning is deliberate: it is append-only (no read-modify-write
of a growing day file every poll) and crash-safe -- each snapshot is an
independent file. Use ``read_snapshots_for_day`` to load a day back into
``OptionChain`` objects for research.

WHY THIS EXISTS
---------------
Research (Exp 001-009) closed the price-only directional branch: no strategy
beats buy & hold after costs, and intraday returns are unpredictable. The only
forecastable dimension is volatility. The next experiments (variance risk
premium, dealer gamma, OI migration) need implied / positioning data we do NOT
have -- and intraday option snapshots CANNOT be backfilled. So the highest-value
action is to start recording them now.

SETUP (once, this weekend)
--------------------------
  1. Put Angel One creds in .env:
       ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_MPIN (or ANGEL_PASSWORD),
       ANGEL_TOTP_SECRET
  2. pip install smartapi-python pyotp
  3. Dry run a single poll (sandbox; writes to data_test/, ignores hours):
       python scripts/collect_market_data.py --once --ignore-market-hours --test

MONDAY (and every trading day)
------------------------------
  Start before 09:15 and leave running; Ctrl+C to stop after 15:30:
       python scripts/collect_market_data.py --num-expiries 2 --strike-band-pct 10 --poll 60

This NEVER trades. It only reads market data and writes parquet files.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nifty_quant.data.models import OptionChain, OptionQuote, OptionType
from nifty_quant.dotenv import load_dotenv
from nifty_quant.log import get_logger

_log = get_logger("scripts.collect_market_data")

IST = timezone(timedelta(hours=5, minutes=30))
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)

# Known NSE trading holidays (equity segment). VERIFY/UPDATE against the
# official NSE calendar each year -- an out-of-date entry could wrongly block a
# real trading day. Weekends are always treated as closed regardless of this set.
NSE_HOLIDAYS: set[date] = {
    # --- 2026 (best-effort; confirm on the official NSE list) ---
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 6),    # Holi
    date(2026, 3, 21),   # (placeholder/verify)
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day (Sat)
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 11, 9),   # Diwali (verify; muhurat session separate)
    date(2026, 12, 25),  # Christmas
}

_STOP = False


def _handle_sigint(signum, frame):  # graceful Ctrl+C
    global _STOP
    _STOP = True
    print("\n[collector] stop requested; finishing current poll...")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect India VIX + option chain (read-only).")
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--expiries", default=None,
                   help="Comma-separated YYYY-MM-DD list. Overrides --num-expiries.")
    p.add_argument("--num-expiries", type=int, default=2,
                   help="Auto-pick the nearest N expiries from the scrip master.")
    p.add_argument("--strike-band-pct", type=float, default=10.0,
                   help="Keep strikes within +/- this %% of spot (0 = all).")
    p.add_argument("--poll", type=float, default=60.0, help="Seconds between polls.")
    p.add_argument("--request-pause", type=float, default=1.1,
                   help="Seconds between underlying API requests (raise to avoid "
                        "Angel rate-limit errors).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--test", "--sandbox", dest="test", action="store_true",
                   help="Sandbox mode: write to data_test/ (never the production "
                        "data/ warehouse).")
    p.add_argument("--once", action="store_true", help="Single poll then exit (dry run).")
    p.add_argument("--ignore-market-hours", action="store_true",
                   help="Poll regardless of session window (for testing).")
    return p.parse_args()


def now_ist() -> datetime:
    return datetime.now(IST)


def market_closed_reason(d: date) -> str | None:
    """Return a reason string if the market is closed on ``d``, else None."""
    if d.weekday() == 5:
        return "Saturday"
    if d.weekday() == 6:
        return "Sunday"
    if d in NSE_HOLIDAYS:
        return "NSE holiday"
    return None


def in_session(dt: datetime) -> bool:
    if dt.weekday() >= 5:  # Sat/Sun
        return False
    return SESSION_OPEN <= dt.time() <= SESSION_CLOSE


def pick_expiries(master, underlying: str, n: int) -> list[date]:
    today = now_ist().date()
    future = [e for e in master.available_expiries(underlying) if e >= today]
    return future[:n]


def trim_to_band(chain: OptionChain, band_pct: float, india_vix: float | None) -> OptionChain:
    """Return a new chain with strikes within +/- band_pct of spot and VIX in context."""
    quotes = chain.quotes
    if band_pct > 0 and chain.spot > 0:
        band = chain.spot * band_pct / 100.0
        quotes = tuple(q for q in chain.quotes if abs(q.strike - chain.spot) <= band)
    context = dict(chain.context)
    if india_vix is not None:
        context["india_vix"] = india_vix
    return OptionChain(
        underlying=chain.underlying,
        spot=chain.spot,
        expiry=chain.expiry,
        timestamp=chain.timestamp,
        quotes=quotes,
        context=context,
    )


def write_vix(data_dir: str, ts: datetime, value: float) -> None:
    """Append one India VIX reading to data/vix/<YEAR>/INDIAVIX_<DATE>.parquet."""
    ts = pd.Timestamp(ts)
    path = Path(data_dir) / "vix" / str(ts.year) / f"INDIAVIX_{ts.date().isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"timestamp": ts, "india_vix": float(value)}])
    if path.exists():
        existing = pd.read_parquet(path)
        row = pd.concat([existing, row], ignore_index=True)
    row = row.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    tmp = path.with_suffix(".parquet.tmp")
    row.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)


# --- per-minute partitioned option-chain snapshots -------------------------

_SNAP_COLUMNS = [
    "snapshot_ts", "underlying", "spot", "expiry", "strike", "option_type",
    "last_price", "bid", "ask", "volume", "open_interest", "oi_change",
    "implied_volatility", "context",
]
_SNAP_KEYS = ["snapshot_ts", "underlying", "expiry", "strike", "option_type"]


def _chain_rows(chain: OptionChain) -> list[dict]:
    ctx = json.dumps(chain.context, default=str)
    return [
        {
            "snapshot_ts": pd.Timestamp(chain.timestamp),
            "underlying": chain.underlying,
            "spot": chain.spot,
            "expiry": pd.Timestamp(chain.expiry),
            "strike": q.strike,
            "option_type": q.option_type.value,
            "last_price": q.last_price,
            "bid": q.bid,
            "ask": q.ask,
            "volume": q.volume,
            "open_interest": q.open_interest,
            "oi_change": q.oi_change,
            "implied_volatility": q.implied_volatility,
            "context": ctx,
        }
        for q in chain.quotes
    ]


def snapshot_path(data_dir: str, ts: datetime) -> Path:
    t = pd.Timestamp(ts)
    return (Path(data_dir) / "option_chain" / f"{t.year:04d}" / f"{t.month:02d}"
            / f"{t.day:02d}" / f"{t.hour:02d}_{t.minute:02d}.parquet")


def write_snapshot(data_dir: str, ts: datetime, chains: list[OptionChain]) -> Path:
    """Write all chains for one poll into data/option_chain/YYYY/MM/DD/HH_MM.parquet.

    Append-only per minute: if the file exists (sub-minute polling), rows are
    merged and de-duplicated on (snapshot_ts, underlying, expiry, strike, type).
    """
    rows: list[dict] = []
    for ch in chains:
        rows.extend(_chain_rows(ch))
    df = pd.DataFrame(rows, columns=_SNAP_COLUMNS)
    path = snapshot_path(data_dir, ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = df.drop_duplicates(subset=_SNAP_KEYS, keep="last")
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)
    return path


def read_snapshots_for_day(data_dir: str, day: date) -> list[OptionChain]:
    """Load all per-minute snapshots for a day back into OptionChain objects."""
    folder = (Path(data_dir) / "option_chain" / f"{day.year:04d}"
              / f"{day.month:02d}" / f"{day.day:02d}")
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    chains: list[OptionChain] = []
    for (ts, und, exp), g in df.groupby(["snapshot_ts", "underlying", "expiry"]):
        quotes = [
            OptionQuote(
                strike=float(r.strike), option_type=OptionType(r.option_type),
                expiry=pd.Timestamp(r.expiry).date(), last_price=float(r.last_price),
                bid=float(r.bid), ask=float(r.ask), volume=float(r.volume),
                open_interest=float(r.open_interest), oi_change=float(r.oi_change),
                implied_volatility=(None if pd.isna(r.implied_volatility)
                                    else float(r.implied_volatility)),
            )
            for r in g.itertuples(index=False)
        ]
        ctx = {}
        first = g.iloc[0].get("context")
        if isinstance(first, str):
            try:
                ctx = json.loads(first)
            except (ValueError, TypeError):
                ctx = {}
        chains.append(OptionChain(
            underlying=str(und), spot=float(g.iloc[0]["spot"]),
            expiry=pd.Timestamp(exp).date(), timestamp=pd.Timestamp(ts).to_pydatetime(),
            quotes=tuple(quotes), context=ctx,
        ))
    return chains


def _is_retryable(exc: Exception) -> bool:
    """Transient errors worth retrying: rate limits and network blips."""
    s = str(exc).lower()
    rate = "access rate" in s or "access denied" in s or ("rate" in s and "exceed" in s)
    network = ("timed out" in s or "timeout" in s or "max retries" in s
               or "connection" in s or "temporarily unavailable" in s)
    return rate or network


def fetch_chain_with_retry(provider, underlying, expiry, *, retries=3, backoff=4.0):
    """Fetch one expiry's chain, retrying on Angel rate-limit or network errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            return provider.get_option_chain(underlying, expiry)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_retryable(exc) and attempt < retries - 1:
                wait = backoff * (attempt + 1)
                _log.event("chain_retry", level=30, expiry=expiry.isoformat(),
                           attempt=attempt + 1, wait_s=wait, error=str(exc)[:80])
                time.sleep(wait)
                continue
            raise
    raise last_exc  # pragma: no cover
    raise last_exc  # pragma: no cover


def poll_once(provider, data_dir, underlying, expiries, band_pct, expiry_gap=1.5) -> None:
    ts = now_ist()
    vix = None
    try:
        vix = provider.get_india_vix()
        if vix is not None:
            write_vix(data_dir, ts, vix)
    except Exception as exc:  # noqa: BLE001 - keep collecting other data
        _log.event("vix_fetch_failed", level=30, error=str(exc))

    chains: list[OptionChain] = []
    for i, expiry in enumerate(expiries):
        if i > 0:
            time.sleep(expiry_gap)  # space requests under Angel's rate limit
        try:
            chain = fetch_chain_with_retry(provider, underlying, expiry)
            chains.append(trim_to_band(chain, band_pct, vix))
        except Exception as exc:  # noqa: BLE001
            _log.event("chain_fetch_failed", level=40, expiry=expiry.isoformat(),
                       error=str(exc))
            print(f"[{ts:%H:%M:%S}] chain fetch FAILED for {expiry}: {exc}")

    if chains:
        path = write_snapshot(data_dir, ts, chains)
        total_q = sum(len(c.quotes) for c in chains)
        complete = len(chains) == len(expiries)
        _log.event("snapshot_written", underlying=underlying,
                   expiries=[e.isoformat() for e in expiries], quotes=total_q,
                   complete=complete, india_vix=vix, path=str(path))
        spots = ", ".join(f"{c.expiry}:{c.spot:.0f}" for c in chains)
        flag = "" if complete else "  [PARTIAL]"
        print(f"[{ts:%H:%M:%S}] {underlying} vix={vix} | {total_q} quotes "
              f"-> {path.name}  ({spots}){flag}")


def main() -> int:
    args = parse_args()
    load_dotenv()
    signal.signal(signal.SIGINT, _handle_sigint)

    # Sandbox isolation: in test mode, never touch the production warehouse.
    if args.test:
        args.data_dir = "data_test"
        print("TEST MODE - No production data will be written.")

    # Startup safety: refuse to run on a closed market unless explicitly in
    # test/sandbox mode or overriding the session window.
    if not args.test and not args.ignore_market_hours:
        reason = market_closed_reason(now_ist().date())
        if reason:
            print(f"Market is closed ({reason}). Collector exited without "
                  f"writing any production data.")
            return 0

    try:
        from nifty_quant.data.providers.angelone import AngelOneProvider
        provider = AngelOneProvider.from_env()   # live_trading_enabled stays False
        provider.request_pause = args.request_pause  # throttle to respect rate limits
    except Exception as exc:  # noqa: BLE001
        print(f"[collector] could not start Angel provider: {exc}")
        print("  - check .env creds and `pip install smartapi-python pyotp`")
        return 2

    master = provider._get_instrument_master()

    if args.expiries:
        expiries = [date.fromisoformat(s.strip()) for s in args.expiries.split(",")]
    else:
        expiries = pick_expiries(master, args.underlying, args.num_expiries)
    if not expiries:
        print(f"[collector] no upcoming expiries found for {args.underlying}")
        return 2

    print("=" * 80)
    print(f"MARKET DATA COLLECTOR (read-only)  {args.underlying}")
    print(f"Expiries: {[e.isoformat() for e in expiries]}  "
          f"strike-band=+/-{args.strike_band_pct}%  poll={args.poll}s")
    print(f"Writing: {args.data_dir}/option_chain/...  and  {args.data_dir}/vix/...")
    print("NO ORDERS are placed. Ctrl+C to stop.")
    print("=" * 80)

    polls = 0
    while not _STOP:
        dt = now_ist()
        if args.ignore_market_hours or in_session(dt):
            poll_once(provider, args.data_dir, args.underlying,
                      expiries, args.strike_band_pct)
            polls += 1
        else:
            # Outside session: idle-log occasionally, exit after close on weekdays.
            if dt.weekday() < 5 and dt.time() > SESSION_CLOSE:
                print(f"[{dt:%H:%M:%S}] session closed; {polls} polls collected. Exiting.")
                break
            print(f"[{dt:%H:%M:%S} IST] outside session window; waiting...")

        if args.once:
            print(f"[collector] --once complete ({polls} poll). Exiting.")
            break
        # Sleep in short increments so Ctrl+C is responsive.
        slept = 0.0
        while slept < args.poll and not _STOP:
            time.sleep(min(1.0, args.poll - slept))
            slept += 1.0

    print(f"[collector] stopped after {polls} polls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
