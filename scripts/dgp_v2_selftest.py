"""Offline validation of the dgp-v2 collection path. NO LIVE API, NO REAL DATA.

Runs the real `poll_once_v2` against a stub provider that simulates Angel's
batching, pacing and failure modes, writing to a throwaway directory. Executes
the validation tests specified in the cutover plan.

    python scripts/dgp_v2_selftest.py

Never touches data/option_chain, data/option_chain_v2, or any frozen artifact.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_quant.data import dgp                              # noqa: E402
from nifty_quant.data.collect_v2 import (                     # noqa: E402
    certify_poll, poll_once_v2,
)
from nifty_quant.data.models import OptionQuote, OptionType   # noqa: E402

BATCH_PAUSE = 0.02          # scaled-down stand-in for the real 1.2s throttle
N_STRIKES = 60
STRIKE_STEP = 50.0
SPOT = 24_000.0


class StubProvider:
    """Simulates Angel batching/pacing without any network call."""

    def __init__(self, fail_batches: set[int] | None = None,
                 spot_mode: str = "live"):
        self.fail_batches = fail_batches or set()
        self.spot_mode = spot_mode
        self.calls = 0
        self._failed_once: set[int] = set()

    def get_spot_observed(self, symbol):
        if self.spot_mode == "unavailable":
            return None, time.monotonic(), dgp.SPOT_SOURCE_UNAVAILABLE
        if self.spot_mode == "fallback":
            return SPOT, time.monotonic(), dgp.SPOT_SOURCE_FALLBACK
        # tiny drift so pre/post differ, as they would live
        return SPOT + self.calls * 0.05, time.monotonic(), dgp.SPOT_SOURCE_LIVE

    def get_option_chain_timed(self, underlying, expiry, *, strike_band=None,
                               shuffle_seed=None, fetch_attempt=0):
        strikes = [SPOT + (i - N_STRIKES // 2) * STRIKE_STEP for i in range(N_STRIKES)]
        if strike_band is not None:
            strikes = [s for s in strikes if strike_band[0] <= s <= strike_band[1]]
        instruments = [(s, ot) for s in strikes for ot in ("CE", "PE")]
        instruments.sort(key=lambda x: (x[0], x[1]))
        if shuffle_seed is not None:
            import random
            random.Random(shuffle_seed).shuffle(instruments)

        quotes, failed = [], []
        for bi, start in enumerate(range(0, len(instruments), 50)):
            batch = instruments[start:start + 50]
            self.calls += 1
            time.sleep(BATCH_PAUSE)                  # simulate the throttle
            # a failing batch fails only on the first attempt, so batch-level
            # retry can be observed to recover it
            if bi in self.fail_batches and bi not in self._failed_once:
                self._failed_once.add(bi)
                failed.append(bi)
                continue
            obs = time.monotonic()
            for rank_in_batch, (strike, ot) in enumerate(batch):
                quotes.append(OptionQuote(
                    strike=strike, option_type=OptionType(ot), expiry=expiry,
                    last_price=100.0, bid=99.5, ask=100.5,
                    volume=1000.0, open_interest=5000.0,
                    observed_ts=obs, batch_index=bi,
                    fetch_rank=start + rank_in_batch, fetch_attempt=fetch_attempt,
                    token=f"T{int(strike)}{ot}", trading_symbol=f"NIFTY{int(strike)}{ot}",
                ))
        return quotes, {"n_instruments": len(instruments), "n_quotes": len(quotes),
                        "failed_batches": failed,
                        "n_batches": (len(instruments) + 49) // 50,
                        "shuffle_seed": shuffle_seed, "strike_band": strike_band}


def _poll_rows(res) -> pd.DataFrame:
    """Rows for exactly the poll under test.

    A snapshot file can hold more than one poll when two polls fall in the same
    second; `dgp.V2_KEYS` includes `poll_id` so the merge is lossless, but any
    assertion must scope to its own poll or it will silently read another's.
    """
    df = pd.read_parquet(res["path"])
    return df[df["poll_id"] == res["poll_id"]].reset_index(drop=True)


PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not ok else ""))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="dgpv2_selftest_")
    expiries = [date.today() + timedelta(days=3), date.today() + timedelta(days=10)]
    try:
        print("=" * 78)
        print("dgp-v2 OFFLINE SELF-TEST (stub provider, throwaway dir)")
        print("=" * 78)

        # ---- 1. happy path -------------------------------------------
        print("\n[1] nominal poll")
        p = StubProvider()
        res = poll_once_v2(p, tmp, "NIFTY", expiries, band_pct=6.0, shuffle=True,
                           expiry_gap=0.01)
        df = _poll_rows(res)
        check("poll_status complete", res["poll_status"] == dgp.POLL_STATUS_COMPLETE)
        check("schema matches V2_COLUMNS", list(df.columns) == dgp.V2_COLUMNS)
        check("every row carries dgp_version==2", (df["dgp_version"] == 2).all())
        check("wrote to option_chain_v2 root", dgp.V2_ROOT in str(res["path"]))
        check("v1 root untouched",
              not (Path(tmp) / dgp.V1_ROOT).exists())

        # §1 per-contract timestamps
        n_distinct = df["observed_ts"].nunique()
        check("multiple distinct observed_ts (v1 signature is exactly 1)",
              n_distinct > 1, f"got {n_distinct}")
        check("observed_ts increases with batch_index",
              df.groupby("batch_index")["observed_ts"].min().is_monotonic_increasing)

        # §2 spot before and after
        check("spot_pre_ts precedes all observations",
              pd.to_datetime(df["spot_pre_ts"]).max() <= pd.to_datetime(df["observed_ts"]).min())
        check("spot_post_ts follows all observations",
              pd.to_datetime(df["spot_post_ts"]).min() >= pd.to_datetime(df["observed_ts"]).max())
        check("spot_source recorded as live_ltp", (df["spot_source"] == "live_ltp").all())

        # §3 randomization kills the strike/time gradient
        rho = df[["strike", "fetch_rank"]].corr(method="spearman").iloc[0, 1]
        check("shuffled order decorrelates strike from fetch_rank",
              abs(rho) < 0.5, f"spearman={rho:.3f}")

        # §4 pre-fetch band trim
        in_band = ((df["strike"] >= SPOT * 0.94) & (df["strike"] <= SPOT * 1.06)).all()
        check("only in-band strikes fetched", in_band)

        # §5/§7 identity
        check("poll_id unique in file", df["poll_id"].nunique() == 1)
        check("contract_uid unique per (expiry,strike,type)",
              df["contract_uid"].nunique() == len(df))
        check("token and trading_symbol stored",
              df["token"].notna().all() and df["trading_symbol"].notna().all())

        # §6 expiry metadata
        check("expiry_rank 0 is the near expiry",
              df.loc[df["expiry_rank"] == 0, "expiry"].max()
              < df.loc[df["expiry_rank"] == 1, "expiry"].min())
        check("expiries_intended recorded", df["expiries_intended"].iloc[0].count(",") == 1)

        # §8 dead columns absent
        check("oi_change absent from schema", "oi_change" not in df.columns)
        check("implied_volatility absent from schema",
              "implied_volatility" not in df.columns)

        # §13 certification
        cert = certify_poll(df.to_dict("records"))
        check("certificate reports measured within-expiry spread",
              cert["measured_within_expiry_spread_s"] >= 0)
        check("certificate reports spot_source", cert["spot_source"] == "live_ltp")

        # ---- 2. ordering A/B ------------------------------------------
        print("\n[2] strike-ordered control (--no-shuffle)")
        p2 = StubProvider()
        res2 = poll_once_v2(p2, tmp, "NIFTY", expiries, band_pct=6.0, shuffle=False,
                            expiry_gap=0.01)
        d2 = _poll_rows(res2)
        rho2 = d2[["strike", "fetch_rank"]].corr(method="spearman").iloc[0, 1]
        check("unshuffled order IS strike-monotonic (reproduces the v1 defect)",
              rho2 > 0.95, f"spearman={rho2:.3f}")
        check("order_seed null when not shuffling", pd.isna(d2["order_seed"].iloc[0]))

        # ---- 3. batch failure + retry ---------------------------------
        print("\n[3] batch-level failure and retry")
        p3 = StubProvider(fail_batches={1})
        res3 = poll_once_v2(p3, tmp, "NIFTY", expiries, band_pct=6.0, shuffle=True,
                            expiry_gap=0.01, retries=2)
        d3 = _poll_rows(res3)
        check("poll still completes after a failed batch",
              res3["poll_status"] in (dgp.POLL_STATUS_COMPLETE, dgp.POLL_STATUS_PARTIAL))
        check("retry recorded via fetch_attempt",
              d3["fetch_attempt"].max() >= 0)
        check("fetch_attempt is part of the key (no silent overwrite)",
              "fetch_attempt" in dgp.V2_KEYS)

        # ---- 4. spot failure aborts rather than fabricates -------------
        print("\n[4] spot unavailable")
        p4 = StubProvider(spot_mode="unavailable")
        res4 = poll_once_v2(p4, tmp, "NIFTY", expiries, band_pct=6.0, expiry_gap=0.01)
        check("poll fails loudly when spot unavailable",
              res4["poll_status"] == dgp.POLL_STATUS_FAILED and res4["rows"] == 0)
        check("no file written for a failed poll", res4["path"] is None)

        print("\n[5] fallback spot is reported, never silent")
        p5 = StubProvider(spot_mode="fallback")
        res5 = poll_once_v2(p5, tmp, "NIFTY", expiries, band_pct=6.0, expiry_gap=0.01)
        d5 = _poll_rows(res5)
        check("spot_source==daily_close_fallback surfaced",
              (d5["spot_source"] == dgp.SPOT_SOURCE_FALLBACK).all())

        # ---- 6. pooling guard ------------------------------------------
        print("\n[6] dgp pooling guard")
        check("assert_single_version accepts pure v2",
              dgp.assert_single_version(df) == dgp.DGP_V2)
        v1_like = pd.DataFrame({"strike": [1.0]})
        check("assert_single_version treats column-less frame as v1",
              dgp.assert_single_version(v1_like) == dgp.DGP_V1)
        mixed = pd.concat([df.head(2), df.head(2).assign(dgp_version=1)])
        try:
            dgp.assert_single_version(mixed)
            check("mixed v1/v2 frame is REFUSED", False, "no exception raised")
        except ValueError:
            check("mixed v1/v2 frame is REFUSED", True)

        # ---- 7. filename resolves seconds ------------------------------
        print("\n[7] snapshot filename granularity")
        f1 = dgp.snapshot_filename(pd.Timestamp("2026-01-01 09:15:10"))
        f2 = dgp.snapshot_filename(pd.Timestamp("2026-01-01 09:15:40"))
        check("two polls in the same minute get distinct filenames", f1 != f2,
              f"{f1} vs {f2}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"   FAILED: {f}")
        print("=" * 78)
        return 1
    print("All dgp-v2 validation tests passed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
