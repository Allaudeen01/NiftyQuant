"""Angel One instrument master (scrip master) loader.

Angel publishes a master JSON of every tradable instrument with its numeric
token. There is no option-chain endpoint, so to build a chain we look up the
option contracts (CE/PE per strike) for an underlying+expiry here, then fetch
their live data via ``getMarketData``.

Scrip-master record shape (example index entry):
    {"token":"99926000","symbol":"Nifty 50","name":"NIFTY","expiry":"",
     "strike":"0.000000","lotsize":"1","instrumenttype":"AMXIDX",
     "exch_seg":"NSE","tick_size":"0.000000"}

For index options: instrumenttype="OPTIDX", exch_seg="NFO", name="NIFTY",
expiry like "27MAR2025", strike in paise (2500000 == 25000.0), option type is
the CE/PE suffix of the trading symbol.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from nifty_quant.data.models import OptionType
from nifty_quant.log import get_logger

_log = get_logger("providers.angel_instruments")

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/"
    "OpenAPIScripMaster.json"
)


@dataclass(frozen=True)
class OptionInstrument:
    token: str
    trading_symbol: str
    strike: float
    option_type: OptionType
    expiry: date


class InstrumentMaster:
    """Loads and queries the Angel scrip master (with on-disk caching)."""

    def __init__(
        self,
        cache_path: str | Path = "data/angel_scrip_master.json",
        *,
        url: str = SCRIP_MASTER_URL,
        max_age_hours: float = 12.0,
        records: list[dict] | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.url = url
        self.max_age_hours = max_age_hours
        self._records: list[dict] | None = records

    # --- construction / loading --------------------------------------------

    @classmethod
    def from_records(cls, records: list[dict]) -> "InstrumentMaster":
        """Build directly from in-memory records (used in tests)."""
        return cls(records=records)

    def _ensure_loaded(self) -> list[dict]:
        if self._records is not None:
            return self._records
        if self._cache_is_fresh():
            self._records = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self._records = self._download_and_cache()
        return self._records

    def _cache_is_fresh(self) -> bool:
        if not self.cache_path.exists():
            return False
        age_h = (time.time() - self.cache_path.stat().st_mtime) / 3600.0
        return age_h <= self.max_age_hours

    def _download_and_cache(self) -> list[dict]:
        _log.event("scrip_master_download", url=self.url)
        with urllib.request.urlopen(self.url, timeout=60) as resp:  # noqa: S310
            data = resp.read()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(data)
        return json.loads(data.decode("utf-8"))

    # --- queries ------------------------------------------------------------

    def option_instruments(
        self,
        name: str,
        expiry: date,
        *,
        exch_seg: str = "NFO",
        instrumenttype: str = "OPTIDX",
    ) -> list[OptionInstrument]:
        """All CE/PE option contracts for an underlying and expiry."""
        name = name.upper()
        out: list[OptionInstrument] = []
        for rec in self._ensure_loaded():
            if (
                rec.get("name", "").upper() != name
                or rec.get("instrumenttype") != instrumenttype
                or rec.get("exch_seg") != exch_seg
            ):
                continue
            rec_expiry = _parse_expiry(rec.get("expiry", ""))
            if rec_expiry != expiry:
                continue
            symbol = rec.get("symbol", "")
            otype = _option_type_from_symbol(symbol)
            if otype is None:
                continue
            out.append(OptionInstrument(
                token=str(rec["token"]),
                trading_symbol=symbol,
                strike=float(rec["strike"]) / 100.0,  # paise -> rupees
                option_type=otype,
                expiry=expiry,
            ))
        return sorted(out, key=lambda o: (o.strike, o.option_type.value))

    def available_expiries(
        self, name: str, *, exch_seg: str = "NFO", instrumenttype: str = "OPTIDX"
    ) -> list[date]:
        name = name.upper()
        expiries: set[date] = set()
        for rec in self._ensure_loaded():
            if (
                rec.get("name", "").upper() == name
                and rec.get("instrumenttype") == instrumenttype
                and rec.get("exch_seg") == exch_seg
            ):
                d = _parse_expiry(rec.get("expiry", ""))
                if d:
                    expiries.add(d)
        return sorted(expiries)

    def index_token(self, name: str, *, exch_seg: str = "NSE") -> str | None:
        """Resolve an index token by name (e.g. 'INDIA VIX', 'NIFTY')."""
        name = name.upper()
        for rec in self._ensure_loaded():
            if (
                rec.get("exch_seg") == exch_seg
                and rec.get("instrumenttype") in ("AMXIDX", "", None)
                and (rec.get("name", "").upper() == name
                     or rec.get("symbol", "").upper() == name)
            ):
                return str(rec["token"])
        return None


def _parse_expiry(value: str) -> date | None:
    if not value:
        return None
    v = value.strip()
    for fmt in ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d"):
        for candidate in (v, v.upper(), v.title()):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    return None


def _option_type_from_symbol(symbol: str) -> OptionType | None:
    s = symbol.upper()
    if s.endswith("CE"):
        return OptionType.CALL
    if s.endswith("PE"):
        return OptionType.PUT
    return None
