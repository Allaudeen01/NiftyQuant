"""SessionCatalog -- collected-session enumeration over the frozen Storage.

The catalog answers "which sessions has the warehouse actually collected?" so
the comparison stage can compute the ``Collected_Session_Count`` and build its
historical window from only the Prior_Sessions that truly exist (Req 5.1, 5.2)
-- option-chain history is forward-collected and cannot be backfilled, so the
catalog never assumes any session it has not seen.

It is implemented **entirely over the existing ``Storage`` interface** and
introduces no new backend (Req 2.5, 16.1): a single
``Storage.read_option_chains`` call over a **bounded** discovery window is read
once, cached, and grouped by ``snapshot_ts.date()`` (the real ``OptionChain``
field is ``timestamp``; ``timestamp.date()`` is the session date). Every query
is then served from that cached grouping, so repeated calls never re-read the
warehouse.

The discovery window is bounded on purpose. ``Storage.read_option_chains``
iterates one filesystem ``stat`` per day in ``[start, end]`` (the
``ParquetStorage`` implementation walks ``_days_between`` and probes each day's
path), so an unbounded epoch..far-future span would issue millions of probes and
hang the pipeline. We therefore scan a fixed, **deterministic** window
(``earliest``..``latest``, defaulting to 2000-01-01..2100-01-01, ~36,500 days)
that is fast yet generous enough to capture every realistic collected session.
The bounds are deterministic by design (no ``date.today()`` / wall-clock value,
Req 17) and are overridable via ``SessionCatalog.__init__`` if a caller needs a
wider or narrower window.

Besides the per-target session queries the catalog exposes the two inventory
counts the ROADMAP control center renders (Req 18.3): the number of collected
option-chain snapshots and the number of collected trading days.

This module mirrors the existing ``nifty_quant`` style (``from __future__
import annotations`` + a small, explicit class with no hidden global state).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycle
    from nifty_quant.data.storage.base import Storage


# Default bounds for the single warehouse scan. ``read_option_chains`` reads
# ``[start, end]`` inclusive and -- in the ParquetStorage implementation --
# issues one filesystem ``stat`` per day in that span (it walks ``_days_between``
# and probes each day's path). An unbounded epoch..far-future scan would
# therefore make millions of probes and hang the pipeline, so we bound the
# window to a fixed, deterministic range that is fast yet generous enough to
# cover every realistic collected session (including 2024-2026 data). These are
# deterministic by design (no ``date.today()`` / wall-clock value, Req 17) and
# overridable via ``SessionCatalog.__init__``.
_DEFAULT_EARLIEST = date(2000, 1, 1)
_DEFAULT_LATEST = date(2100, 1, 1)


class SessionCatalog:
    """Enumerate collected sessions by grouping option chains by snapshot date.

    The catalog reads the underlying's full option-chain history once (lazily,
    on first query) through ``Storage.read_option_chains`` and caches the count
    of snapshots collected on each session date. All queries are pure reads of
    that cache, so constructing the catalog is cheap and repeated queries do not
    touch the warehouse again.
    """

    def __init__(
        self,
        storage: "Storage",
        underlying: str,
        timeframe: str,
        earliest: date = _DEFAULT_EARLIEST,
        latest: date = _DEFAULT_LATEST,
    ) -> None:
        self._storage = storage
        self._underlying = underlying
        # Retained for parity with the other stages' constructor signatures; the
        # catalog enumerates sessions from option chains only (Req 2.5/16.1).
        self._timeframe = timeframe
        # Bounded, deterministic discovery window for the single warehouse scan
        # (see module docstring -- keeps the per-day stat cost finite, Req 17).
        self._earliest = earliest
        self._latest = latest
        # Cache: session date -> number of option-chain snapshots that day.
        # ``None`` until the single warehouse read has been performed.
        self._counts_by_date: dict[date, int] | None = None

    # --- Warehouse read (performed once, then cached) ------------------------

    def _counts(self) -> dict[date, int]:
        """Return the cached ``date -> snapshot count`` map, reading once.

        The single ``read_option_chains`` call uses only the existing ``Storage``
        interface (no new backend, Req 16.1) and is grouped by
        ``OptionChain.timestamp.date()`` -- the session date. The scan is bounded
        to the deterministic ``[earliest, latest]`` window so the per-day stat
        cost stays finite (see module docstring).
        """
        if self._counts_by_date is None:
            chains = self._storage.read_option_chains(
                self._underlying,
                datetime.combine(self._earliest, datetime.min.time()),
                datetime.combine(self._latest, datetime.max.time()),
            )
            counts: dict[date, int] = {}
            for chain in chains:
                session_date = chain.timestamp.date()
                counts[session_date] = counts.get(session_date, 0) + 1
            self._counts_by_date = counts
        return self._counts_by_date

    def refresh(self) -> None:
        """Drop the cache so the next query re-reads the warehouse.

        Useful when new snapshots have been collected after the catalog was
        constructed; ordinary use never needs to call this.
        """
        self._counts_by_date = None

    # --- Per-target session queries (Req 5.1, 5.2) ---------------------------

    def sessions_up_to(self, target: date) -> list[date]:
        """Distinct collected session dates ``<= target``, ascending (Req 5.1)."""
        return sorted(d for d in self._counts() if d <= target)

    def collected_count(self, target: date) -> int:
        """``Collected_Session_Count``: collected sessions at or before ``target``.

        Counts only the sessions actually present in the warehouse, so it never
        assumes any backfilled history (Req 5.1, 5.2).
        """
        return sum(1 for d in self._counts() if d <= target)

    def prior_sessions(self, target: date) -> list[date]:
        """Collected Prior_Sessions: distinct collected dates strictly before
        ``target``, ascending (Req 5.2)."""
        return sorted(d for d in self._counts() if d < target)

    # --- Inventory counts for the ROADMAP control center (Req 18.3) ----------

    def option_chain_count(self, up_to: date | None = None) -> int:
        """Collected Option Chains count: total snapshots collected.

        Counts every option-chain snapshot in the warehouse, or only those at or
        before ``up_to`` when a bound is supplied (Req 18.3).
        """
        counts = self._counts()
        if up_to is None:
            return sum(counts.values())
        return sum(n for d, n in counts.items() if d <= up_to)

    def trading_day_count(self, up_to: date | None = None) -> int:
        """Collected Trading Days count: distinct session dates collected.

        A trading day counts once regardless of how many snapshots were taken
        that day; bounded by ``up_to`` when supplied (Req 18.3).
        """
        counts = self._counts()
        if up_to is None:
            return len(counts)
        return sum(1 for d in counts if d <= up_to)
