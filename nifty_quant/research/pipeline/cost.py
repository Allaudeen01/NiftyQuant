"""Per-stage computational cost capture (Req 22) -- instrumentation only.

Every pipeline stage is invoked through :func:`run_with_cost`, which measures
the stage's wall-clock execution time (Req 22.1), peak memory usage, and the
number of rows it processed, packaging them into a frozen :class:`StageCost`.
The orchestrator appends one :class:`StageCost` per executed stage to the
pipeline context so the metrics are retained for the run (Req 22.2) and later
rendered in the report (Req 22.3).

These values are pure **instrumentation for human analysis**. Execution time
and peak memory are inherently non-deterministic, so -- per the determinism
strategy (Req 17, 21.4) -- they are *never* fed back into any score, verdict,
or section content. Nothing in this module participates in a deterministic
computation; it only observes a stage's resource use as a side effect of
running it.

Dataclasses mirror the existing ``nifty_quant`` style: ``from __future__ import
annotations`` plus frozen dataclasses holding only typed values.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Union


@dataclass(frozen=True)
class StageCost:
    """Computational cost of a single executed pipeline stage (Req 22.1).

    All fields are observational instrumentation and are excluded from every
    deterministic computation in the pipeline.
    """
    stage_name: str
    elapsed_seconds: float    # wall-clock execution time (Req 22.1)
    peak_memory_bytes: int    # peak allocation during the stage, via tracemalloc
    rows_processed: int       # snapshots/candles/rows reported by the stage


# How ``rows_processed`` may be supplied to :func:`run_with_cost`:
#   * an ``int``                -> used as-is,
#   * a ``Callable[[Any], int]`` -> applied to the stage result,
#   * ``None``                   -> defaults to ``0`` (rows unknown/not reported).
RowsSpec = Union[int, Callable[[Any], int], None]


def run_with_cost(
    stage_name: str,
    fn: Callable[..., Any],
    *args: Any,
    rows: RowsSpec = None,
    **kwargs: Any,
) -> tuple[Any, StageCost]:
    """Invoke ``fn`` while measuring its computational cost (Req 22).

    Runs ``fn(*args, **kwargs)`` once, measuring elapsed wall-clock time via
    :func:`time.perf_counter` and peak memory via :mod:`tracemalloc`, and
    returns ``(result, StageCost)``.

    ``rows`` derives :attr:`StageCost.rows_processed` from the stage: pass an
    ``int`` for a known count, a callable applied to the result to count rows
    after the fact, or leave it ``None`` to default to ``0``.

    The captured time/memory/rows are pure instrumentation -- the caller must
    never feed them into any deterministic score, verdict, or report content.
    """
    # tracemalloc may already be tracing (e.g. an outer profiler); only manage
    # it ourselves when it is not, and never tear down a tracer we did not start.
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    else:
        # Reset the peak so we measure this stage's allocation, not prior ones.
        tracemalloc.reset_peak()

    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed_seconds = time.perf_counter() - start
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        if started_here:
            tracemalloc.stop()

    cost = StageCost(
        stage_name=stage_name,
        elapsed_seconds=elapsed_seconds,
        peak_memory_bytes=int(peak_memory_bytes),
        rows_processed=_resolve_rows(rows, result),
    )
    return result, cost


def _resolve_rows(rows: RowsSpec, result: Any) -> int:
    """Resolve the ``rows`` spec into a concrete, non-negative row count."""
    if rows is None:
        return 0
    if callable(rows):
        return int(rows(result))
    return int(rows)
