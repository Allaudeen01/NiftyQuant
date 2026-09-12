"""NiftyQuant dashboard — visualization layer. READ-ONLY.

    streamlit run dashboard/app.py

This dashboard never writes to `data/`, never modifies any experiment artifact,
and is strictly read-only with respect to the frozen experiments EXP032,
EXP033 and EXP034-A.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="NiftyQuant", page_icon="📊", layout="wide")

st.title("NiftyQuant")
st.caption("Read-only visualization layer. Frozen experiments are never modified.")

st.markdown(
    """
### Pages

**Data Collection Diagnostics** — live collection timing, within-snapshot
synchronization, sequential-skew analysis and daily collection-quality scoring.

---

#### Research status

| Experiment | Status |
|---|---|
| EXP032 | FROZEN — `exp032_final` |
| EXP033 | FROZEN — `EXP033_FINAL_FROZEN` |
| EXP034-A | FROZEN — `EXP034_A_FINAL_FROZEN` |
| EXP035 | DESIGN TERMINATED BEFORE PREREGISTRATION |

EXP035 was stopped at the design-audit stage. **No outcome was ever inspected,
no model fitted, no backtest run.** It is not a failed trading strategy — see
`docs/exp035_design_archive.md`.
"""
)

st.info(
    "**Per-contract observation timestamps do not exist under the current "
    "data-generating process (`dgp-v1`).** Every row of a poll shares one "
    "`snapshot_ts`. The diagnostics page therefore labels each quantity "
    "MEASURED, MODELLED or UNAVAILABLE, and never presents a modelled value as "
    "an observed one. See `docs/data_collection_redesign.md`.",
    icon="ℹ️",
)
