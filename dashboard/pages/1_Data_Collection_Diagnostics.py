"""Data Collection Diagnostics — live collection timing. READ-ONLY."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collection_metrics as cm  # noqa: E402

st.set_page_config(page_title="Data Collection Diagnostics", page_icon="🧪", layout="wide")
st.title("Data Collection Diagnostics")

MEASURED = ":green[MEASURED]"
MODELLED = ":orange[MODELLED]"
UNAVAIL = ":red[UNAVAILABLE — requires dgp-v2]"


@st.cache_data(show_spinner=False)
def _days():
    return cm.available_days()


@st.cache_data(show_spinner=False)
def _day(d):
    return cm.load_day(d)


@st.cache_data(show_spinner=False)
def _quality(d):
    return cm.day_quality(d, cm.load_day(d))


days = _days()
if not days:
    st.error("No collected option-chain data found under `data/option_chain`.")
    st.stop()

sel = st.sidebar.selectbox("Trading day", days, index=len(days) - 1,
                           format_func=lambda d: d.isoformat())
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Skew warning ≥ {cm.SKEW_WARN_S}s · critical ≥ {cm.SKEW_CRIT_S}s\n\n"
    f"Batch size {cm.BATCH_SIZE} · pause {cm.REQUEST_PAUSE_S}s · "
    f"expiry gap {cm.EXPIRY_GAP_S}s"
)

df = _day(sel)
ps = cm.poll_summary(df)
q = _quality(sel)
per_contract = cm.has_per_contract_timing(df)

if not per_contract:
    st.warning(
        "**Per-contract timestamps are not recorded under `dgp-v1`.** Every row "
        "of a poll shares one `snapshot_ts`, so earliest/latest contract time, "
        "realized within-snapshot spread and true per-contract offsets cannot "
        "be observed. Values below labelled MODELLED are reconstructed from the "
        "collector algorithm (strike-sorted tokens, batches of "
        f"{cm.BATCH_SIZE}, {cm.REQUEST_PAUSE_S}s pause) — they are **not "
        "observations**. Quantities that cannot be modelled honestly are marked "
        "UNAVAILABLE.",
        icon="⚠️",
    )

# ---------------------------------------------------------------- warnings
if q["warnings"]:
    for w in q["warnings"]:
        st.error(f"⚠️ {w}")
else:
    st.success("No collection-quality warnings for this day.")

# ---------------------------------------------------------------- live timing
st.subheader("Live collection timing")
latest = ps.iloc[-1]
c = st.columns(4)
c[0].metric("Latest poll timestamp", str(latest["snapshot_ts"])[:19])
c[0].caption(MEASURED)
c[1].metric("Latest spot", f"{latest['spot']:,.2f}")
c[1].caption(f"{MEASURED} — but stamped *before* the fetch under dgp-v1")
c[2].metric("Contract count", int(latest["n_contracts"]))
c[2].caption(MEASURED)
c[3].metric("Batch count", int(latest["modelled_batches"]))
c[3].caption(f"{MODELLED} — ⌈contracts/{cm.BATCH_SIZE}⌉")

c = st.columns(4)
c[0].metric("Earliest contract timestamp", "—")
c[0].caption(UNAVAIL)
c[1].metric("Latest contract timestamp", "—")
c[1].caption(UNAVAIL)
skew = latest["modelled_skew_within_expiry_s"]
c[2].metric("Within-expiry spread", f"{skew:.1f}s",
            help=f"Full-poll spread across expiries: {latest['modelled_skew_s']:.1f}s")
c[2].caption(f"{MODELLED} — **lower bound** (full ladder is fetched, then trimmed)")
c[3].metric("Collection duration", f"{latest['gap_s'] - cm.NOMINAL_POLL_S:.1f}s"
            if np.isfinite(latest["gap_s"]) else "—")
c[3].caption(f"{MODELLED} — realized gap minus nominal {cm.NOMINAL_POLL_S:.0f}s poll")

# ---------------------------------------------------------------- quality score
st.subheader("Daily collection-quality score")
c = st.columns([1, 3])
c[0].metric(f"{sel.isoformat()}", f"{q['score']:.1f} / 100", q["verdict"])
comp = pd.DataFrame({"component": list(q["components"]),
                     "value": [round(v, 3) for v in q["components"].values()]})
c[1].plotly_chart(
    px.bar(comp, x="component", y="value", range_y=[0, 1],
           title="Score components (1.0 = ideal)"),
    use_container_width=True)

st.dataframe(pd.DataFrame([{k: v for k, v in q.items()
                            if k not in ("components", "warnings")}]),
             use_container_width=True, hide_index=True)

# ------------------------------------------- 1. strike/timestamp synchronization
st.subheader("1. Strike vs timestamp offset")
snap_choice = st.select_slider(
    "Poll", options=list(ps["snapshot_ts"]), value=ps["snapshot_ts"].iloc[-1],
    format_func=lambda t: str(t)[11:19])
off = cm.modelled_contract_offsets(df, snap_choice)
if not off.empty:
    off["expiry_label"] = off["expiry"].dt.date.astype(str)
    fig = px.scatter(off, x="strike", y="modelled_offset_s", color="expiry_label",
                     symbol="option_type",
                     labels={"modelled_offset_s": "offset within poll (s)"},
                     title="MODELLED per-contract offset — monotonic in strike by construction")
    fig.add_hline(y=cm.SKEW_WARN_S, line_dash="dot", annotation_text="warn")
    fig.add_hline(y=cm.SKEW_CRIT_S, line_dash="dash", annotation_text="critical")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{MODELLED}. `option_instruments()` sorts by (strike, option_type) and "
        f"`_fetch_market_data()` chunks in {cm.BATCH_SIZE}s with a "
        f"{cm.REQUEST_PAUSE_S}s pause, so the skew is **systematically monotonic "
        "in strike** — the shape that can imitate strike migration. This is the "
        "finding that made EXP035's migration family unfalsifiable.")

# ------------------------------------------------- 2. offset heatmap
st.subheader("2. Contract collection-offset heatmap")
if not off.empty:
    piv = off.pivot_table(index="option_type", columns="strike",
                          values="modelled_offset_s", aggfunc="first")
    st.plotly_chart(
        go.Figure(go.Heatmap(z=piv.to_numpy(), x=[str(c) for c in piv.columns],
                             y=list(piv.index), colorbar_title="offset (s)"))
        .update_layout(title="MODELLED offset by strike and option type",
                       xaxis_title="strike", yaxis_title=""),
        use_container_width=True)
    st.caption(MODELLED)

# ------------------------------------------ 3. historical skew distribution
st.subheader("3. Historical within-snapshot skew")
n_hist = st.slider("Days of history", 5, len(days), min(30, len(days)))
hist = []
for d in days[-n_hist:]:
    qq = _quality(d)
    hist.append({"day": qq["day"], "modelled_skew_within_expiry_s": qq["modelled_max_skew_within_expiry_s"],
                 "score": qq["score"], "verdict": qq["verdict"],
                 "median_gap_s": qq["median_gap_s"], "spot_std": qq["spot_std"],
                 "n_snapshots": qq["n_snapshots"]})
h = pd.DataFrame(hist)
c = st.columns(2)
c[0].plotly_chart(
    px.histogram(h, x="modelled_skew_within_expiry_s", nbins=20,
                 title="Distribution of modelled max skew (per day)"),
    use_container_width=True)
c[1].plotly_chart(
    px.line(h, x="day", y="score", markers=True, range_y=[0, 100],
            title="Daily collection-quality score"),
    use_container_width=True)
st.dataframe(h, use_container_width=True, hide_index=True)

# ------------------------------------------------- expiry vs offset
st.subheader("Expiry vs timestamp offset")
if not off.empty:
    e = (off.groupby(off["expiry"].dt.date)["modelled_offset_s"]
         .agg(["min", "max", "mean"]).reset_index()
         .rename(columns={"expiry": "expiry_date"}))
    st.dataframe(e, use_container_width=True, hide_index=True)
    st.caption(
        f"{MODELLED}. Each expiry is fetched in turn with a "
        f"{cm.EXPIRY_GAP_S}s gap, and the *next* expiry reuses the cached spot — "
        "which is why EXP034 restricted itself to the near expiry.")

st.markdown("---")
st.caption(
    "Read-only. Reads `data/option_chain` only. Frozen experiments EXP032, "
    "EXP033 and EXP034-A are never accessed or modified. Remediation proposals: "
    "`docs/data_collection_redesign.md`.")
