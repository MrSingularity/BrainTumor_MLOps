"""NeuroScan Operations Dashboard.

Developer-facing monitoring page for prediction volume, prediction mix,
and a lightweight operational drift summary based on stored logs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brain_tumor_mlops.api.logs_viewer import (  # noqa: E402
    build_drift_summary,
    load_logs,
    summarize_logs,
)
from brain_tumor_mlops.monitoring.drift_job import get_status_file  # noqa: E402

OPS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0b1016 !important;
    color: #d5dbe3 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1300px !important; }

[data-testid="stMetric"] {
    background: #0f1620;
    border: 1px solid #223043;
    border-radius: 2px;
    padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #71c7ff !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #eef4fb !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stExpander"] {
    background: #0f1620 !important;
    border: 1px solid #223043 !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
    color: #71c7ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.ops-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #71c7ff;
    border-bottom: 1px solid #223043;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    margin-top: 1rem;
}
.ops-note {
    color: #8ca0b8;
    font-size: 0.9rem;
}

/* Make the log table white with black text for readability */
.stDataFrame, .stTable, .css-1v0mbdj.e16nr0p30 {
    background: white !important;
    color: black !important;
}

.stDataFrame table td, .stTable table td {
    background: white !important;
    color: black !important;
}

/* Keep charts aligned with the dashboard theme */
.stPlotlyChart, .stImage {
    background: transparent !important;
}

</style>
"""


def main() -> None:
    """Render the developer-facing monitoring dashboard."""
    st.set_page_config(
        page_title="NeuroScan Ops Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(OPS_CSS, unsafe_allow_html=True)

    logs = load_logs()
    summary = summarize_logs(logs)
    drift = build_drift_summary(logs)

    st.markdown("<div class='ops-header'>NeuroScan · Operations Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ops-note'>Developer-facing monitoring only. This page summarizes prediction logs, model mix, and a lightweight operational drift signal.</div>",
        unsafe_allow_html=True,
    )

    # (Drift status card is shown after the model mix and chart)

    if not logs:
        st.info("No prediction logs found yet. Run the main app and generate a few predictions first.")
        return

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predictions", f"{summary['total_events']}")
    with m2:
        st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
    with m3:
        st.metric("Positive Rate", f"{summary['positive_prediction_rate']:.1f}%")

    # Small per-drift indicators (keep deltas and current window in the top row;
    # the full drift-status card is shown after the model mix)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Avg Latency", f"{summary['average_latency_ms']:.0f} ms")

    with d2:
        st.metric("Positive Delta", f"{drift['positive_rate_delta']:+.1f} pts")
    with d3:
        st.metric("Current Window", f"{drift['current_events']}")

    mix_rows: list[dict[str, Any]] = []
    for model_name, count in summary["predictions_by_model"].items():
        mix_rows.append({"model": model_name, "predictions": count})
    mix_df = pd.DataFrame(mix_rows)
    if not mix_df.empty:
        mix_df = mix_df.sort_values("predictions", ascending=False).reset_index(drop=True)

    mix_col, chart_col = st.columns([3, 1], gap="large")
    with mix_col:
        st.markdown("<div class='ops-header'>Model mix</div>", unsafe_allow_html=True)
        if mix_df.empty:
            st.info("No model mix data yet.")
        else:
            st.dataframe(mix_df, width="stretch")

    with chart_col:
        st.markdown("<div class='ops-header'>Outcomes</div>", unsafe_allow_html=True)

        records: list[dict[str, Any]] = []
        for e in logs:
            is_success = "label" in e and e.get("label") is not None
            records.append({"status": "✓" if is_success else "✗"})

        df = pd.DataFrame.from_records(records)
        if df.empty:
            st.info("No chart data yet.")
        else:
            counts = df["status"].value_counts().reindex(["✓", "✗"]).fillna(0)
            fig, ax = plt.subplots(figsize=(2.4, 2.4))
            fig.patch.set_facecolor("#0b1016")
            ax.set_facecolor("#0b1016")
            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=["Success", "Failure"],
                autopct="%1.0f%%",
                startangle=90,
                colors=["#22c55e", "#ef4444"],
                textprops={"fontsize": 8, "color": "#eef4fb"},
                wedgeprops={"edgecolor": "#0b1016", "linewidth": 1},
            )
            ax.set_title("Success Rate", fontsize=9, color="#eef4fb")
            for text in texts:
                text.set_color("#eef4fb")
            for text in autotexts:
                text.set_color("#eef4fb")
            st.pyplot(fig, width="stretch")

    # --- Drift status card lives after the model mix and chart ---
    # Load persistent drift status (written by the scheduled drift job)
    status_path = Path(get_status_file())
    drift_status: dict | None = None
    if status_path.exists():
        try:
            drift_status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            drift_status = None

    st.markdown("<div class='ops-header'>Drift status</div>", unsafe_allow_html=True)
    if not drift_status:
        # Fallback to operational summary produced from recent logs
        st.info("Drift status: no scheduled report found. Showing live operational summary below.")
        st.json(drift)
    else:
        status = drift_status.get("status", "UNKNOWN")
        score = float(drift_status.get("score", 0.0))
        generated_at = drift_status.get("generated_at", "")
        report_html = drift_status.get("report_html", "")
        confidence_delta = drift_status.get("confidence_delta")
        positive_rate_delta = drift_status.get("positive_rate_delta")
        latency_delta_ms = drift_status.get("latency_delta_ms")

        color_map = {"OK": "#22c55e", "WARN": "#f59e0b", "ALERT": "#ef4444", "UNKNOWN": "#94a3b8"}
        badge_color = color_map.get(status, "#94a3b8")

        cols = st.columns([2, 4, 3])
        with cols[0]:
            st.markdown(
                f"<div style='background:{badge_color};color:#fff;padding:0.6rem;border-radius:6px;text-align:center;font-weight:600'>{status}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Updated: {generated_at}")
        with cols[1]:
            st.progress(min(max(score, 0.0), 1.0), text=f"Drift score: {score:.3f}")
            if report_html:
                try:
                    p = Path(report_html)
                    if p.exists():
                        # clickable file:// link and a download button for the HTML report
                        try:
                            st.markdown(f"[Open latest report]({p.as_uri()})")
                        except Exception:
                            st.markdown(f"Latest report path: {report_html}")
                        with open(p, "rb") as fh:
                            data = fh.read()
                        st.download_button("Download latest report", data, file_name=p.name, mime="text/html")
                    else:
                        st.markdown("Latest report: unavailable")
                except Exception:
                    st.markdown("Latest report: unavailable")
        with cols[2]:
            st.write("**Deltas**")
            st.write(f"Confidence: {confidence_delta if confidence_delta is not None else 'N/A'}")
            st.write(f"Positive rate: {positive_rate_delta if positive_rate_delta is not None else 'N/A'}")
            st.write(f"Latency (ms): {latency_delta_ms if latency_delta_ms is not None else 'N/A'}")

    st.markdown("<div class='ops-header'>Recent events</div>", unsafe_allow_html=True)
    # Build a tidy DataFrame for events and show a styled table with ticks/crosses
    records: list[dict[str, Any]] = []
    for e in logs:
        is_success = "label" in e and e.get("label") is not None
        records.append(
            {
                "time": e.get("timestamp", ""),
                "status": "✓" if is_success else "✗",
                "label": e.get("label", ""),
                "confidence": round(float(e.get("confidence", 0.0)), 3) if is_success else None,
                "latency_ms": round(float(e.get("latency_ms", 0.0)), 1) if e.get("latency_ms") is not None else None,
                "model": e.get("model_name", ""),
                "image": e.get("image_hash", ""),
                "note": e.get("error", "") if not is_success else "",
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        st.info("No recent events to display.")
    else:
        # Show most recent first
        df = df.iloc[::-1].reset_index(drop=True)
        # Limit columns shown for clarity
        display_df = df[["time", "status", "label", "confidence", "latency_ms", "model", "note"]]
        st.dataframe(display_df, width="stretch")

    with st.expander("Log summary", expanded=False):
        st.json(summary)

    with st.expander("Drift details", expanded=False):
        st.json(drift)


if __name__ == "__main__":
    main()
