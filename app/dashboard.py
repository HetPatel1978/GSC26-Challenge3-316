"""
Streamlit dashboard for Challenge 3: data-center resource usage +
predicted task-failure risk, and the model comparison table.

Runs entirely off small, git-tracked precomputed files (data/dashboard_
sample.parquet, data/dashboard_events.parquet, results/model_metrics.json)
built by src/eval/build_dashboard_sample.py and src/eval/compare_models.py
-- no need for the full ~27GB local trace download to demo this.

Usage:
    streamlit run app/dashboard.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
US_PER_DAY = 1_000_000 * 86400

MACHINE_COLOR = "#3b6ea5"
MEM_COLOR = "#3ba55d"
RISK_COLOR = "#a53b3b"
EVENT_COLORS = {"task_fail": "#c98a2c", "machine_remove": "#a53b3b"}

st.set_page_config(page_title="Challenge 3: Failure Prediction", layout="wide")


@st.cache_data
def load_data():
    sample_path = REPO_ROOT / "data" / "dashboard_sample.parquet"
    events_path = REPO_ROOT / "data" / "dashboard_events.parquet"
    metrics_path = REPO_ROOT / "results" / "model_metrics.json"
    if not sample_path.exists():
        return None, None, None
    sample = pl.read_parquet(sample_path).to_pandas()
    events = pl.read_parquet(events_path).to_pandas() if events_path.exists() else pd.DataFrame()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return sample, events, metrics


sample, events, metrics = load_data()

st.title("Data-Center Resource Usage & Failure Prediction")
st.caption(
    "Google Cluster Trace 2011-2 — task/machine resource usage, engineered 30-min-window "
    "features, and imminent-failure (next-30-min) predictions from the tuned XGBoost model."
)

if sample is None:
    st.error(
        "data/dashboard_sample.parquet not found. Build it first:\n\n"
        "```\npython src/eval/build_dashboard_sample.py\n```"
    )
    st.stop()

machines = sorted(sample["machine_id"].unique().tolist())
removed_machines = set(events.loc[events["event"] == "machine_remove", "machine_id"].tolist()) if not events.empty else set()

st.sidebar.header("Machine selector")
machine_labels = {m: f"{m} {'(removed)' if m in removed_machines else '(healthy)'}" for m in machines}
selected = st.sidebar.selectbox("Machine", machines, format_func=lambda m: machine_labels[m])

st.sidebar.markdown(
    "Machines are a small precomputed sample (2 EDA worked examples + a few "
    "random removed/healthy machines) -- not the full ~12.5k-machine cluster."
)

mdf = sample[sample["machine_id"] == selected].sort_values("window_start").copy()
mdf["trace_day"] = mdf["window_start"] / US_PER_DAY
mevents = events[events["machine_id"] == selected] if not events.empty else pd.DataFrame()

tab_overview, tab_sim = st.tabs(["Overview", "Live Simulation"])

with tab_overview:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Machine {selected}: usage & predicted risk over time")

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        axes[0].plot(mdf["trace_day"], mdf["cpu_sum"], color=MACHINE_COLOR, lw=1, label="summed CPU rate")
        axes[0].set_ylabel("summed CPU rate\n(all tasks)", color=MACHINE_COLOR)
        axes[0].set_title("Resource usage")

        axes[1].plot(mdf["trace_day"], mdf["max_predicted_risk"], color=RISK_COLOR, lw=1,
                     label="max predicted failure risk (XGBoost, tuned)")
        axes[1].set_ylabel("predicted risk\n(0-1)", color=RISK_COLOR)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("trace day")
        axes[1].set_title("Predicted imminent-failure risk (next 30 min)")

        for _, ev in mevents.iterrows():
            day = ev["time"] / US_PER_DAY
            for ax in axes:
                ax.axvline(day, color=EVENT_COLORS.get(ev["event"], "gray"), ls=":", lw=1, alpha=0.8)
            axes[0].annotate(ev["event"], (day, axes[0].get_ylim()[1]), rotation=90,
                              fontsize=7, va="top", ha="right", color=EVENT_COLORS.get(ev["event"], "gray"))

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Machine summary")
        st.metric("Windows observed", len(mdf))
        st.metric("Peak predicted risk", f"{mdf['max_predicted_risk'].max():.3f}" if len(mdf) else "n/a")
        st.metric("Task-fail events", int((mevents["event"] == "task_fail").sum()) if not mevents.empty else 0)
        st.metric("Machine REMOVE", "yes" if selected in removed_machines else "no")

    st.divider()
    st.subheader("Model comparison")
    st.caption("Same time-based test split across all 5 models (src/eval/dataset.py) -- see docs/03-BASELINE-RESULTS.md.")

    if metrics:
        table_rows = []
        for key, m in metrics.items():
            table_rows.append({
                "Model": m["label"], "Precision": m["precision"], "Recall": m["recall"],
                "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
    else:
        st.info("results/model_metrics.json not found -- run `python src/eval/compare_models.py` first.")

    plot_path = REPO_ROOT / "results" / "plots" / "model_comparison.png"
    if plot_path.exists():
        st.image(str(plot_path), width="stretch")

with tab_sim:
    st.subheader(f"Live simulation: machine {selected}")
    st.caption(
        "Scrub the slider forward in time to watch resource usage and predicted failure risk "
        "unfold, exactly as an operator monitoring this machine would have seen it. The dotted "
        "line marks when the actual event happened."
    )

    if mdf.empty:
        st.info("No windows for this machine.")
    else:
        day_min, day_max = float(mdf["trace_day"].min()), float(mdf["trace_day"].max())
        if day_min == day_max:
            day_max = day_min + 1e-6
        current_day = st.slider(
            "Simulated time (trace day)", min_value=day_min, max_value=day_max,
            value=day_min, step=(day_max - day_min) / 200,
        )

        visible = mdf[mdf["trace_day"] <= current_day]
        upcoming_events = mevents[mevents["time"] / US_PER_DAY <= current_day] if not mevents.empty else pd.DataFrame()

        m1, m2, m3, m4 = st.columns(4)
        if len(visible):
            latest = visible.iloc[-1]
            m1.metric("Summed CPU rate", f"{latest['cpu_sum']:.4f}")
            m2.metric("Summed memory", f"{latest['mem_sum']:.4f}")
            m3.metric("Tasks running", int(latest["n_tasks"]))
            m4.metric("Predicted risk", f"{latest['max_predicted_risk']:.3f}",
                      delta="ALERT" if latest["max_predicted_risk"] >= 0.40 else None,
                      delta_color="inverse")
        else:
            m1.metric("Summed CPU rate", "n/a")
            m2.metric("Summed memory", "n/a")
            m3.metric("Tasks running", "n/a")
            m4.metric("Predicted risk", "n/a")

        if not mevents.empty and len(upcoming_events) > 0:
            st.error(f"Event has occurred as of this point in the simulation: "
                     f"{', '.join(upcoming_events['event'].unique())}")
        elif not mevents.empty:
            next_ev_day = (mevents["time"] / US_PER_DAY).min()
            st.info(f"No event yet -- next real event on this machine is at trace day {next_ev_day:.3f} "
                    f"({(next_ev_day - current_day) * 24 * 60:.0f} min from the slider's current position).")

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(visible["trace_day"], visible["cpu_sum"], color=MACHINE_COLOR, lw=1.3)
        axes[0].set_xlim(day_min, day_max)
        axes[0].set_ylim(0, max(mdf["cpu_sum"].max(), 1e-6) * 1.1)
        axes[0].set_ylabel("summed CPU rate", color=MACHINE_COLOR)
        axes[0].set_title("Resource usage (revealed up to slider position)")

        axes[1].plot(visible["trace_day"], visible["max_predicted_risk"], color=RISK_COLOR, lw=1.3)
        axes[1].axhline(0.40, color="gray", ls="--", lw=0.8, label="XGBoost (tuned) alert threshold")
        axes[1].set_xlim(day_min, day_max)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("predicted risk", color=RISK_COLOR)
        axes[1].set_xlabel("trace day")
        axes[1].set_title("Predicted imminent-failure risk (revealed up to slider position)")
        axes[1].legend(fontsize=8, loc="upper left")

        for _, ev in mevents.iterrows():
            day = ev["time"] / US_PER_DAY
            for ax in axes:
                ax.axvline(day, color=EVENT_COLORS.get(ev["event"], "gray"), ls=":", lw=1.5)
            axes[0].annotate(f"actual {ev['event']}", (day, axes[0].get_ylim()[1]), rotation=90,
                              fontsize=7, va="top", ha="right", color=EVENT_COLORS.get(ev["event"], "gray"))
        axes[0].axvline(current_day, color="black", lw=1, alpha=0.5)
        axes[1].axvline(current_day, color="black", lw=1, alpha=0.5)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
