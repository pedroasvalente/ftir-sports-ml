import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ftir.config import RESULTS_DIR
from ftir.data.config import SAMPLE_TYPES
from shared_settings import render_appearance_sidebar

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

results_dir = Path(RESULTS_DIR)
csv_files = list(results_dir.rglob("results_summary.csv"))

if not csv_files:
    st.info("No results found. Run `docker compose run --rm train` first.")
    st.stop()

run_names = [f.parent.name for f in csv_files]

with st.sidebar:
    st.header("Experiment")
    selected_run = st.selectbox("Run", run_names)
    selected_file = csv_files[run_names.index(selected_run)]
    df = pd.read_csv(selected_file)

    st.header("Filters")
    matrix = st.selectbox("Matrix", SAMPLE_TYPES)
    if "timepoints" in df.columns:
        tp_opts = sorted(df["timepoints"].unique())
        timepoints = st.selectbox("Timepoints", tp_opts)
    else:
        timepoints = None

_, _, model_colors, _ = render_appearance_sidebar(show_models=True)

METRIC_COLS = [c for c in ["balanced_accuracy", "mcc", "cohen_kappa", "f1_weighted", "f1_macro", "roc_auc"]
               if c in df.columns]

# ── Filter to matrix + timepoints ─────────────────────────────────────────────
sub = df[df["sample_type"] == matrix].copy()
if timepoints is not None and "timepoints" in sub.columns:
    sub = sub[sub["timepoints"] == timepoints]

if sub.empty:
    st.warning("No results for this combination yet.")
    st.stop()

# ── Radar chart ───────────────────────────────────────────────────────────────
st.subheader(f"Radar — all metrics per model ({matrix})")

best_per_model = (
    sub.sort_values("balanced_accuracy", ascending=False)
    .groupby("model")
    .first()
    .reset_index()
)

fig_radar = go.Figure()
colors = px.colors.qualitative.Set1

for i, row in best_per_model.iterrows():
    values = [row[m] if pd.notna(row.get(m)) else 0 for m in METRIC_COLS]
    values += [values[0]]  # close polygon
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=METRIC_COLS + [METRIC_COLS[0]],
        name=row["model"],
        line=dict(color=colors[i % len(colors)], width=2),
        fill="toself",
        fillcolor=colors[i % len(colors)],
        opacity=0.15,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    height=500,
    title=f"{matrix} — model comparison across all metrics",
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Side-by-side metric bars ───────────────────────────────────────────────────
st.subheader("Metrics side-by-side")

melted = best_per_model.melt(
    id_vars="model", value_vars=METRIC_COLS,
    var_name="metric", value_name="value"
)

fig_bar = px.bar(
    melted, x="metric", y="value", color="model",
    barmode="group",
    title=f"{matrix} — metric comparison",
    labels={"value": "Score", "metric": "Metric"},
    color_discrete_map=model_colors,
    text_auto=".3f",
)
fig_bar.update_layout(height=420, yaxis_range=[0, 1.05])
st.plotly_chart(fig_bar, use_container_width=True)

# ── Scatter: balanced_accuracy vs MCC ────────────────────────────────────────
if "mcc" in sub.columns:
    st.subheader("Balanced accuracy vs MCC — all runs")
    fig_sc = px.scatter(
        sub, x="mcc", y="balanced_accuracy",
        color="model", symbol="model",
        hover_data=[c for c in ["model", "search", "timepoints", "n_synthetic"] if c in sub.columns],
        title=f"{matrix} — all runs",
        labels={"mcc": "MCC", "balanced_accuracy": "Balanced Accuracy"},
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_sc.add_hline(y=0.8, line_dash="dot", line_color="gray",
                     annotation_text="BA = 0.80")
    fig_sc.add_vline(x=0.6, line_dash="dot", line_color="gray",
                     annotation_text="MCC = 0.60")
    fig_sc.update_traces(marker_size=10)
    fig_sc.update_layout(height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Cross-matrix summary table ────────────────────────────────────────────────
st.subheader("Cross-matrix summary — best model per matrix")

all_best = []
for mat in SAMPLE_TYPES:
    sub_mat = df[df["sample_type"] == mat]
    if timepoints is not None and "timepoints" in sub_mat.columns:
        sub_mat = sub_mat[sub_mat["timepoints"] == timepoints]
    if sub_mat.empty:
        continue
    row = sub_mat.sort_values("balanced_accuracy", ascending=False).iloc[0]
    all_best.append(row)

if all_best:
    summary = pd.DataFrame(all_best)
    show_cols = [c for c in ["sample_type", "model", "search"] + METRIC_COLS if c in summary.columns]
    FMT = {c: "{:.3f}" for c in METRIC_COLS}
    st.dataframe(
        summary[show_cols].reset_index(drop=True).style
        .format(FMT)
        .background_gradient(subset=["balanced_accuracy"], cmap="RdYlGn", vmin=0.5, vmax=1.0),
        use_container_width=True,
    )
