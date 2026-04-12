import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from ftir.config import RESULTS_DIR
from ftir.data.config import SAMPLE_TYPES
from shared_settings import METRIC_COLS_ALL, render_appearance_sidebar, render_data_source_sidebar

st.set_page_config(page_title="ML Results", layout="wide")
st.title("ML Results")

# ── Load data ─────────────────────────────────────────────────────────────────
df, use_dagshub = render_data_source_sidebar(RESULTS_DIR)
if df is None:
    st.stop()

# ── Shared sidebar controls ───────────────────────────────────────────────────
_, _, model_colors, matrix_colors = render_appearance_sidebar(show_models=True, show_matrices=True)

METRIC_COLS = [c for c in METRIC_COLS_ALL if c in df.columns]
FMT = {c: "{:.3f}" for c in METRIC_COLS}

with st.sidebar:
    st.header("Filters")
    matrix_filter = st.multiselect("Matrix", SAMPLE_TYPES, default=SAMPLE_TYPES)
    model_filter = st.multiselect(
        "Model", sorted(df["model"].dropna().unique()),
        default=list(df["model"].dropna().unique())
    )
    tp_opts = sorted(df["timepoints"].dropna().unique()) if "timepoints" in df.columns else []
    tp_filter = st.multiselect("Timepoints", tp_opts, default=tp_opts) if tp_opts else []

    metric = st.selectbox("Sort by", [c for c in METRIC_COLS if df[c].notna().any()])
    top_n = st.slider("Top-N per matrix", 1, 10, 3)

    search_query = st.text_input(
        "Search model/matrix",
        value="",
        placeholder="e.g. Random Forest, SERUM \u2026",
        help="Case-insensitive substring match against matrix (sample_type) and model columns.",
    )

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = df[df["sample_type"].isin(matrix_filter) & df["model"].isin(model_filter)]
if tp_filter and "timepoints" in df.columns:
    filtered = filtered[filtered["timepoints"].isin(tp_filter)]

if search_query.strip():
    q = search_query.strip().lower()
    mask = (
        filtered["sample_type"].str.lower().str.contains(q, na=False)
        | filtered["model"].str.lower().str.contains(q, na=False)
    )
    filtered = filtered[mask]

filtered = filtered.sort_values(metric, ascending=False).reset_index(drop=True)
filtered.index += 1

st.markdown(f"**{len(filtered)} results**")

# ── Full table ────────────────────────────────────────────────────────────────
with st.expander("All results", expanded=False):
    show_df = filtered.copy()
    if "run_id" in show_df.columns:
        show_df["DagsHub"] = show_df["run_id"].apply(
            lambda rid: f"[view](https://dagshub.com/pedroasvalente/ftir-sports-ml.mlflow/#/experiments/0/runs/{rid})"
        )
    st.dataframe(show_df.style.format(FMT), use_container_width=True)

# Download filtered results as CSV
if not filtered.empty:
    csv_bytes = filtered.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="Download filtered results (CSV)",
        data=csv_bytes,
        file_name="ml_results_filtered.csv",
        mime="text/csv",
    )

# ── Top-N per matrix ──────────────────────────────────────────────────────────
st.subheader(f"Top-{top_n} per matrix \u2014 {metric}")
best_cols = [c for c in ["sample_type", "timepoints", "model", "search"] + METRIC_COLS
             if c in filtered.columns]
best = (
    filtered.sort_values(metric, ascending=False)
    .groupby("sample_type").head(top_n)
    .reset_index(drop=True)[best_cols]
)
best.index += 1
st.dataframe(
    best.style.format(FMT).background_gradient(subset=[metric], cmap="RdYlGn", vmin=0.5, vmax=1.0),
    use_container_width=True,
)

# ── Performance heatmap ───────────────────────────────────────────────────────
st.subheader(f"Heatmap \u2014 {metric} (best per matrix \u00d7 model)")
if not filtered.empty:
    pivot = (
        filtered.sort_values(metric, ascending=False)
        .groupby(["sample_type", "model"])[metric].first()
        .reset_index()
        .pivot(index="model", columns="sample_type", values=metric)
        .reindex(columns=SAMPLE_TYPES, fill_value=None)
    )
    fig_heat = px.imshow(
        pivot, color_continuous_scale="RdYlGn", zmin=0.5, zmax=1.0,
        text_auto=".3f", aspect="auto",
        labels={"color": metric, "x": "Matrix", "y": "Model"},
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Distribution boxplot ──────────────────────────────────────────────────────
st.subheader(f"Distribution of {metric} by matrix")
fig_box = px.box(
    filtered, x="sample_type", y=metric, color="model",
    points="all",
    hover_data=[c for c in ["model", "search", "timepoints"] if c in filtered.columns],
    labels={"sample_type": "Matrix", metric: metric},
    category_orders={"sample_type": SAMPLE_TYPES},
    color_discrete_map=model_colors,
)
fig_box.update_layout(height=420)
st.plotly_chart(fig_box, use_container_width=True)

# ── Per-class sensitivity heatmap ─────────────────────────────────────────────
SENSITIVITY_COLS = ["sensitivity_class0", "sensitivity_class1", "sensitivity_class2"]
available_sens_cols = [c for c in SENSITIVITY_COLS if c in filtered.columns]

if available_sens_cols and not filtered.empty:
    st.subheader("Per-class sensitivity \u2014 heatmap (matrix \u00d7 class)")

    # Average sensitivity per matrix across all filtered runs
    sens_df = (
        filtered.groupby("sample_type")[available_sens_cols]
        .mean()
        .reindex(SAMPLE_TYPES)
        .dropna(how="all")
    )

    # Friendly column labels: "Class 0", "Class 1", "Class 2"
    sens_df.columns = [
        c.replace("sensitivity_", "").replace("class", "Class ") for c in sens_df.columns
    ]
    sens_df.index.name = "Matrix"

    fig_sens = px.imshow(
        sens_df,
        color_continuous_scale="Blues",
        zmin=0.0,
        zmax=1.0,
        text_auto=".3f",
        aspect="auto",
        labels={"color": "Sensitivity", "x": "Class", "y": "Matrix"},
    )
    fig_sens.update_layout(
        height=max(250, 60 * len(sens_df) + 100),
        xaxis_title="Class",
        yaxis_title="Matrix",
    )
    st.plotly_chart(fig_sens, use_container_width=True)
