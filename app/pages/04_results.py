import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from ftir.config import RESULTS_DIR
from ftir.data.config import SAMPLE_TYPES
from shared_settings import render_appearance_sidebar

st.set_page_config(page_title="ML Results", layout="wide")
st.title("ML Results")

results_dir = Path(RESULTS_DIR)
csv_files = list(results_dir.rglob("results_summary.csv"))

if not csv_files:
    st.info("No results found. Run `docker compose run --rm train` first.")
    st.stop()

run_names = [f.parent.name for f in csv_files]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Experiment")
    selected_run = st.selectbox("Run", run_names)
    selected_file = csv_files[run_names.index(selected_run)]
    df = pd.read_csv(selected_file)

    st.header("Filters")
    matrix_filter = st.multiselect("Matrix", SAMPLE_TYPES, default=SAMPLE_TYPES)
    model_filter = st.multiselect("Model", sorted(df["model"].unique()), default=list(df["model"].unique()))
    timepoint_opts = sorted(df["timepoints"].unique()) if "timepoints" in df.columns else []
    if timepoint_opts:
        tp_filter = st.multiselect("Timepoints", timepoint_opts, default=timepoint_opts)
    else:
        tp_filter = []

    st.header("Sorting")
    metric_opts = [c for c in ["balanced_accuracy", "mcc", "cohen_kappa", "f1_weighted", "f1_macro", "roc_auc"]
                   if c in df.columns]
    metric = st.selectbox("Sort by", metric_opts)
    top_n = st.slider("Top-N per matrix", 1, 10, 3)

_, _, model_colors, matrix_colors = render_appearance_sidebar(show_models=True, show_matrices=True)

METRIC_COLS = [c for c in ["balanced_accuracy", "mcc", "cohen_kappa", "f1_weighted", "f1_macro", "roc_auc"]
               if c in df.columns]
FMT = {c: "{:.3f}" for c in METRIC_COLS}
FMT["train_pct"] = "{:.0%}"

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = df[df["sample_type"].isin(matrix_filter) & df["model"].isin(model_filter)]
if tp_filter and "timepoints" in df.columns:
    filtered = filtered[filtered["timepoints"].isin(tp_filter)]
filtered = filtered.sort_values(metric, ascending=False).reset_index(drop=True)
filtered.index += 1

st.markdown(f"**{len(filtered)} results** — run `{selected_run}`")

# ── Full table ────────────────────────────────────────────────────────────────
with st.expander("All results", expanded=False):
    st.dataframe(filtered.style.format(FMT), use_container_width=True)

# ── Top-N per matrix ──────────────────────────────────────────────────────────
st.subheader(f"Top-{top_n} per matrix — sorted by {metric}")

best_cols = [c for c in ["sample_type", "timepoints", "model", "search"] + METRIC_COLS + ["n_synthetic", "apply_pls"]
             if c in filtered.columns]

best = (
    filtered.sort_values(metric, ascending=False)
    .groupby("sample_type")
    .head(top_n)
    .reset_index(drop=True)
    [best_cols]
)
best.index += 1

# Colour scale on the primary metric
st.dataframe(
    best.style
    .format(FMT)
    .background_gradient(subset=[metric], cmap="RdYlGn", vmin=0.5, vmax=1.0),
    use_container_width=True,
)

# ── Performance heatmap ───────────────────────────────────────────────────────
st.subheader(f"Performance heatmap — {metric} (best per matrix × model)")

if not filtered.empty:
    pivot = (
        filtered.sort_values(metric, ascending=False)
        .groupby(["sample_type", "model"])[metric]
        .first()
        .reset_index()
        .pivot(index="model", columns="sample_type", values=metric)
    )
    pivot = pivot.reindex(columns=SAMPLE_TYPES, fill_value=None)

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        zmin=0.5, zmax=1.0,
        text_auto=".3f",
        title=f"{metric} — best per matrix × model",
        labels={"color": metric, "x": "Matrix", "y": "Model"},
        aspect="auto",
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Bar chart — metric by matrix ──────────────────────────────────────────────
st.subheader(f"Distribution of {metric} by matrix")

fig_box = px.box(
    filtered, x="sample_type", y=metric, color="model",
    points="all",
    hover_data=[c for c in ["model", "search", "timepoints"] if c in filtered.columns],
    title=f"{metric} distribution by matrix and model",
    labels={"sample_type": "Matrix", metric: metric},
    category_orders={"sample_type": SAMPLE_TYPES},
    color_discrete_map=model_colors,
)
fig_box.update_layout(height=420)
st.plotly_chart(fig_box, use_container_width=True)
