import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import os
import pandas as pd
import plotly.express as px
import streamlit as st

from ftir.config import RESULTS_DIR
from ftir.data.config import SAMPLE_TYPES
from shared_settings import render_appearance_sidebar

st.set_page_config(page_title="ML Results", layout="wide")
st.title("ML Results")

# ── Load results: local CSV or DagsHub MLflow ─────────────────────────────────

DAGSHUB_MLFLOW_URI = "https://dagshub.com/pedroasvalente/ftir-sports-ml.mlflow"
METRIC_COLS_ALL = ["balanced_accuracy", "mcc", "cohen_kappa", "f1_weighted", "f1_macro", "roc_auc"]


@st.cache_data(ttl=300)
def load_from_dagshub(token: str) -> pd.DataFrame:
    """Fetch all child runs from DagsHub MLflow and return as a dataframe."""
    import mlflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = "pedroasvalente"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    rows = []
    for exp in client.search_experiments():
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.model != ''",   # only child runs (have model tag)
        )
        for run in runs:
            tags = run.data.tags
            metrics = run.data.metrics
            params = run.data.params
            row = {
                "run_id": run.info.run_id,
                "experiment": exp.name,
                "sample_type": tags.get("sample_type", ""),
                "target": tags.get("target", ""),
                "timepoints": tags.get("timepoints", ""),
                "model": tags.get("model", ""),
                "search": tags.get("search", ""),
                "config": tags.get("config", ""),
                "n_train": params.get("n_train"),
                "n_test": params.get("n_test"),
                "n_synthetic": params.get("n_synthetic"),
                "apply_pls": params.get("apply_pls"),
                **{m: metrics.get(m) for m in METRIC_COLS_ALL},
            }
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_from_local() -> tuple[pd.DataFrame, list[str]]:
    results_dir = Path(RESULTS_DIR)
    csv_files = list(results_dir.rglob("results_summary.csv"))
    if not csv_files:
        return pd.DataFrame(), []
    run_names = [f.parent.name for f in csv_files]
    return csv_files, run_names


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data source")
    token = os.environ.get("DAGSHUB_USER_TOKEN", "")
    use_dagshub = st.toggle(
        "Load from DagsHub",
        value=bool(token),
        help="Fetches live runs from DagsHub MLflow. Requires token in secrets.",
    )

csv_files, run_names = load_from_local()
has_local = len(run_names) > 0

with st.sidebar:
    st.header("Filters")

if use_dagshub:
    if not token:
        st.warning("No DAGSHUB_USER_TOKEN found in secrets. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    with st.spinner("Fetching runs from DagsHub…"):
        df = load_from_dagshub(token)
    if df.empty:
        st.info("No runs found on DagsHub yet. Run the training first.")
        st.stop()
    st.caption(f"**{len(df)} runs** loaded from DagsHub MLflow")
elif has_local:
    with st.sidebar:
        selected_run = st.selectbox("Local run", run_names)
        selected_file = csv_files[run_names.index(selected_run)]
    df = pd.read_csv(selected_file)
else:
    st.info("No results found locally or on DagsHub. Run `docker compose run --rm train` first.")
    st.stop()

# ── Shared sidebar controls ───────────────────────────────────────────────────
_, _, model_colors, matrix_colors = render_appearance_sidebar(show_models=True, show_matrices=True)

with st.sidebar:
    matrix_filter = st.multiselect("Matrix", SAMPLE_TYPES, default=SAMPLE_TYPES)
    model_filter = st.multiselect(
        "Model", sorted(df["model"].dropna().unique()),
        default=list(df["model"].dropna().unique())
    )
    tp_opts = sorted(df["timepoints"].dropna().unique()) if "timepoints" in df.columns else []
    tp_filter = st.multiselect("Timepoints", tp_opts, default=tp_opts) if tp_opts else []

    metric_opts = [c for c in METRIC_COLS_ALL if c in df.columns and df[c].notna().any()]
    metric = st.selectbox("Sort by", metric_opts)
    top_n = st.slider("Top-N per matrix", 1, 10, 3)

METRIC_COLS = [c for c in METRIC_COLS_ALL if c in df.columns]
FMT = {c: "{:.3f}" for c in METRIC_COLS}

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = df[df["sample_type"].isin(matrix_filter) & df["model"].isin(model_filter)]
if tp_filter and "timepoints" in df.columns:
    filtered = filtered[filtered["timepoints"].isin(tp_filter)]
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

# ── Top-N per matrix ──────────────────────────────────────────────────────────
st.subheader(f"Top-{top_n} per matrix — {metric}")
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
st.subheader(f"Heatmap — {metric} (best per matrix × model)")
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
