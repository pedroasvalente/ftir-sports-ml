import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import plotly.express as px

from ftir.config import RESULTS_DIR
from ftir.data.config import SAMPLE_TYPES
from shared_settings import METRIC_COLS_ALL, render_appearance_sidebar, render_data_source_sidebar

st.set_page_config(page_title="ML Results", layout="wide")
st.title("ML Results")

# ── Constants ─────────────────────────────────────────────────────────────────
CM_LABELS = ["Football", "Sedentary", "Ultrarunning"]
MATRIX_DISPLAY = {
    "CAPILAR": "Capillary Blood",
    "PLASMA": "Plasma",
    "SERUM": "Serum",
    "URINE": "Urine",
    "SALIVA": "Saliva",
}

# ── Load results CSV ──────────────────────────────────────────────────────────
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

    sens_df = (
        filtered.groupby("sample_type")[available_sens_cols]
        .mean()
        .reindex(SAMPLE_TYPES)
        .dropna(how="all")
    )
    sens_df.columns = [
        c.replace("sensitivity_", "").replace("class", "Class ") for c in sens_df.columns
    ]
    sens_df.index.name = "Matrix"

    fig_sens = px.imshow(
        sens_df,
        color_continuous_scale="Blues",
        zmin=0.0, zmax=1.0,
        text_auto=".3f", aspect="auto",
        labels={"color": "Sensitivity", "x": "Class", "y": "Matrix"},
    )
    fig_sens.update_layout(
        height=max(250, 60 * len(sens_df) + 100),
        xaxis_title="Class", yaxis_title="Matrix",
    )
    st.plotly_chart(fig_sens, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Confusion Matrices")

# Search in RESULTS_DIR root first, then in run subdirectories
_cm_candidates = [RESULTS_DIR / "cm_data.json"] + sorted(
    RESULTS_DIR.glob("*/cm_data.json"), reverse=True  # most recent run first
)
cm_path = next((p for p in _cm_candidates if p.exists()), None)

if cm_path is None:
    st.info("No confusion matrix data found. Run training to generate cm_data.json.")
    st.stop()

with open(cm_path) as f:
    cm_raw = json.load(f)


def tp_str_to_key(tp_val: str) -> str:
    """
    Normalise any timepoints representation to the cm_data.json key format.
    '[1]'      -> 'tp1'
    '[1, 2, 3]'-> 'tp1_2_3'
    'tp1'      -> 'tp1'      (DagsHub already prefixed)
    '1'        -> 'tp1'
    """
    s = str(tp_val).strip()
    if s.startswith("tp"):          # already normalised (DagsHub format)
        return s
    s = s.strip("[]").replace(", ", "_")   # '[1, 2, 3]' -> '1_2_3'
    return "tp" + s


def build_cm_key(matrix: str, tp_val: str, model: str, search: str = "grid") -> str:
    return f"{matrix}__{tp_str_to_key(tp_val)}__{model}__{search}"


def make_cm_figure(cm_matrix: list, title: str, height: int = 380) -> go.Figure:
    """Plotly heatmap with row-normalised colours and count+% annotations."""
    cm_arr = np.array(cm_matrix, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = cm_arr / np.where(row_sums == 0, 1, row_sums)

    # Cell text: count + %
    text = [
        [f"<b>{int(cm_arr[i, j])}</b><br>{cm_norm[i, j]:.0%}" for j in range(3)]
        for i in range(3)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=cm_norm,
            x=CM_LABELS,
            y=CM_LABELS,
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            showscale=False,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5, xanchor="center"),
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        height=height,
        margin=dict(l=10, r=10, t=60, b=50),
    )
    return fig


# ── Overview: best model per matrix (5 columns) ───────────────────────────────
st.markdown("##### Best model per matrix — overview")
st.caption(
    "Rows = Actual class · Columns = Predicted class · "
    "Colour = row-normalised proportion (green = correct)"
)

best_per_matrix = (
    filtered.sort_values("balanced_accuracy", ascending=False)
    .groupby("sample_type")
    .first()
    .reset_index()
)

overview_cols = st.columns(5)
for col_idx, matrix in enumerate(SAMPLE_TYPES):
    row = best_per_matrix[best_per_matrix["sample_type"] == matrix]
    if row.empty:
        overview_cols[col_idx].warning(f"No data\n{matrix}")
        continue

    row = row.iloc[0]
    model_name = str(row["model"])
    tp_val = str(row["timepoints"])
    ba = row.get("balanced_accuracy", float("nan"))
    search = str(row.get("search", "grid"))

    cm_key = build_cm_key(matrix, tp_val, model_name, search)
    if cm_key not in cm_raw:
        overview_cols[col_idx].warning(f"CM missing\n{cm_key}")
        continue

    label = MATRIX_DISPLAY.get(matrix, matrix)
    title = f"<b>{label}</b><br><span style='font-size:11px'>{model_name} · BA={ba:.3f}</span>"
    fig = make_cm_figure(cm_raw[cm_key], title, height=310)
    overview_cols[col_idx].plotly_chart(fig, use_container_width=True)

# ── Detail view ───────────────────────────────────────────────────────────────
st.divider()
st.markdown("##### Detail — select matrix, timepoint & model")

d1, d2, d3 = st.columns([1, 1, 1])
with d1:
    sel_matrix = st.selectbox(
        "Matrix", SAMPLE_TYPES,
        format_func=lambda x: MATRIX_DISPLAY.get(x, x),
        key="cm_matrix_detail",
    )
with d2:
    matrix_runs = df[df["sample_type"] == sel_matrix]
    avail_tps = sorted(matrix_runs["timepoints"].dropna().unique()) if "timepoints" in matrix_runs.columns else ["[1]"]
    def _tp_label(x: str) -> str:
        s = str(x).strip()
        # DagsHub format: "tp1" = single, "tp1_2_3" = all
        # Local CSV format: "[1]" = single, "[1, 2, 3]" = all
        if s in ("tp1", "[1]", "1"):
            return "Single timepoint (T1)"
        return "All timepoints (T1+T2+T3)"

    sel_tp = st.selectbox(
        "Timepoints", avail_tps,
        format_func=_tp_label,
        key="cm_tp_detail",
    )
with d3:
    avail_models = sorted(matrix_runs["model"].dropna().unique())
    # Pre-select best model for this matrix
    best_for_matrix = (
        matrix_runs[matrix_runs["timepoints"].astype(str) == str(sel_tp)]
        .sort_values("balanced_accuracy", ascending=False)
    )
    default_model_idx = (
        avail_models.index(best_for_matrix.iloc[0]["model"])
        if not best_for_matrix.empty and best_for_matrix.iloc[0]["model"] in avail_models
        else 0
    )
    sel_model = st.selectbox(
        "Model", avail_models,
        index=default_model_idx,
        key="cm_model_detail",
    )

# Retrieve and display
sel_search = "grid"
cm_key_detail = build_cm_key(sel_matrix, str(sel_tp), sel_model, sel_search)

detail_row = df[
    (df["sample_type"] == sel_matrix)
    & (df["model"] == sel_model)
    & (df["timepoints"].astype(str) == str(sel_tp))
]

# Metrics row
if not detail_row.empty:
    dr = detail_row.iloc[0]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Balanced Accuracy", f"{dr.get('balanced_accuracy', float('nan')):.3f}")
    mc2.metric("AUC", f"{dr.get('roc_auc', float('nan')):.3f}")
    mc3.metric("MCC", f"{dr.get('mcc', float('nan')):.3f}")
    mc4.metric("F1 (weighted)", f"{dr.get('f1_weighted', float('nan')):.3f}")

if cm_key_detail in cm_raw:
    label_detail = MATRIX_DISPLAY.get(sel_matrix, sel_matrix)
    tp_friendly = "T1" if str(sel_tp).strip() in ("tp1", "[1]", "1") else "T1+T2+T3"
    title_detail = (
        f"<b>{label_detail} — {sel_model}</b>  "
        f"<span style='font-size:12px'>| Timepoints: {tp_friendly}</span>"
    )
    # Centre the figure
    _, fig_col, _ = st.columns([1, 3, 1])
    with fig_col:
        fig_detail = make_cm_figure(cm_raw[cm_key_detail], title_detail, height=480)
        st.plotly_chart(fig_detail, use_container_width=True)

    # Per-class breakdown table
    cm_arr = np.array(cm_raw[cm_key_detail], dtype=float)
    row_sums = cm_arr.sum(axis=1)
    breakdown = pd.DataFrame({
        "Class": CM_LABELS,
        "Total (test)": row_sums.astype(int),
        "Correct": [int(cm_arr[i, i]) for i in range(3)],
        "Recall": [cm_arr[i, i] / row_sums[i] if row_sums[i] > 0 else 0 for i in range(3)],
        "Misclassified as Football": [int(cm_arr[i, 0]) if i != 0 else "-" for i in range(3)],
        "Misclassified as Sedentary": [int(cm_arr[i, 1]) if i != 1 else "-" for i in range(3)],
        "Misclassified as Ultrarunning": [int(cm_arr[i, 2]) if i != 2 else "-" for i in range(3)],
    })
    st.dataframe(
        breakdown.style.format({"Recall": "{:.1%}"})
        .background_gradient(subset=["Recall"], cmap="RdYlGn", vmin=0, vmax=1),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning(
        f"No confusion matrix found for: **{MATRIX_DISPLAY.get(sel_matrix, sel_matrix)}** "
        f"/ {sel_model} / {sel_tp}.\n\n"
        f"Key tried: `{cm_key_detail}`"
    )
