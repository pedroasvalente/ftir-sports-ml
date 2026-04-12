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
from shared_settings import METRIC_COLS_ALL, render_appearance_sidebar, render_data_source_sidebar

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

# ── Load data ─────────────────────────────────────────────────────────────────
df, use_dagshub = render_data_source_sidebar(RESULTS_DIR)
if df is None:
    st.stop()

METRIC_COLS = [c for c in METRIC_COLS_ALL if c in df.columns and df[c].notna().any()]

_, _, model_colors, _ = render_appearance_sidebar(show_models=True)

# ── Filters ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    matrix = st.selectbox("Matrix", SAMPLE_TYPES)
    if "timepoints" in df.columns:
        tp_opts = sorted(df["timepoints"].dropna().unique())
        timepoints = st.selectbox("Timepoints", tp_opts)
    else:
        timepoints = None

# ── Filter to matrix + timepoints ─────────────────────────────────────────────
sub = df[df["sample_type"] == matrix].copy()
if timepoints is not None and "timepoints" in sub.columns:
    sub = sub[sub["timepoints"] == timepoints]

if sub.empty:
    st.warning("No results found for this matrix / timepoint combination.")
    st.stop()

# ── Radar chart ───────────────────────────────────────────────────────────────
st.subheader(f"Performance radar — all metrics by model ({matrix})")

best_per_model = (
    sub.sort_values("balanced_accuracy", ascending=False)
    .groupby("model")
    .first()
    .reset_index()
)

fig_radar = go.Figure()
colors = px.colors.qualitative.Set1

for i, row in best_per_model.iterrows():
    values = [float(row[m]) if pd.notna(row.get(m)) else 0.0 for m in METRIC_COLS]
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
    title=f"{matrix} — model comparison across all classification metrics",
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Side-by-side metric bars ───────────────────────────────────────────────────
st.subheader("Classification metrics — best run per model")

melted = best_per_model.melt(
    id_vars="model", value_vars=METRIC_COLS,
    var_name="metric", value_name="value"
)

fig_bar = px.bar(
    melted, x="metric", y="value", color="model",
    barmode="group",
    title=f"{matrix} — metric comparison (best run per model)",
    labels={"value": "Score", "metric": "Metric"},
    color_discrete_map=model_colors,
    text_auto=".3f",
)
fig_bar.update_layout(height=420, yaxis_range=[0, 1.05])
st.plotly_chart(fig_bar, use_container_width=True)

# ── Scatter: balanced_accuracy vs MCC ────────────────────────────────────────
if "mcc" in sub.columns and sub["mcc"].notna().any():
    st.subheader("Balanced accuracy vs. MCC — all runs")
    fig_sc = px.scatter(
        sub, x="mcc", y="balanced_accuracy",
        color="model", symbol="model",
        hover_data=[c for c in ["model", "search", "timepoints", "n_synthetic"] if c in sub.columns],
        title=f"{matrix} — all runs",
        labels={"mcc": "Matthews Correlation Coefficient (MCC)",
                "balanced_accuracy": "Balanced Accuracy"},
        color_discrete_map=model_colors,
    )
    fig_sc.add_hline(y=0.8, line_dash="dot", line_color="gray",
                     annotation_text="Balanced accuracy = 0.80")
    fig_sc.add_vline(x=0.6, line_dash="dot", line_color="gray",
                     annotation_text="MCC = 0.60")
    fig_sc.update_traces(marker_size=10)
    fig_sc.update_layout(height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Cross-matrix summary table ────────────────────────────────────────────────
st.subheader("Cross-matrix summary — best-performing model per biological matrix")

all_best = []
for mat in SAMPLE_TYPES:
    sub_mat = df[df["sample_type"] == mat].copy()
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

# ══════════════════════════════════════════════════════════════════════════════
# Timepoint configuration comparison — Wilcoxon signed-rank test
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Timepoint configuration comparison — Wilcoxon signed-rank test")

_TP_SINGLE = "[1]"
_TP_ALL    = "[1, 2, 3]"

_required_tps = {_TP_SINGLE, _TP_ALL}
_available_tps = set(df["timepoints"].dropna().unique()) if "timepoints" in df.columns else set()

if not _required_tps.issubset(_available_tps):
    st.info(
        "Both timepoint configurations `[1]` and `[1, 2, 3]` must be present in the results "
        "to run the Wilcoxon signed-rank test. "
        f"Available configurations: {sorted(_available_tps) or 'none'}."
    )
else:
    try:
        from scipy import stats as _scipy_stats

        _df_tp1   = df[df["timepoints"] == _TP_SINGLE].copy()
        _df_tp123 = df[df["timepoints"] == _TP_ALL].copy()

        # For each model, compute best BA per matrix under each timepoint config
        # then pair them across matrices for the signed-rank test.
        _models = sorted(df["model"].dropna().unique()) if "model" in df.columns else []

        _wilcox_rows = []
        for _model in _models:
            _sub1   = _df_tp1[_df_tp1["model"] == _model]
            _sub123 = _df_tp123[_df_tp123["model"] == _model]

            # Best balanced_accuracy per matrix
            _best1 = (
                _sub1.groupby("sample_type")["balanced_accuracy"]
                .max()
                .rename("ba_tp1")
            )
            _best123 = (
                _sub123.groupby("sample_type")["balanced_accuracy"]
                .max()
                .rename("ba_tp123")
            )

            _paired = pd.concat([_best1, _best123], axis=1).dropna()
            if len(_paired) < 2:
                continue  # not enough paired observations

            _x = _paired["ba_tp1"].values
            _y = _paired["ba_tp123"].values

            _res = _scipy_stats.wilcoxon(_x, _y, alternative="two-sided")
            _w   = _res.statistic
            _p   = _res.pvalue

            if _p >= 0.05:
                _sig = "n.s."
            elif _p < 0.01:
                _sig = "**"
            else:
                _sig = "*"

            _wilcox_rows.append({
                "Model":                _model,
                "Mean BA — tp=[1]":     f"{_x.mean():.3f}",
                "Mean BA — tp=[1,2,3]": f"{_y.mean():.3f}",
                "W statistic":          f"{_w:.1f}",
                "p-value":              f"{_p:.4f}",
                "Significance":         _sig,
                # Keep numeric copies for chart
                "_ba_tp1":   _x.mean(),
                "_ba_tp123": _y.mean(),
            })

        if _wilcox_rows:
            _wdf = pd.DataFrame(_wilcox_rows)

            # Display table without internal numeric cols
            _display_cols = [
                "Model", "Mean BA — tp=[1]", "Mean BA — tp=[1,2,3]",
                "W statistic", "p-value", "Significance",
            ]
            st.dataframe(_wdf[_display_cols].set_index("Model"), use_container_width=True)

            # Bar chart comparing mean BA per model for each timepoint config
            _chart_df = pd.melt(
                _wdf[["Model", "_ba_tp1", "_ba_tp123"]],
                id_vars="Model",
                value_vars=["_ba_tp1", "_ba_tp123"],
                var_name="Timepoints",
                value_name="Mean balanced accuracy",
            )
            _chart_df["Timepoints"] = _chart_df["Timepoints"].map(
                {"_ba_tp1": "tp = [1]", "_ba_tp123": "tp = [1, 2, 3]"}
            )

            fig_wilcox = px.bar(
                _chart_df,
                x="Model",
                y="Mean balanced accuracy",
                color="Timepoints",
                barmode="group",
                color_discrete_sequence=["#2166ac", "#d6604d"],
                labels={"Model": "Classifier", "Mean balanced accuracy": "Mean balanced accuracy"},
                title="Mean balanced accuracy per model — timepoint [1] vs. [1, 2, 3] (best run per matrix)",
                text_auto=".3f",
            )
            fig_wilcox.update_layout(height=420, yaxis_range=[0, 1.05])
            st.plotly_chart(fig_wilcox, use_container_width=True)

            st.caption(
                "Statistical significance legend: n.s. p ≥ 0.05 · * p < 0.05 · ** p < 0.01  "
                "(two-sided Wilcoxon signed-rank test, paired by biological matrix)."
            )
        else:
            st.info("Insufficient paired data to run Wilcoxon signed-rank tests.")

    except ImportError:
        st.error("scipy is required for the Wilcoxon signed-rank test (`pip install scipy`).")
    except Exception as _exc:
        st.error(f"Could not compute Wilcoxon tests: {_exc}")
