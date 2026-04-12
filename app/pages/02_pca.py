import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

from ftir.data.config import SAMPLE_TYPES
from ftir.data.loader import get_ftir_columns, load_data
from ftir.reduction.pca import pca_results_df, run_pca
from shared_settings import apply_group_labels, render_appearance_sidebar

st.set_page_config(page_title="PCA Explorer", layout="wide")
st.title("PCA Explorer")


@st.cache_data
def _load():
    return load_data()


@st.cache_data
def _run_pca(matrix: str, timepoints: tuple, scale: bool):
    df = _load()
    ftir_cols = get_ftir_columns(df)
    data = df[df["sample_type"] == matrix].copy()
    if timepoints:
        data = data[data["timepoint"].isin(list(timepoints))]
    data = data.dropna(subset=ftir_cols)
    X = data[ftir_cols].values.astype(float)
    scores, loadings, evr = run_pca(X, scale=scale)
    return data, scores, loadings, evr, ftir_cols


df_full = _load()
ftir_cols = get_ftir_columns(df_full)
wavenumbers = np.array([float(c) for c in ftir_cols])

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    matrix = st.selectbox("Matrix", SAMPLE_TYPES)
    timepoints = tuple(st.multiselect("Timepoints", [1, 2, 3], default=[1]))
    color_by = st.selectbox(
        "Colour by",
        [c for c in ["group_fam", "group", "timepoint", "age_years", "bodyfat_perc", "ffm_kg"]
         if c in df_full.columns],
    )
    scale = st.checkbox("Scale (StandardScaler)", value=True)

    st.header("Axes")
    pc_x = st.selectbox("X axis", [f"PC{i+1}" for i in range(10)], index=0)
    pc_y = st.selectbox("Y axis", [f"PC{i+1}" for i in range(10)], index=1)

    st.header("Overlays")
    show_ellipses = st.checkbox("95% confidence ellipses", value=True)
    show_biplot = st.checkbox("Biplot (top loadings)", value=False)
    n_biplot = st.slider("Biplot arrows", 5, 20, 10) if show_biplot else 10
    show_loadings = st.checkbox("Loadings heatmap", value=True)
    compare_matrices = st.checkbox("Compare all matrices", value=False)

    st.header("Appearance")
    plot_title = st.text_input("Plot title", value=f"PCA — {matrix}")
    marker_size = st.slider("Marker size", 4, 16, 8)
    opacity = st.slider("Opacity", 0.3, 1.0, 0.75)

group_colors, group_labels, model_colors, _ = render_appearance_sidebar(show_groups=True)
custom_colors = group_colors

data, scores, loadings, evr, ftir_cols = _run_pca(matrix, timepoints, scale)
result_df = pca_results_df(scores, data, n_components=scores.shape[1])

pc_x_idx = int(pc_x[2:]) - 1
pc_y_idx = int(pc_y[2:]) - 1

# ── Variance explained ───────────────────────────────────────────────────────
st.subheader("Variance explained")
n_show = min(15, len(evr))
fig_var = px.bar(
    x=list(range(1, n_show + 1)),
    y=evr[:n_show] * 100,
    labels={"x": "PC", "y": "Variance (%)"},
    title=f"{matrix} — Variance explained per PC",
)
fig_var.update_traces(marker_color="#2c7bb6")
fig_var.update_layout(showlegend=False, height=300)
st.plotly_chart(fig_var, use_container_width=True)

# ── Scores plot ──────────────────────────────────────────────────────────────
st.subheader(f"Scores: {pc_x} vs {pc_y}")

is_categorical = (
    result_df[color_by].dtype == object or result_df[color_by].nunique() <= 10
)

hover_cols = [c for c in ["group_fam", "person_code", "timepoint", "age_years", "bodyfat_perc"]
              if c in result_df.columns]

if is_categorical:
    plot_df = result_df.copy()
    plot_df["_label"] = apply_group_labels(plot_df[color_by], group_labels)
    renamed_colors = {group_labels.get(str(k), str(k)): v for k, v in custom_colors.items()}

    fig_sc = px.scatter(
        plot_df, x=pc_x, y=pc_y,
        color="_label",
        hover_data=hover_cols,
        title=plot_title,
        labels={
            pc_x: f"{pc_x} ({evr[pc_x_idx]*100:.1f}%)",
            pc_y: f"{pc_y} ({evr[pc_y_idx]*100:.1f}%)",
            "_label": st.session_state.get("legend_title", "Group"),
        },
        opacity=opacity,
        color_discrete_map=renamed_colors,
    )
    fig_sc.update_traces(marker_size=marker_size)

    # 95% confidence ellipses
    if show_ellipses:
        groups = plot_df["_label"].unique()
        colors = list(renamed_colors.values())
        for i, grp in enumerate(groups):
            sub = plot_df[plot_df["_label"] == grp][[pc_x, pc_y]].dropna().values
            if len(sub) < 3:
                continue
            cov = np.cov(sub.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            chi2_val = 5.991
            width, height = 2 * np.sqrt(chi2_val * eigenvalues)
            t = np.linspace(0, 2 * np.pi, 100)
            ell_x = (width / 2) * np.cos(t)
            ell_y = (height / 2) * np.sin(t)
            angle_rad = np.radians(angle)
            rot_x = ell_x * np.cos(angle_rad) - ell_y * np.sin(angle_rad) + sub.mean(0)[0]
            rot_y = ell_x * np.sin(angle_rad) + ell_y * np.cos(angle_rad) + sub.mean(0)[1]
            color = colors[i % len(colors)]
            fig_sc.add_trace(go.Scatter(
                x=rot_x, y=rot_y, mode="lines",
                line=dict(color=color, dash="dash", width=1.5),
                showlegend=False, hoverinfo="skip",
            ))
else:
    fig_sc = px.scatter(
        result_df, x=pc_x, y=pc_y,
        color=color_by, color_continuous_scale="viridis",
        hover_data=hover_cols,
        title=plot_title,
        labels={
            pc_x: f"{pc_x} ({evr[pc_x_idx]*100:.1f}%)",
            pc_y: f"{pc_y} ({evr[pc_y_idx]*100:.1f}%)",
        },
        opacity=opacity,
    )
    fig_sc.update_traces(marker_size=marker_size)

# Biplot arrows
if show_biplot:
    scale_factor = max(result_df[pc_x].abs().max(), result_df[pc_y].abs().max()) * 0.7
    top_idx = np.argsort(np.abs(loadings[pc_x_idx]) + np.abs(loadings[pc_y_idx]))[-n_biplot:]
    for idx in top_idx:
        x_end = loadings[pc_x_idx, idx] * scale_factor
        y_end = loadings[pc_y_idx, idx] * scale_factor
        fig_sc.add_annotation(
            x=x_end, y=y_end, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowcolor="gray", arrowwidth=1,
            text=f"{wavenumbers[idx]:.0f}", font=dict(size=9, color="gray"),
        )

fig_sc.add_hline(y=0, line_color="lightgray", line_width=0.8)
fig_sc.add_vline(x=0, line_color="lightgray", line_width=0.8)
fig_sc.update_layout(height=550)
st.plotly_chart(fig_sc, use_container_width=True)

# ── Loadings heatmap ─────────────────────────────────────────────────────────
if show_loadings:
    st.subheader("Loadings (top wavenumbers per PC)")
    valid_mask = (wavenumbers < 1850) | (wavenumbers > 2500)
    wn_valid = wavenumbers[valid_mask]
    load_valid = loadings[:min(6, loadings.shape[0]), :][:, valid_mask]
    n_top = 25
    top_idx = np.argsort(np.abs(load_valid).sum(axis=0))[-n_top:][::-1]

    # Sort selected wavenumbers in descending order for readability
    top_idx_sorted = top_idx[np.argsort(wn_valid[top_idx])[::-1]]
    x_labels = [f"{wn_valid[i]:.1f}" for i in top_idx_sorted]
    z_values = load_valid[:, top_idx_sorted]
    y_labels = [f"PC{i+1}" for i in range(load_valid.shape[0])]

    fig_load = go.Figure(go.Heatmap(
        z=z_values,
        x=x_labels,       # categorical — no continuous axis problem
        y=y_labels,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Loading"),
        hovertemplate="Wavenumber: %{x} cm⁻¹<br>%{y}<br>Loading: %{z:.4f}<extra></extra>",
    ))
    fig_load.update_layout(
        title=f"Top-{n_top} wavenumber loadings — {matrix}",
        xaxis=dict(title="Wavenumber (cm⁻¹)", tickangle=-45),
        yaxis=dict(title="PC"),
        height=320,
    )
    st.plotly_chart(fig_load, use_container_width=True)

# ── Compare all matrices ─────────────────────────────────────────────────────
if compare_matrices:
    st.subheader("PC1 vs PC2 — all matrices")
    cols = st.columns(len(SAMPLE_TYPES))
    for i, mat in enumerate(SAMPLE_TYPES):
        d2, s2, _, e2, _ = _run_pca(mat, timepoints, scale)
        r2 = pca_results_df(s2, d2, n_components=2)
        if color_by in r2.columns:
            fig_m = px.scatter(
                r2, x="PC1", y="PC2", color=color_by,
                title=mat, opacity=0.7,
                labels={"PC1": f"PC1 ({e2[0]*100:.0f}%)", "PC2": f"PC2 ({e2[1]*100:.0f}%)"},
            )
        else:
            fig_m = px.scatter(r2, x="PC1", y="PC2", title=mat, opacity=0.7)
        fig_m.update_traces(marker_size=5)
        fig_m.update_layout(showlegend=False, height=300, margin=dict(t=40, b=20, l=20, r=20))
        cols[i].plotly_chart(fig_m, use_container_width=True)
