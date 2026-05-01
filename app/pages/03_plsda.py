import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ftir.data.config import SAMPLE_TYPES
from ftir.data.loader import get_ftir_columns, load_data
from ftir.reduction.pls_da import PLSDA
from shared_settings import apply_group_labels, render_appearance_sidebar

st.set_page_config(page_title="PLS-DA", layout="wide")
st.title("PLS-DA + VIP Scores")


@st.cache_data
def _load():
    return load_data()


@st.cache_data
def _run_plsda(matrix: str, timepoints: tuple, n_components: int, target_col: str):
    df = _load()
    ftir_cols = get_ftir_columns(df)
    wavenumbers = np.array([float(c) for c in ftir_cols])

    data = df[df["sample_type"] == matrix].copy()
    if timepoints:
        data = data[data["timepoint"].isin(list(timepoints))]
    data = data.dropna(subset=ftir_cols + [target_col])

    X = data[ftir_cols].values.astype(float)
    y_raw = data[target_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_sc = StandardScaler().fit_transform(X)
    pls = PLSDA(n_components=n_components)
    scores = pls.fit_transform(X_sc, y)
    vip = pls.vip_scores()

    return data, scores, vip, wavenumbers, list(le.classes_), y


df_full = _load()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    matrix = st.selectbox("Matrix", SAMPLE_TYPES)
    timepoints = tuple(st.multiselect("Timepoints", [1, 2, 3], default=[1]))
    n_components = st.slider("PLS components", 2, 10, 3)
    target_col = st.selectbox("Target", ["group_fam", "bodyfat_classes_simplified"])
    top_n_vip = st.slider("Top-N VIP wavenumbers", 10, 50, 20)

    st.divider()
    st.subheader("VIP Regions Table")
    vip_pct_threshold = st.slider("Top % threshold", 5, 30, 20,
                                  help="Select wavenumbers in the top N% of VIP scores")
    min_consecutive   = st.slider("Min. consecutive points", 3, 20, 5,
                                  help="Minimum consecutive points to define a region (~2.78 cm⁻¹ per point)")

    st.header("Appearance")
    plot_title = st.text_input("Scores plot title", value=f"PLS-DA — {matrix}")
    marker_size = st.slider("Marker size", 4, 16, 8)
    opacity = st.slider("Opacity", 0.3, 1.0, 0.75)
    show_ellipses = st.checkbox("95% confidence ellipses", value=True)
    lv_x = st.selectbox("X axis", [f"LV{i+1}" for i in range(n_components)], index=0)
    lv_y = st.selectbox("Y axis", [f"LV{i+1}" for i in range(n_components)], index=1)


group_colors, group_labels, model_colors, _ = render_appearance_sidebar(show_groups=True)

try:
    data, scores, vip, wavenumbers, classes, y = _run_plsda(
        matrix, timepoints, n_components, target_col
    )
except Exception as e:
    st.error(str(e))
    st.stop()

lv_x_idx = int(lv_x[2:]) - 1
lv_y_idx = int(lv_y[2:]) - 1

valid_mask = (wavenumbers < 1850) | (wavenumbers > 2500)
wn_valid = wavenumbers[valid_mask]
vip_valid = vip[valid_mask]

# Build scores dataframe
scores_df = pd.DataFrame(
    scores, columns=[f"LV{i+1}" for i in range(scores.shape[1])]
)
for col in ["group_fam", "person_code", "timepoint", "age_years", "bodyfat_perc"]:
    if col in data.columns:
        scores_df[col] = data[col].values
scores_df["class_label"] = [classes[i] for i in y]
scores_df["group_display"] = apply_group_labels(scores_df["class_label"], group_labels)
renamed_colors = {group_labels.get(str(k), str(k)): v for k, v in group_colors.items()}

hover_cols = [c for c in ["person_code", "timepoint", "age_years", "bodyfat_perc"]
              if c in scores_df.columns]

col1, col2 = st.columns(2)

# ── Scores plot ───────────────────────────────────────────────────────────────
with col1:
    st.subheader(f"Scores: {lv_x} vs {lv_y}")
    fig_sc = px.scatter(
        scores_df, x=lv_x, y=lv_y,
        color="group_display",
        hover_data=hover_cols,
        title=plot_title,
        labels={lv_x: lv_x, lv_y: lv_y,
                "group_display": st.session_state.get("legend_title", "Group")},
        opacity=opacity,
        color_discrete_map=renamed_colors,
    )
    fig_sc.update_traces(marker_size=marker_size)

    if show_ellipses:
        for i, cls in enumerate(scores_df["group_display"].unique()):
            color = renamed_colors.get(cls, px.colors.qualitative.Set1[i % 9])
            sub = scores_df[scores_df["group_display"] == cls][[lv_x, lv_y]].dropna().values
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
            fig_sc.add_trace(go.Scatter(
                x=rot_x, y=rot_y, mode="lines",
                line=dict(color=color, dash="dash", width=1.5),
                showlegend=False, hoverinfo="skip",
            ))

    fig_sc.add_hline(y=0, line_color="lightgray", line_width=0.8)
    fig_sc.add_vline(x=0, line_color="lightgray", line_width=0.8)
    fig_sc.update_layout(height=480, legend_title=target_col)
    st.plotly_chart(fig_sc, use_container_width=True)

# ── VIP top-N bar chart ───────────────────────────────────────────────────────
with col2:
    st.subheader(f"Top-{top_n_vip} VIP scores")
    top_idx = np.argsort(vip_valid)[-top_n_vip:][::-1]
    top_wn = wn_valid[top_idx]
    top_vip = vip_valid[top_idx]

    vip_df = pd.DataFrame({
        "wavenumber": [f"{w:.1f}" for w in top_wn],
        "VIP": top_vip,
        "important": top_vip >= 1,
    })
    fig_vip = px.bar(
        vip_df, x="VIP", y="wavenumber",
        orientation="h",
        color="important",
        color_discrete_map={True: "#d73027", False: "#4575b4"},
        hover_data={"VIP": ":.3f", "important": False},
        title=f"VIP scores — {matrix} ({target_col})",
        labels={"wavenumber": "Wavenumber (cm⁻¹)", "VIP": "VIP score"},
    )
    fig_vip.add_vline(x=1.0, line_dash="dash", line_color="gray",
                      annotation_text="VIP = 1", annotation_position="top right")
    fig_vip.update_layout(
        height=480, showlegend=False,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_vip, use_container_width=True)

# ── VIP spectrum ──────────────────────────────────────────────────────────────
st.subheader("VIP spectrum (full range, atmospheric CO₂ region excluded)")

# Left panel: >2500 | Right panel: <1850
fig_spec = go.Figure()
for mask, name in [
    (wavenumbers > 2500, "High wavenumber"),
    (wavenumbers < 1850, "Fingerprint region"),
]:
    wn = wavenumbers[mask]
    vi = vip[mask]
    fig_spec.add_trace(go.Scatter(
        x=wn, y=vi, mode="lines", name=name,
        line=dict(color="#2c7bb6", width=1),
        hovertemplate="<b>%{x:.1f} cm⁻¹</b><br>VIP = %{y:.3f}<extra></extra>",
    ))
    # Shade VIP > 1
    above = vi >= 1
    if above.any():
        fig_spec.add_trace(go.Scatter(
            x=np.concatenate([wn, wn[::-1]]),
            y=np.concatenate([np.where(above, vi, 1), np.ones(len(wn))]),
            fill="toself", fillcolor="rgba(215,48,39,0.25)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

fig_spec.add_hline(y=1, line_dash="dash", line_color="gray",
                   annotation_text="VIP = 1")
fig_spec.update_xaxes(autorange="reversed")
fig_spec.update_layout(
    height=350,
    title=f"{matrix} — VIP spectrum ({target_col})",
    xaxis_title="Wavenumber (cm⁻¹)",
    yaxis_title="VIP score",
    hovermode="x unified",
)
st.plotly_chart(fig_spec, use_container_width=True)

n_important = int((vip_valid >= 1).sum())
st.caption(
    f"**{n_important} / {len(vip_valid)} wavenumbers** with VIP > 1 (above-average contribution). "
    "VIP scores are computed directly from the PLS-DA model, replacing the previous back-projection approach."
)

# ── VIP Regions Table ─────────────────────────────────────────────────────────
st.subheader("Key Spectral Regions (VIP)")
st.caption(
    f"Contiguous regions in the top {vip_pct_threshold}% of VIP scores "
    f"with ≥ {min_consecutive} consecutive points (~{min_consecutive * 2.78:.0f} cm⁻¹). "
    "Atmospheric CO₂ / water vapour region (1850–2500 cm⁻¹) excluded."
)

BAND_ASSIGNMENTS = [
    (929,  1000, "C–O–C ring deformation",           "Polysaccharides / glycoproteins"),
    (1000, 1080, "C–O stretch / phosphodiester",      "Carbohydrates, nucleic acids"),
    (1080, 1200, "C–O / P=O symmetric stretch",       "Carbohydrates, phospholipids"),
    (1200, 1300, "P=O asymmetric stretch",             "Phospholipids, DNA/RNA backbone"),
    (1300, 1400, "C–N / CH₂ wag (Amide III)",         "Proteins (Amide III)"),
    (1400, 1480, "CH₂/CH₃ bending",                   "Lipids, fatty acids"),
    (1480, 1600, "N–H bend + C–N (Amide II)",          "Proteins (Amide II)"),
    (1600, 1700, "C=O stretch (Amide I)",              "Proteins (Amide I)"),
    (1700, 1800, "C=O stretch (esters / acids)",       "Lipids, fatty acids"),
    (2500, 2620, "S–H stretch",                        "Thiols / cysteine residues"),
    (2620, 2800, "Overtone region",                    "—"),
    (2800, 2870, "CH₂ symmetric stretch",              "Lipids, fatty acids"),
    (2870, 2960, "CH₃ asymmetric stretch",             "Lipids, proteins"),
    (2960, 3051, "C–H aromatic / overtone",            "Aromatic amino acids"),
]

def _assign(lo, hi):
    best, best_overlap = ("—", "—"), 0
    for blo, bhi, band, bio in BAND_ASSIGNMENTS:
        ov = min(hi, bhi) - max(lo, blo)
        if ov > best_overlap:
            best_overlap, best = ov, (band, bio)
    return best

def _find_regions(wn, vip, pct=20, min_pts=5):
    water_mask = (wn < 1850) | (wn > 2500)
    wn_f, vip_f = wn[water_mask], vip[water_mask]
    order = np.argsort(wn_f)
    wn_s, vip_s = wn_f[order], vip_f[order]
    threshold = np.percentile(vip_s, 100 - pct)
    above = vip_s >= threshold
    rows, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= min_pts:
                band, bio = _assign(wn_s[i], wn_s[j - 1])
                rows.append({
                    "Region (cm⁻¹)":    f"{wn_s[i]:.0f} – {wn_s[j-1]:.0f}",
                    "Width (cm⁻¹)":     round(wn_s[j-1] - wn_s[i], 1),
                    "Max VIP":          round(float(vip_s[i:j].max()), 3),
                    "Mean VIP":         round(float(vip_s[i:j].mean()), 3),
                    "Band assignment":  band,
                    "Biochemical origin": bio,
                })
            i = j
        else:
            i += 1
    return rows

regions = _find_regions(wavenumbers, vip, pct=vip_pct_threshold, min_pts=min_consecutive)

if regions:
    reg_df = pd.DataFrame(regions)
    st.dataframe(
        reg_df.style.background_gradient(subset=["Max VIP", "Mean VIP"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
    )
    csv_bytes = reg_df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download table (CSV)",
        data=csv_bytes,
        file_name=f"vip_regions_{matrix}_top{vip_pct_threshold}pct_min{min_consecutive}pts.csv",
        mime="text/csv",
    )
else:
    st.info("No regions found with the current settings — try lowering the threshold or minimum points.")
