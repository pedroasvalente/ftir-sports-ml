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
    (1200, 1300, "P=O asymmetric stretch",             "Phospholipids"),
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

# ── Spectral Region Analysis (zone panels) ────────────────────────────────────
if regions:
    from scipy import stats as _scipy_stats
    from scipy.integrate import trapezoid as _trapz
    import plotly.subplots as _sp

    st.divider()
    st.subheader("Spectral Region Analysis")
    st.caption(
        "Mean ± SD spectra and AUC distributions per VIP region. "
        "Pairwise comparisons via two-sided Mann–Whitney U test."
    )

    ftir_cols = get_ftir_columns(data)
    X_raw     = data[ftir_cols].values.astype(float)
    grp_arr   = np.array([classes[i] for i in y])
    cls_sorted = sorted(set(grp_arr))

    disp_name  = {c: group_labels.get(str(c), str(c)) for c in cls_sorted}
    # Robust color lookup: works regardless of whether raw values are 'F' or 'football'
    _p1 = dict(group_colors)                                             # 'F' → color
    _p2 = {group_labels.get(str(k), str(k)): v for k, v in group_colors.items()}  # 'Football' → color
    _p3 = {k.lower(): v for k, v in _p2.items()}                        # 'football' → color
    _fb = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00", "#984ea3"]       # final fallback
    clr = {}
    for _i, c in enumerate(cls_sorted):
        _s = str(c)
        clr[c] = (_p1.get(_s) or _p2.get(_s) or _p3.get(_s.lower()) or _fb[_i % len(_fb)])

    def _hex_rgba(h, a=0.18):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    def _zone_aucs(X, wn, lo, hi):
        """AUC via trapezoidal integration — sorts wavenumbers ascending first."""
        msk = (wn >= lo) & (wn <= hi)
        if msk.sum() < 2:
            return np.zeros(X.shape[0])
        srt  = np.argsort(wn[msk])
        wn_s = wn[msk][srt]
        X_s  = X[:, msk][:, srt]
        return np.array([np.abs(_trapz(X_s[i], wn_s)) for i in range(X.shape[0])])

    def _sig(p):
        if p < 0.0001: return "****"
        if p < 0.001:  return "***"
        if p < 0.01:   return "**"
        if p < 0.05:   return "*"
        return "ns"

    def _parse_region(s):
        parts = s.replace("–", "-").replace("−", "-").split("-")
        return float(parts[0].strip()), float(parts[1].strip())

    ZONE_LETTERS = list("ABCDEFGHIJ")

    # ── Overview: mean spectra + neutral zone shading + letter labels ──────────
    fig_ov = go.Figure()
    for cls in cls_sorted:
        idx_c = np.where(grp_arr == cls)[0]
        for seg_mask, show_leg in [
            (wavenumbers < 1850, True),
            (wavenumbers > 2500, False),
        ]:
            wn_s = wavenumbers[seg_mask]
            mn_s = X_raw[np.ix_(idx_c, seg_mask)].mean(axis=0)
            fig_ov.add_trace(go.Scatter(
                x=wn_s, y=mn_s, mode="lines",
                name=disp_name[cls],
                line=dict(color=clr[cls], width=2),
                legendgroup=cls, showlegend=show_leg,
                hovertemplate=f"{disp_name[cls]}: %{{y:.4f}}<extra></extra>",
            ))

    for z_i, reg in enumerate(regions):
        lo, hi = _parse_region(reg["Region (cm⁻¹)"])
        fig_ov.add_vrect(
            x0=lo, x1=hi,
            fillcolor="rgba(160,160,160,0.20)",
            layer="below", line_width=0,
        )
        fig_ov.add_annotation(
            x=(lo + hi) / 2, y=1.08, yref="paper",
            text=f"<b>{ZONE_LETTERS[z_i]}</b>",
            showarrow=False, font=dict(size=12, color="#444"),
        )

    fig_ov.update_xaxes(
        autorange="reversed", title="Wavenumber (cm⁻¹)",
        rangebreaks=[dict(bounds=[1851, 2499])],
    )
    fig_ov.update_yaxes(title="Absorbance")
    fig_ov.update_layout(
        height=300,
        title=f"Mean spectra — {matrix} ({target_col}) · VIP zones",
        margin=dict(t=55, b=40),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig_ov, use_container_width=True)

    # ── Pre-compute all zones and filter out all-ns ───────────────────────────
    zone_data = []
    for z_i, reg in enumerate(regions):
        lo, hi  = _parse_region(reg["Region (cm⁻¹)"])
        z_mask  = (wavenumbers >= lo) & (wavenumbers <= hi)
        wn_z    = np.sort(wavenumbers[z_mask])
        X_z     = X_raw[:, z_mask][:, np.argsort(wavenumbers[z_mask])]
        aucs_z  = _zone_aucs(X_raw, wavenumbers, lo, hi)
        per_auc  = {c: aucs_z[grp_arr == c] for c in cls_sorted}
        per_spec = {c: X_z[grp_arr == c]    for c in cls_sorted}

        pw_res = []
        for pi, pj in [(0,1),(0,2),(1,2)]:
            a, b = per_auc[cls_sorted[pi]], per_auc[cls_sorted[pj]]
            if len(a) >= 3 and len(b) >= 3:
                _, p = _scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
                pw_res.append((cls_sorted[pi], cls_sorted[pj], _sig(p), p))

        # Skip zones where every comparison is non-significant
        if pw_res and all(lbl == "ns" for _, _, lbl, _ in pw_res):
            continue

        zone_data.append(dict(
            z_i=z_i, lo=lo, hi=hi,
            letter=ZONE_LETTERS[z_i],
            band_nm=reg["Band assignment"],
            bio_nm=reg["Biochemical origin"],
            wn_z=wn_z, X_z=X_z, aucs_z=aucs_z,
            per_auc=per_auc, per_spec=per_spec,
            pw_res=pw_res,
        ))

    if not zone_data:
        st.info("No regions with significant group differences found at current settings.")
    else:
        st.caption(
            f"Showing {len(zone_data)} zone(s) with ≥ 1 significant pairwise difference "
            f"(all-ns zones hidden)."
        )

    # ── Zone panels (2-column grid, only significant) ─────────────────────────
    for row_i in range((len(zone_data) + 1) // 2):
        grid = st.columns(2)
        for col_j in range(2):
            zd_idx = row_i * 2 + col_j
            if zd_idx >= len(zone_data):
                break

            zd      = zone_data[zd_idx]
            lo, hi  = zd["lo"], zd["hi"]
            letter  = zd["letter"]
            band_nm = zd["band_nm"]
            bio_nm  = zd["bio_nm"]
            wn_z    = zd["wn_z"]
            aucs_z  = zd["aucs_z"]
            per_auc = zd["per_auc"]
            per_spec= zd["per_spec"]
            pw_res  = zd["pw_res"]

            fig_z = _sp.make_subplots(
                rows=1, cols=2,
                column_widths=[0.44, 0.56],
                horizontal_spacing=0.08,
                subplot_titles=["Mean ± SD spectrum", "AUC by group"],
            )

            # ── Left: mean ± SD spectra (group colours) ───────────────────────
            for cls in cls_sorted:
                sp_arr = per_spec[cls]
                mn, sd = sp_arr.mean(axis=0), sp_arr.std(axis=0)
                c_hex  = clr[cls]
                dn     = disp_name[cls]
                fig_z.add_trace(go.Scatter(
                    x=np.concatenate([wn_z, wn_z[::-1]]),
                    y=np.concatenate([mn + sd, (mn - sd)[::-1]]),
                    fill="toself", fillcolor=_hex_rgba(c_hex, 0.18),
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ), row=1, col=1)
                fig_z.add_trace(go.Scatter(
                    x=wn_z, y=mn, mode="lines",
                    name=dn, line=dict(color=c_hex, width=2.5),
                    legendgroup=cls, showlegend=True,
                    hovertemplate=f"{dn}: %{{y:.5f}}<extra></extra>",
                ), row=1, col=1)

            # ── Right: violin plots (group colours) ───────────────────────────
            for cls in cls_sorted:
                c_hex = clr[cls]
                dn    = disp_name[cls]
                vals  = per_auc[cls]
                fig_z.add_trace(go.Violin(
                    x=[dn] * len(vals), y=vals,
                    name=dn, legendgroup=cls, showlegend=False,
                    line_color=c_hex,
                    fillcolor=_hex_rgba(c_hex, 0.50),
                    box_visible=True,
                    meanline_visible=True,
                    points="all", jitter=0.25, pointpos=0,
                    marker=dict(color=c_hex, size=4, opacity=0.70),
                ), row=1, col=2)

            # ── Significance brackets ─────────────────────────────────────────
            y_top  = float(np.percentile(aucs_z, 99))
            y_bot  = float(np.percentile(aucs_z,  1))
            y_span = max(y_top - y_bot, 1e-12)
            step   = y_span * 0.22
            dn_list = [disp_name[c] for c in cls_sorted]

            for b_i, (ca, cb, lbl, _) in enumerate(pw_res):
                y_br  = y_top + step * (b_i + 1.2)
                xa, xb = disp_name[ca], disp_name[cb]
                xi, xj = dn_list.index(xa), dn_list.index(xb)
                x_mid  = dn_list[(xi + xj) // 2]
                for kw in [
                    dict(x0=xa, x1=xb, y0=y_br,            y1=y_br),
                    dict(x0=xa, x1=xa, y0=y_br - step*0.1, y1=y_br),
                    dict(x0=xb, x1=xb, y0=y_br - step*0.1, y1=y_br),
                ]:
                    fig_z.add_shape(type="line", row=1, col=2,
                                    line=dict(color="#222", width=1.2), **kw)
                fig_z.add_annotation(
                    x=x_mid, y=y_br + step * 0.15,
                    xref="x2", yref="y2",
                    text=f"<b>{lbl}</b>",
                    showarrow=False, font=dict(size=12, color="#111"),
                )

            fig_z.update_xaxes(autorange="reversed", title="Wavenumber (cm⁻¹)", row=1, col=1)
            fig_z.update_yaxes(title="Absorbance", row=1, col=1)
            fig_z.update_yaxes(
                title="AUC",
                range=[y_bot - y_span*0.05,
                       y_top + step*(len(pw_res) + 2.2)],
                row=1, col=2,
            )
            fig_z.update_layout(
                title=dict(
                    text=(
                        f"<b>Zone {letter}  ·  {lo:.0f}–{hi:.0f} cm⁻¹  ·  {band_nm}</b><br>"
                        f"<span style='font-size:10px;color:#666'>{bio_nm}</span>"
                    ),
                    font=dict(size=13), x=0.5, xanchor="center",
                ),
                height=400,
                margin=dict(t=75, b=45, l=60, r=20),
                paper_bgcolor="white",
                plot_bgcolor="white",
                violingap=0.25, violingroupgap=0.1,
                # Legend inside spectrum subplot (top-left), not overlapping title
                legend=dict(
                    x=0.01, y=0.99,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="#ddd", borderwidth=1,
                    font=dict(size=10),
                ),
            )

            grid[col_j].plotly_chart(fig_z, use_container_width=True)
            dl_df = pd.DataFrame({
                "group_code":    grp_arr,
                "group_display": [disp_name[c] for c in grp_arr],
                "AUC":           aucs_z,
            })
            grid[col_j].download_button(
                label=f"⬇ Zone {letter} AUC data (CSV)",
                data=dl_df.to_csv(index=False).encode(),
                file_name=f"zone_{letter}_{matrix}_{lo:.0f}_{hi:.0f}_auc.csv",
                mime="text/csv",
                key=f"dl_zone_{zd['z_i']}_{matrix}_{vip_pct_threshold}",
            )

# ── Confounder Analysis (ANCOVA) ──────────────────────────────────────────────
st.divider()
st.subheader("🔬 Confounder Analysis (ANCOVA)")
st.caption(
    "Tests whether **age** and **body fat %** explain PLS-DA scores beyond group membership. "
    "A small ΔR² indicates that demographic variables account for negligible additional variance, "
    "confirming that group separation is not a demographic artefact."
)

try:
    import statsmodels.formula.api as smf

    demo_cols = ["age_years", "bodyfat_perc"]
    n_lv = min(3, scores.shape[1])

    # Build a working dataframe with scores + demographics + group label
    anc_base = scores_df.copy()
    for dc in demo_cols:
        if dc not in anc_base.columns and dc in data.columns:
            anc_base[dc] = data[dc].values
    for lv in range(n_lv):
        anc_base[f"LV{lv+1}"] = scores[:, lv]
    anc_base["grp"] = [classes[i] for i in y]

    available_demo = [dc for dc in demo_cols if dc in anc_base.columns]

    if not available_demo:
        st.info("No demographic columns (age_years / bodyfat_perc) found in the current data slice.")
    else:
        anc_rows = []
        for lv in range(n_lv):
            col = f"LV{lv+1}"
            d = anc_base[["grp", col] + available_demo].dropna()
            if len(d) < 20:
                continue

            m1 = smf.ols(f"{col} ~ C(grp)", data=d).fit()
            m2_formula = f"{col} ~ C(grp) + " + " + ".join(available_demo)
            m2 = smf.ols(m2_formula, data=d).fit()

            delta = m2.rsquared - m1.rsquared
            anc_rows.append({
                "Component":                 col,
                "R² — group only":           round(m1.rsquared, 3),
                "R² — group + covariates":   round(m2.rsquared, 3),
                "ΔR² (age + body fat)":      round(delta, 3),
                "Interpretation":            (
                    "Negligible confounding" if delta < 0.05
                    else "Moderate — group still dominant" if delta < 0.15
                    else "Substantial — interpret with caution"
                ),
                "n": int(len(d)),
            })

        if anc_rows:
            anc_df = pd.DataFrame(anc_rows)

            st.dataframe(
                anc_df.style
                    .background_gradient(subset=["R² — group only"],         cmap="Blues")
                    .background_gradient(subset=["ΔR² (age + body fat)"],    cmap="Reds")
                    .format({
                        "R² — group only":         "{:.3f}",
                        "R² — group + covariates": "{:.3f}",
                        "ΔR² (age + body fat)":    "{:.3f}",
                    }),
                hide_index=True,
                use_container_width=True,
            )

            # ── Stacked bar chart ─────────────────────────────────────────────
            fig_anc = go.Figure()
            shown_labels = set()
            for row in anc_rows:
                for name, val, colour in [
                    ("R² — group",                  row["R² — group only"],        "#2166ac"),
                    ("ΔR² — age + body fat",         row["ΔR² (age + body fat)"],   "#d6604d"),
                    ("Unexplained",                  max(0, 1 - row["R² — group + covariates"]), "#d9d9d9"),
                ]:
                    fig_anc.add_trace(go.Bar(
                        name=name,
                        x=[row["Component"]],
                        y=[val],
                        marker_color=colour,
                        showlegend=(name not in shown_labels),
                    ))
                    shown_labels.add(name)

            fig_anc.update_layout(
                barmode="stack",
                height=340,
                title=f"Variance decomposition of PLS-DA scores — {matrix} ({target_col})",
                yaxis_title="R²",
                yaxis_range=[0, 1.0],
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                bargap=0.35,
            )
            st.plotly_chart(fig_anc, use_container_width=True)

            max_delta = anc_df["ΔR² (age + body fat)"].max()
            st.caption(
                f"Maximum ΔR² across the first {n_lv} LVs: **{max_delta:.3f}**. "
                + (
                    "Age and body fat explain negligible additional variance beyond group — "
                    "spectral separation is not a demographic artefact."
                    if max_delta < 0.05 else
                    "Some demographic variance present; group membership remains the dominant predictor."
                )
            )
        else:
            st.info("Insufficient data (< 20 observations) for ANCOVA in the current selection.")

except ImportError:
    st.info("statsmodels not installed — run `pip install statsmodels` to enable ANCOVA.")
