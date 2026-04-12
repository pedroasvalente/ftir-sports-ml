import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ftir.data.config import DATA_COLS, SAMPLE_TYPES
from ftir.data.loader import get_ftir_columns, load_data
from shared_settings import apply_group_labels, render_appearance_sidebar

st.set_page_config(page_title="Spectra", layout="wide")
st.title("Spectra Visualisation")

# ── Atmospheric absorption region ─────────────────────────────────────────────
WATER_CO2_MIN = 1850.0
WATER_CO2_MAX = 2500.0


@st.cache_data
def _load() -> pd.DataFrame:
    return load_data()


df = _load()
ftir_cols = get_ftir_columns(df)
wavenumbers = np.array([float(c) for c in ftir_cols])

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    matrix = st.selectbox("Matrix", SAMPLE_TYPES)
    group_col = st.selectbox("Colour by", ["group", "group_fam", "timepoint"])
    timepoints = st.multiselect("Timepoints", [1, 2, 3], default=[1])
    show_individual = st.checkbox("Overlay individual spectra", value=False)

group_colors, group_labels, _, _ = render_appearance_sidebar(show_groups=True)

# ── Filter data ────────────────────────────────────────────────────────────────
data = df[df["sample_type"] == matrix].copy()
if timepoints:
    data = data[data["timepoint"].isin(timepoints)]

if data.empty:
    st.warning("No data available for this selection.")
    st.stop()

st.markdown(f"**{len(data)} samples** — {matrix}")

# ── Build mean spectra per group ───────────────────────────────────────────────
groups = sorted(data[group_col].dropna().unique())

# Extended color + label lookup: works for both short keys (S/F/U) and full names.
# group_labels maps "S"->"Sedentary", so invert it to also resolve "sedentary"->"S".
_inv_labels = {v.lower(): k for k, v in group_labels.items()}
_inv_labels.update({k.lower(): k for k in group_colors})  # ensure short keys also resolve

def _resolve_color(grp_str: str) -> str:
    """Return colour for a group value regardless of whether it is a short key or full name."""
    if grp_str in group_colors:
        return group_colors[grp_str]
    key = _inv_labels.get(grp_str.lower())
    return group_colors.get(key, "#888888")

def _resolve_label(grp_str: str) -> str:
    """Return display label for a group value."""
    if grp_str in group_labels:
        return group_labels[grp_str]
    key = _inv_labels.get(grp_str.lower())
    if key and key in group_labels:
        return group_labels[key]
    return grp_str.title()

fig = go.Figure()

# Atmospheric CO₂ / H₂O shaded region
fig.add_vrect(
    x0=WATER_CO2_MIN,
    x1=WATER_CO2_MAX,
    fillcolor="grey",
    opacity=0.15,
    line_width=0,
    annotation_text="Atmospheric CO\u2082 / H\u2082O",
    annotation_position="top left",
    annotation_font_size=11,
    annotation_font_color="grey",
)

# ── Individual spectra (optional) and mean spectra per group ──────────────────
mean_records: list[dict] = []

for grp in groups:
    grp_str = str(grp)
    label = _resolve_label(grp_str)
    color = _resolve_color(grp_str)

    sub = data[data[group_col] == grp]
    spectra = sub[ftir_cols].values.astype(float)

    # Individual spectra overlay
    if show_individual:
        for i, row_vals in enumerate(spectra):
            sample_row = sub.iloc[i]
            person = sample_row.get("person_code", "")
            tp = sample_row.get("timepoint", "")
            fig.add_trace(
                go.Scatter(
                    x=wavenumbers,
                    y=row_vals,
                    mode="lines",
                    line=dict(color=color, width=0.7),
                    opacity=0.25,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"Person: {person}<br>"
                        f"Timepoint: {tp}<br>"
                        "Wavenumber: %{x:.1f} cm\u207b\u00b9<br>"
                        "Absorbance: %{y:.4f} a.u."
                        "<extra></extra>"
                    ),
                )
            )

    # Mean spectrum
    mean_spectrum = np.nanmean(spectra, axis=0)
    fig.add_trace(
        go.Scatter(
            x=wavenumbers,
            y=mean_spectrum,
            mode="lines",
            name=label,
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{label} — mean</b><br>"
                "Wavenumber: %{x:.1f} cm\u207b\u00b9<br>"
                "Absorbance: %{y:.4f} a.u."
                "<extra></extra>"
            ),
        )
    )

    # Accumulate mean data for CSV export
    mean_records.append(
        {"group": grp_str, "group_label": label, **dict(zip(ftir_cols, mean_spectrum))}
    )

# ── Layout: inverted x-axis (standard FTIR convention) ────────────────────────
fig.update_layout(
    title=dict(text=f"Mean FTIR Spectra \u2014 {matrix}", font_size=16),
    xaxis=dict(
        title="Wavenumber (cm\u207b\u00b9)",
        autorange="reversed",
        showgrid=True,
        gridcolor="#e5e5e5",
    ),
    yaxis=dict(
        title="Absorbance (a.u.)",
        showgrid=True,
        gridcolor="#e5e5e5",
    ),
    legend=dict(title="Group"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=480,
    hovermode="x unified",
    margin=dict(l=60, r=30, t=60, b=60),
)

st.plotly_chart(fig, use_container_width=True)

# ── Download mean spectra as CSV ───────────────────────────────────────────────
if mean_records:
    mean_df = pd.DataFrame(mean_records)
    csv_bytes = mean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download mean spectra (CSV)",
        data=csv_bytes,
        file_name=f"mean_spectra_{matrix}.csv",
        mime="text/csv",
    )
