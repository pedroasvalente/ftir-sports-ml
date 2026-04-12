import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from shared_settings import render_appearance_sidebar

from ftir.data.config import DATA_COLS, SAMPLE_TYPES
from ftir.data.loader import get_ftir_columns, load_data
from ftir.visualization.plots import plot_mean_spectra

st.set_page_config(page_title="Spectra", layout="wide")
st.title("Spectra Visualisation")


@st.cache_data
def _load():
    return load_data()


df = _load()
ftir_cols = get_ftir_columns(df)

with st.sidebar:
    matrix = st.selectbox("Matrix", SAMPLE_TYPES)
    group_col = st.selectbox("Colour by", ["group_fam", "group", "timepoint"])
    timepoints = st.multiselect("Timepoints", [1, 2, 3], default=[1])
    show_individual = st.checkbox("Overlay individual spectra", value=False)

render_appearance_sidebar(show_groups=True)

data = df[df["sample_type"] == matrix].copy()
if timepoints:
    data = data[data["timepoint"].isin(timepoints)]

if data.empty:
    st.warning("No data for this selection.")
    st.stop()

st.markdown(f"**{len(data)} samples** — {matrix}")

fig = plot_mean_spectra(
    df=data,
    ftir_cols=ftir_cols,
    group_col=group_col,
    sample_type=matrix,
)
st.pyplot(fig)
plt.close()

if show_individual:
    wavenumbers = np.array([float(c) for c in ftir_cols])
    fig2, ax = plt.subplots(figsize=(12, 4))
    groups = data[group_col].unique()
    import seaborn as sns
    palette = sns.color_palette("Set1", len(groups))
    for grp, color in zip(groups, palette):
        sub = data[data[group_col] == grp][ftir_cols].values.astype(float)
        for row in sub:
            ax.plot(wavenumbers, row, color=color, alpha=0.15, linewidth=0.4)
    ax.invert_xaxis()
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance")
    ax.set_title(f"Individual spectra — {matrix}")
    st.pyplot(fig2)
    plt.close()
