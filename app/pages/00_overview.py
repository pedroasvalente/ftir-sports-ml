import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ftir.data.loader import load_data, get_ftir_columns
from ftir.data.config import SAMPLE_TYPES
from shared_settings import render_appearance_sidebar

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("Data Overview & Participant Characteristics")

# ── Appearance sidebar ─────────────────────────────────────────────────────────
group_colors, group_labels, _, matrix_colors = render_appearance_sidebar(
    show_groups=True, show_matrices=True
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def _load():
    df = load_data()
    ftir_cols = get_ftir_columns(df)
    return df, ftir_cols

df, ftir_cols = _load()

# ── Section 1: Sample count summary table ─────────────────────────────────────
st.subheader("Sample count by matrix × sport group × timepoint")

if {"sample_type", "group_fam", "timepoint"}.issubset(df.columns):
    summary_pivot = (
        df.groupby(["sample_type", "group_fam", "timepoint"])
        .size()
        .reset_index(name="n_samples")
        .pivot_table(
            index=["sample_type", "group_fam"],
            columns="timepoint",
            values="n_samples",
            fill_value=0,
        )
    )
    summary_pivot.columns = [f"Timepoint {c}" for c in summary_pivot.columns]
    summary_pivot["Total"] = summary_pivot.sum(axis=1)
    st.dataframe(
        summary_pivot.style.background_gradient(
            cmap="Blues", subset=summary_pivot.columns.tolist()
        ).format("{:.0f}"),
        use_container_width=True,
    )
else:
    st.warning("Columns `sample_type`, `group_fam`, or `timepoint` not found in dataset.")

# ── Section 2: Bar chart — sample counts per matrix ───────────────────────────
st.subheader("Sample counts per biological matrix")

counts = df["sample_type"].value_counts().reset_index()
counts.columns = ["matrix", "count"]
counts["matrix"] = pd.Categorical(counts["matrix"], categories=SAMPLE_TYPES, ordered=True)
counts = counts.sort_values("matrix")

fig_counts = px.bar(
    counts,
    x="matrix",
    y="count",
    color="matrix",
    color_discrete_map=matrix_colors,
    labels={"matrix": "Biological matrix", "count": "Number of samples"},
    title="Total samples per biological matrix",
    text_auto=True,
)
fig_counts.update_layout(showlegend=False, height=380)
st.plotly_chart(fig_counts, use_container_width=True)

# ── Section 4: Descriptive statistics table ───────────────────────────────────
st.subheader("Descriptive statistics per sport group (mean ± SD)")

stat_cols = [c for c in ["age_years", "bodyfat_perc", "ffm_kg"] if c in df.columns]
if stat_cols and "group_fam" in df.columns:
    rows = []
    for grp, sub in df.groupby("group_fam"):
        entry = {"Group": group_labels.get(str(grp), str(grp)), "n": len(sub)}
        for c in stat_cols:
            vals = sub[c].dropna()
            entry[c] = f"{vals.mean():.1f} ± {vals.std():.1f}" if len(vals) > 0 else "—"
        rows.append(entry)
    stat_df = pd.DataFrame(rows)
    col_rename = {
        "age_years": "Age (years)",
        "bodyfat_perc": "Body fat (%)",
        "ffm_kg": "Fat-free mass (kg)",
    }
    stat_df = stat_df.rename(columns=col_rename)
    st.dataframe(stat_df.set_index("Group"), use_container_width=True)
else:
    st.info("Demographic columns not available for descriptive statistics.")

# ── Section 5: Spectral data quality ──────────────────────────────────────────
st.subheader("Spectral data quality — proportion of non-zero FTIR values per matrix")

if ftir_cols and "sample_type" in df.columns:
    quality_rows = []
    for mat in SAMPLE_TYPES:
        sub = df[df["sample_type"] == mat][ftir_cols]
        if sub.empty:
            continue
        nonzero_frac = (sub != 0).all(axis=0).mean() * 100  # % of wavenumbers with no zeros
        n_samples = len(sub)
        quality_rows.append(
            {"matrix": mat, "non_zero_pct": nonzero_frac, "n_samples": n_samples}
        )

    if quality_rows:
        q_df = pd.DataFrame(quality_rows)
        fig_qual = px.bar(
            q_df,
            x="matrix",
            y="non_zero_pct",
            color="matrix",
            color_discrete_map=matrix_colors,
            labels={
                "matrix": "Biological matrix",
                "non_zero_pct": "Non-zero wavenumbers (%)",
            },
            title="Percentage of FTIR wavenumber columns with no zero values per matrix",
            text_auto=".1f",
            hover_data=["n_samples"],
        )
        fig_qual.update_layout(showlegend=False, height=380, yaxis_range=[0, 105])
        st.plotly_chart(fig_qual, use_container_width=True)
    else:
        st.info("No spectral quality data available.")
else:
    st.info("No FTIR columns detected or `sample_type` column missing.")
