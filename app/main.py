import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from shared_settings import render_appearance_sidebar

st.set_page_config(
    page_title="FTIR Sports ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    st.page_link("pages/00_overview.py",    label="📋 Data Overview")
    st.page_link("pages/01_spectra.py",     label="📈 Spectra")
    st.page_link("pages/02_pca.py",         label="🔵 PCA Explorer")
    st.page_link("pages/03_plsda.py",       label="🔴 PLS-DA")
    st.page_link("pages/04_results.py",     label="📊 ML Results")
    st.page_link("pages/05_comparison.py",  label="⚖️  Model Comparison")
    st.page_link("pages/06_diagnostics.py", label="🔬 Diagnostics")

render_appearance_sidebar()

# ── Hero header ────────────────────────────────────────────────────────────────
st.title("FTIR-Based Sport Group Discrimination — Study 1")
st.caption(
    "Supervised machine learning for physical activity level classification "
    "across five biological matrices using ATR-FTIR spectroscopy"
)

st.divider()

# ── Study context ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("Study Overview")
    st.markdown(
        """
        This dashboard presents the analytical pipeline and results for **Study 1** of a PhD
        research project investigating the potential of attenuated total reflectance Fourier-transform
        infrared (ATR-FTIR) spectroscopy as a rapid, reagent-free tool for discriminating individuals
        by physical activity level.

        **Research question:** Can FTIR spectral signatures of biological matrices differentiate
        sedentary individuals from football players and ultramarathon runners?

        **Approach:** Supervised classification models were trained on pre-processed spectral data
        from five biological matrices. Person-aware cross-validation (StratifiedGroupKFold) prevents
        data leakage across repeated measures. Feature relevance is quantified via PLS-DA variable
        importance in projection (VIP) scores.
        """
    )

with col_right:
    st.subheader("Study Design at a Glance")
    st.markdown(
        """
        | Parameter | Details |
        |-----------|---------|
        | **Sport groups** | Sedentary · Football · Ultrarunning |
        | **Biological matrices** | Capillary blood · Plasma · Saliva · Serum · Urine |
        | **Timepoints** | 1 (baseline) · 2 · 3 |
        | **Classifiers** | Random Forest · MLP · Decision Tree · XGBoost |
        | **Dimensionality reduction** | PLS-DA (10 components) |
        | **Resampling** | SMOTE inside CV loop (train only) |
        | **Train / test split** | 70 % / 30 % (person-stratified) |
        | **CV strategy** | 5-fold StratifiedGroupKFold |
        | **Primary metric** | Balanced accuracy (95 % bootstrap CI) |
        """
    )

st.divider()

# ── Getting started ────────────────────────────────────────────────────────────
st.subheader("How to Use This Dashboard")

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(
        """
        **1 — Explore the data**

        Start with **📋 Data Overview** to review participant demographics,
        sample counts per matrix and group, and spectral quality indicators.
        """
    )
with s2:
    st.markdown(
        """
        **2 — Inspect the spectra**

        Use **📈 Spectra** to visualise mean ± SD absorption profiles for each
        sport group. Toggle individual traces and compare across timepoints.
        The atmospheric CO₂/H₂O region (1 850–2 500 cm⁻¹) is marked.
        """
    )
with s3:
    st.markdown(
        """
        **3 — Dimensionality reduction**

        **🔵 PCA Explorer** reveals sample clustering and variance structure.
        **🔴 PLS-DA** shows supervised group separation and VIP scores
        identifying the most discriminant wavenumbers.
        """
    )
with s4:
    st.markdown(
        """
        **4 — Evaluate ML performance**

        **📊 ML Results** lists all runs with sortable metrics and CIs.
        **⚖️ Model Comparison** benchmarks classifiers across matrices.
        **🔬 Diagnostics** shows confusion matrices, ROC curves, and
        calibration plots for any selected run.
        """
    )

st.divider()

# ── Methodological notes ───────────────────────────────────────────────────────
with st.expander("Methodological notes", expanded=False):
    st.markdown(
        """
        ### Data leakage prevention
        Because some participants contributed samples at multiple timepoints (repeated measures),
        a naïve random split could place the same individual in both training and test sets,
        artificially inflating performance estimates. This pipeline uses **person-aware splits**:
        all samples from a given participant are assigned exclusively to either the training
        or the test partition. Cross-validation uses `StratifiedGroupKFold` with `groups = person_code`.

        ### SMOTE placement
        Synthetic Minority Over-sampling Technique (SMOTE) is applied **inside** the cross-validation
        loop using an `imblearn.pipeline.Pipeline`. This ensures that synthetic samples generated from
        minority-class observations in the training fold never appear in the validation fold,
        avoiding inflated CV scores.

        ### VIP scores
        Variable Importance in Projection (VIP) scores are computed directly from the PLS-DA
        solution using the standard formula involving x-scores, x-weights, and y-loadings.
        Wavenumbers with VIP > 1.0 are considered influential for group discrimination.

        ### Confidence intervals
        Bootstrap 95 % CIs on balanced accuracy, MCC, and ROC-AUC are estimated from
        N = 500 bootstrap resamples of the held-out test set.

        ### Timepoint configurations
        Results are reported for two configurations:
        - `[1]` — baseline timepoint only (independent samples)
        - `[1, 2, 3]` — all three timepoints pooled (repeated measures, requires GroupKFold)

        A two-sided Wilcoxon signed-rank test (paired by matrix) is used on the
        **Model Comparison** page to assess whether including additional timepoints
        significantly changes balanced accuracy.
        """
    )
