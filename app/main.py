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

render_appearance_sidebar()

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("ATR-FTIR Spectroscopy for Sport Group Discrimination")
st.caption(
    "Supervised machine learning for physical activity level classification "
    "across five biological matrices"
)

st.divider()

# ── Study overview ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("Background")
    st.markdown(
        """
        The study investigates whether attenuated total reflectance Fourier-transform infrared
        (ATR-FTIR) spectroscopy of biological matrices can discriminate individuals according
        to their physical activity level.

        Spectra from five matrices — capillary blood, plasma, saliva, serum, and urine — were
        collected from sedentary individuals, football players, and ultramarathon runners across
        up to three timepoints. Supervised classifiers were trained on PLS-DA-reduced spectral data;
        person-aware cross-validation (StratifiedGroupKFold on `person_code`) prevents data leakage
        from repeated-measures participants.

        Feature relevance is expressed as PLS-DA variable importance in projection (VIP) scores.
        Bootstrap 95 % confidence intervals on balanced accuracy, MCC, and ROC-AUC are reported
        for all models.
        """
    )

with col_right:
    st.subheader("Study parameters")
    st.markdown(
        """
        | Parameter | Details |
        |-----------|---------|
        | **Sport groups** | Sedentary · Football · Ultrarunning |
        | **Biological matrices** | Capillary blood · Plasma · Saliva · Serum · Urine |
        | **Timepoints** | 1 (baseline) · 2 · 3 |
        | **Classifiers** | Random Forest · MLP · Decision Tree · XGBoost |
        | **Dim. reduction** | PLS-DA (10 components) |
        | **Class balancing** | SMOTE inside CV fold (train only) |
        | **Train / test split** | 70 % / 30 % (person-stratified) |
        | **Cross-validation** | 5-fold StratifiedGroupKFold |
        | **Primary metric** | Balanced accuracy (95 % bootstrap CI) |
        """
    )

st.divider()

# ── Navigation guide ───────────────────────────────────────────────────────────
st.subheader("Dashboard structure")
st.markdown(
    """
    **Data Overview** — Sample counts by matrix, group, and timepoint; descriptive statistics
    (mean ± SD) for age, body fat, and fat-free mass.

    **Spectra** — Mean ± SD ATR-FTIR absorption profiles per sport group for a selected matrix.
    Individual traces can be overlaid. The atmospheric CO₂ / H₂O absorption region
    (1 850–2 500 cm⁻¹) is indicated.

    **PCA Explorer** — Principal component analysis scores coloured by group, timepoint,
    or continuous covariates; variance explained; loading heatmap; biplot.

    **PLS-DA** — Supervised group separation in latent variable space; VIP score spectrum
    with top-10 % and top-20 % thresholds; per-zone mean ± SD profiles with pairwise
    Mann–Whitney U significance annotations between sport groups.

    **ML Results** — Full results table with sortable metrics, per-matrix top-N leaderboard,
    performance heatmap, distribution boxplots, per-class sensitivity breakdown,
    and interactive confusion matrices.

    **Model Comparison** — Radar chart and grouped bar chart benchmarking all classifiers
    across metrics; scatter plot of balanced accuracy vs. MCC; cross-matrix summary;
    Wilcoxon signed-rank test comparing timepoint configurations [1] vs. [1, 2, 3].
    """
)

st.divider()

# ── Methodological notes ───────────────────────────────────────────────────────
with st.expander("Methodological notes", expanded=False):
    st.markdown(
        """
        **Data leakage prevention.**
        Because some participants contributed samples at multiple timepoints, a naïve random
        split could place the same individual in both training and test sets, inflating performance
        estimates. All samples from a given participant are assigned exclusively to either the
        training or the test partition. Cross-validation uses `StratifiedGroupKFold` with
        `groups = person_code`.

        **SMOTE placement.**
        Synthetic minority over-sampling (SMOTE) is applied inside the cross-validation loop
        via an `imblearn.pipeline.Pipeline`. Synthetic samples generated from training-fold
        observations therefore never appear in the validation fold, preventing inflated CV scores.

        **VIP scores.**
        Variable importance in projection (VIP) scores are derived directly from the PLS-DA
        solution (x-scores, x-weights, y-loadings). Wavenumbers with VIP > 1.0 are considered
        influential for group discrimination.

        **Confidence intervals.**
        Bootstrap 95 % CIs on balanced accuracy, MCC, and ROC-AUC are estimated from N = 500
        bootstrap resamples of the held-out test set.

        **Timepoint configurations.**
        Results are reported for two configurations: `[1]` (baseline only, independent samples)
        and `[1, 2, 3]` (all timepoints pooled, repeated measures — GroupKFold required).
        A two-sided Wilcoxon signed-rank test (paired by matrix) on the Model Comparison page
        assesses whether including additional timepoints significantly changes balanced accuracy.
        """
    )
