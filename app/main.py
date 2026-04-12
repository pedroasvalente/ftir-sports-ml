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

st.title("FTIR Sports ML — Study 1")
st.markdown(
    """
    Supervised ML for sport group discrimination (sedentary / football / ultrarunning)
    across five biological matrices using FTIR spectroscopy.

    Use the sidebar to navigate between analysis pages.
    """
)

with st.sidebar:
    st.header("Navigation")
    st.page_link("pages/01_spectra.py", label="Spectra")
    st.page_link("pages/02_pca.py", label="PCA Explorer")
    st.page_link("pages/03_plsda.py", label="PLS-DA")
    st.page_link("pages/04_results.py", label="ML Results")
    st.page_link("pages/05_comparison.py", label="Model Comparison")

render_appearance_sidebar()
