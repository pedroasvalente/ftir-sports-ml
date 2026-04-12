"""
Shared appearance settings for all Streamlit pages.
Call `render_appearance_sidebar()` in any page sidebar to expose controls.
All settings persist in st.session_state across pages within the same session.
"""
import streamlit as st

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_GROUP_COLORS = {
    "S":  "#e41a1c",
    "F":  "#377eb8",
    "U":  "#4daf4a",
}

DEFAULT_GROUP_LABELS = {
    "S": "Sedentary",
    "F": "Football",
    "U": "Ultrarunning",
}

DEFAULT_MODEL_COLORS = {
    "Random Forest":   "#2166ac",
    "MLP Classifier":  "#d6604d",
    "Decision Tree":   "#4dac26",
    "XGBoost":         "#8073ac",
}

DEFAULT_MATRIX_COLORS = {
    "CAPILAR": "#1b9e77",
    "PLASMA":  "#d95f02",
    "SALIVA":  "#7570b3",
    "SERUM":   "#e7298a",
    "URINE":   "#66a61e",
}


def _init_defaults():
    """Initialise session_state with defaults if not already set."""
    if "group_colors" not in st.session_state:
        st.session_state.group_colors = dict(DEFAULT_GROUP_COLORS)
    if "group_labels" not in st.session_state:
        st.session_state.group_labels = dict(DEFAULT_GROUP_LABELS)
    if "model_colors" not in st.session_state:
        st.session_state.model_colors = dict(DEFAULT_MODEL_COLORS)
    if "matrix_colors" not in st.session_state:
        st.session_state.matrix_colors = dict(DEFAULT_MATRIX_COLORS)
    if "legend_title" not in st.session_state:
        st.session_state.legend_title = "Sport group"


def render_appearance_sidebar(show_groups=True, show_models=False, show_matrices=False):
    """
    Render colour + label controls in the current page's sidebar.
    Returns (group_colors, group_labels, model_colors, matrix_colors).
    """
    _init_defaults()

    with st.sidebar.expander("🎨 Appearance", expanded=False):
        st.session_state.legend_title = st.text_input(
            "Legend title", value=st.session_state.legend_title
        )

        if show_groups:
            st.markdown("**Group colours & labels**")
            for key in list(st.session_state.group_colors.keys()):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.session_state.group_colors[key] = st.color_picker(
                        key, value=st.session_state.group_colors[key], key=f"gc_{key}"
                    )
                with c2:
                    st.session_state.group_labels[key] = st.text_input(
                        "Label", value=st.session_state.group_labels[key],
                        key=f"gl_{key}", label_visibility="collapsed"
                    )

        if show_models:
            st.markdown("**Model colours**")
            for key in list(st.session_state.model_colors.keys()):
                st.session_state.model_colors[key] = st.color_picker(
                    key, value=st.session_state.model_colors[key], key=f"mc_{key}"
                )

        if show_matrices:
            st.markdown("**Matrix colours**")
            for key in list(st.session_state.matrix_colors.keys()):
                st.session_state.matrix_colors[key] = st.color_picker(
                    key, value=st.session_state.matrix_colors[key], key=f"matc_{key}"
                )

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <small>
        © 2024 Pedro Afonso Valente<br>
        University of Coimbra<br>
        <a href="https://github.com/pedroasvalente/ftir-sports-ml" target="_blank">
        GitHub repository</a><br>
        Licensed under CC BY-NC-ND 4.0
        </small>
        """,
        unsafe_allow_html=True,
    )

    return (
        st.session_state.group_colors,
        st.session_state.group_labels,
        st.session_state.model_colors,
        st.session_state.matrix_colors,
    )


def apply_group_labels(series, group_labels=None):
    """Rename group values in a pandas Series using the label map."""
    if group_labels is None:
        _init_defaults()
        group_labels = st.session_state.group_labels
    return series.map(lambda x: group_labels.get(str(x), str(x)))
