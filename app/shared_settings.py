"""
Shared appearance settings and data loading for all Streamlit pages.
Call `render_appearance_sidebar()` in any page sidebar to expose controls.
All settings persist in st.session_state across pages within the same session.
"""
import os
from pathlib import Path

import pandas as pd
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


DAGSHUB_MLFLOW_URI = "https://dagshub.com/pedroasvalente/ftir-sports-ml.mlflow"
METRIC_COLS_ALL = ["balanced_accuracy", "mcc", "cohen_kappa", "f1_weighted", "f1_macro", "roc_auc"]


@st.cache_data(ttl=300)
def load_from_dagshub(token: str) -> pd.DataFrame:
    """Fetch all child runs from DagsHub MLflow REST API (no mlflow client needed)."""
    import requests

    base = DAGSHUB_MLFLOW_URI + "/api/2.0/mlflow"
    auth = ("pedroasvalente", token)
    headers = {"Content-Type": "application/json"}

    resp = requests.get(f"{base}/experiments/search", auth=auth, params={"max_results": 1000})
    resp.raise_for_status()
    experiments = resp.json().get("experiments", [])

    rows = []
    for exp in experiments:
        exp_id = exp["experiment_id"]
        exp_name = exp["name"]
        page_token = None
        while True:
            body = {"experiment_ids": [exp_id], "filter": "tags.model != ''", "max_results": 1000}
            if page_token:
                body["page_token"] = page_token
            r = requests.post(f"{base}/runs/search", auth=auth, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            for run in data.get("runs", []):
                info = run.get("info", {})
                tags = {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}
                metrics = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}
                params = {p["key"]: p["value"] for p in run.get("data", {}).get("params", [])}
                if not tags.get("model"):
                    continue
                rows.append({
                    "run_id": info.get("run_id", ""),
                    "experiment": exp_name,
                    "sample_type": tags.get("sample_type", ""),
                    "target": tags.get("target", ""),
                    "timepoints": tags.get("timepoints", ""),
                    "model": tags.get("model", ""),
                    "search": tags.get("search", ""),
                    "config": tags.get("config", ""),
                    "n_train": params.get("n_train"),
                    "n_test": params.get("n_test"),
                    "n_synthetic": params.get("n_synthetic"),
                    "apply_pls": params.get("apply_pls"),
                    **{m: metrics.get(m) for m in METRIC_COLS_ALL},
                })
            page_token = data.get("next_page_token")
            if not page_token:
                break

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    for col in METRIC_COLS_ALL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_local_results(results_dir) -> tuple[list, list]:
    """Return (csv_files, run_names) from local results directory."""
    csv_files = list(Path(results_dir).rglob("results_summary.csv"))
    run_names = [f.parent.name for f in csv_files]
    return csv_files, run_names


def render_data_source_sidebar(results_dir) -> tuple[pd.DataFrame | None, bool]:
    """
    Render 'Data source' section in sidebar with DagsHub toggle + local run selector.
    Returns (df, use_dagshub). df is None if nothing loaded yet (caller should st.stop()).
    """
    token = os.environ.get("DAGSHUB_USER_TOKEN", "")
    csv_files, run_names = load_local_results(results_dir)
    has_local = len(run_names) > 0

    with st.sidebar:
        st.header("Data source")
        use_dagshub = st.toggle(
            "Load from DagsHub",
            value=bool(token),
            help="Fetches live runs from DagsHub MLflow. Requires token in secrets.",
        )

    if use_dagshub:
        if not token:
            st.warning("No DAGSHUB_USER_TOKEN found in secrets.")
            return None, True
        with st.spinner("A carregar runs do DagsHub…"):
            df = load_from_dagshub(token)
        if df.empty:
            st.info("Nenhuma run encontrada no DagsHub. Executa o treino primeiro.")
            return None, True
        st.caption(f"**{len(df)} runs** carregadas do DagsHub MLflow")
        return df, True
    elif has_local:
        with st.sidebar:
            selected_run = st.selectbox("Local run", run_names)
            selected_file = csv_files[run_names.index(selected_run)]
        df = pd.read_csv(selected_file)
        return df, False
    else:
        st.info("Sem resultados locais nem no DagsHub. Executa `docker compose run --rm train` primeiro.")
        return None, False


def apply_group_labels(series, group_labels=None):
    """Rename group values in a pandas Series using the label map."""
    if group_labels is None:
        _init_defaults()
        group_labels = st.session_state.group_labels
    return series.map(lambda x: group_labels.get(str(x), str(x)))
