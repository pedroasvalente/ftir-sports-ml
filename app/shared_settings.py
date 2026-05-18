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
                # run_name is in info.run_name (MLflow ≥ 2.0) or the mlflow.runName tag
                run_name = (
                    info.get("run_name")
                    or tags.get("mlflow.runName")
                    or tags.get("run_name")
                    or ""
                )
                rows.append({
                    "run_id": info.get("run_id", ""),
                    "run_name": run_name,
                    "run_slug": tags.get("run_slug", ""),
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


@st.cache_data(ttl=300)
def load_local_results_df(results_dir: str) -> pd.DataFrame:
    """
    Concatenate every `results_summary.csv` found under `results_dir`,
    tagging each row with the parent folder name as `config` / `run_slug`.
    This is the offline-friendly path used on Streamlit Cloud when DagsHub
    is unavailable.
    """
    base = Path(results_dir)
    csv_files = sorted(base.rglob("results_summary.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        try:
            sub = pd.read_csv(f)
        except Exception:
            continue
        run_name = f.parent.name
        sub["config"] = run_name
        sub["run_slug"] = run_name
        sub["experiment"] = run_name
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in METRIC_COLS_ALL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _render_run_info(df: pd.DataFrame, run_label: str, source: str):
    """
    Render a compact run-identity info box below the data source controls.
    Shows run name, row count, matrices, models and timepoint configs present.
    """
    n_runs = len(df)
    matrices = sorted(df["sample_type"].dropna().unique()) if "sample_type" in df.columns else []
    models   = sorted(df["model"].dropna().unique())       if "model"       in df.columns else []
    tps      = sorted(df["timepoints"].dropna().unique())  if "timepoints"  in df.columns else []

    # Try to surface the config / run_name tag
    config_name = ""
    if "config" in df.columns:
        configs = df["config"].dropna().unique()
        config_name = configs[0] if len(configs) == 1 else ", ".join(configs[:2])

    with st.sidebar.expander("ℹ️ Loaded run", expanded=True):
        st.markdown(f"**Source:** {source}")
        if config_name:
            st.markdown(f"**Run:** `{config_name}`")
        st.markdown(f"**Total rows:** {n_runs}")
        if matrices:
            st.markdown(f"**Matrices ({len(matrices)}):** {', '.join(matrices)}")
        if models:
            st.markdown(f"**Models ({len(models)}):** {', '.join(models)}")
        if tps:
            tp_str = " · ".join([str(t) for t in tps])
            st.markdown(f"**Timepoint configs:** {tp_str}")

        # Best overall balanced accuracy as a quick headline metric
        if "balanced_accuracy" in df.columns and df["balanced_accuracy"].notna().any():
            best_ba   = df["balanced_accuracy"].max()
            best_row  = df.loc[df["balanced_accuracy"].idxmax()]
            best_mat  = best_row.get("sample_type", "?")
            best_mod  = best_row.get("model", "?")
            st.markdown(
                f"**Best BA:** `{best_ba:.3f}` "
                f"<small>({best_mat} / {best_mod})</small>",
                unsafe_allow_html=True,
            )


def render_data_source_sidebar(results_dir) -> tuple[pd.DataFrame | None, bool]:
    """
    Load training results, preferring local CSVs committed to the repo so the
    app keeps working when DagsHub is unavailable. DagsHub MLflow is used as a
    fallback (and can be forced via a sidebar toggle when the token is set).

    Shows a 'Run' selectbox in the sidebar so the user can filter to a
    specific training run or keep 'All'.

    Returns (df, used_dagshub). df is None if no results could be loaded.
    """
    token = os.environ.get("DAGSHUB_USER_TOKEN", "")

    # Try local repo first — fast, offline, and always available on Streamlit Cloud.
    df_local = load_local_results_df(str(results_dir))
    local_available = not df_local.empty

    # Sidebar toggle: only meaningful when DagsHub credentials exist
    with st.sidebar:
        st.header("Data source")
        if token and local_available:
            use_dagshub = st.toggle(
                "Use DagsHub MLflow",
                value=False,
                help="Off: load committed CSVs from the repo (fast, offline). "
                     "On: query the DagsHub MLflow tracking server for the latest runs.",
            )
        elif token and not local_available:
            st.caption("No local results found — using DagsHub.")
            use_dagshub = True
        else:
            if not local_available:
                st.error(
                    "No local results found and no DAGSHUB_USER_TOKEN set. "
                    "Commit `results/*/results_summary.csv` to the repo or set the token."
                )
                return None, False
            st.caption("Local results loaded from the repo.")
            use_dagshub = False

    df_all = pd.DataFrame()
    source = "Local repo"

    if use_dagshub:
        try:
            with st.spinner("Loading runs from DagsHub…"):
                df_all = load_from_dagshub(token)
            source = "DagsHub MLflow"
            if df_all.empty:
                st.info("No runs found on DagsHub — falling back to local CSVs.")
                df_all = df_local
                source = "Local repo (DagsHub empty)"
        except Exception as e:
            st.warning(
                f"Could not reach DagsHub ({type(e).__name__}). "
                "Falling back to local results committed to the repo."
            )
            df_all = df_local
            source = "Local repo (DagsHub unreachable)"
    else:
        df_all = df_local

    if df_all is None or df_all.empty:
        st.error("No results available from either source.")
        return None, use_dagshub

    # Use run_slug tag to distinguish runs (e.g. study1_group_fam_v2).
    # For older runs without the tag, fall back to config filename.
    if "run_slug" in df_all.columns and df_all["run_slug"].astype(str).str.strip().any():
        _name_col = "run_slug"
    elif "config" in df_all.columns:
        _name_col = "config"
    else:
        _name_col = None

    run_vals = (
        sorted(df_all[_name_col].replace("", pd.NA).dropna().unique())
        if _name_col else []
    )

    with st.sidebar:
        st.header("Run")
        if run_vals:
            selected_run = st.selectbox(
                "Filter by run",
                options=["All"] + run_vals,
                help="Select a specific training run or show all.",
            )
            df = df_all[df_all[_name_col] == selected_run].copy() if selected_run != "All" else df_all
            run_label = selected_run if selected_run != "All" else "All runs"
        else:
            df = df_all
            run_label = "All runs"

    _render_run_info(df, run_label=run_label, source=source)
    return df, use_dagshub


def apply_group_labels(series, group_labels=None):
    """Rename group values in a pandas Series using the label map."""
    if group_labels is None:
        _init_defaults()
        group_labels = st.session_state.group_labels
    return series.map(lambda x: group_labels.get(str(x), str(x)))
