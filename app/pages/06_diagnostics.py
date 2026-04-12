import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ftir.config import RESULTS_DIR, WATER_REGION
from ftir.data.config import SAMPLE_TYPES
from shared_settings import render_appearance_sidebar, render_data_source_sidebar, load_local_results

st.set_page_config(page_title="Model Diagnostics", layout="wide")
st.title("Model Diagnostics")

# ── Load results data ──────────────────────────────────────────────────────────
df, use_dagshub = render_data_source_sidebar(RESULTS_DIR)
if df is None:
    st.stop()

# ── Appearance ─────────────────────────────────────────────────────────────────
render_appearance_sidebar(show_models=True)

# ── Sidebar filters + run selector ────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    sample_type = st.selectbox("Matrix", SAMPLE_TYPES)

    if "timepoints" in df.columns:
        tp_opts = sorted(df["timepoints"].dropna().unique())
        tp_label = st.selectbox("Timepoints", tp_opts)
    else:
        tp_label = None

    if "model" in df.columns:
        model_opts = sorted(df["model"].dropna().unique())
        model = st.selectbox("Model", model_opts)
    else:
        model = None

    if "search" in df.columns:
        search_opts = sorted(df["search"].dropna().unique())
        search = st.selectbox("Search strategy", search_opts)
    else:
        search = "default"

    st.divider()
    st.subheader("Diagnostic run")
    # Only show runs that actually contain the JSON diagnostic files,
    # sorted newest-first by directory modification time.
    _results_path = Path(RESULTS_DIR)
    _diag_runs = sorted(
        [
            d.name for d in _results_path.iterdir()
            if d.is_dir() and any((d / f).exists() for f in
                                  ["cm_data.json", "roc_data.json", "vip_data.json"])
        ],
        key=lambda n: (_results_path / n).stat().st_mtime,
        reverse=True,  # newest first
    )
    if _diag_runs:
        run_name = st.selectbox(
            "Run",
            options=_diag_runs,
            help="Local training runs that contain diagnostic JSON files (newest first).",
        )
    else:
        run_name = st.text_input(
            "Run name (subdirectory of results/)",
            value="",
            help="No runs with diagnostic JSON files found locally. Run training first.",
        )

# ── Helper: load JSON diagnostic files ────────────────────────────────────────
def _load_json(run: str, filename: str):
    """Load a JSON file from results/<run>/<filename>. Returns None if not found."""
    if not run:
        return None
    p = Path(RESULTS_DIR) / run / filename
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)

cm_data  = _load_json(run_name, "cm_data.json")
roc_data = _load_json(run_name, "roc_data.json")
vip_data = _load_json(run_name, "vip_data.json")

_any_diag = any(x is not None for x in [cm_data, roc_data, vip_data])
if run_name and not _any_diag:
    st.warning(
        f"No diagnostic JSON files found in `results/{run_name}/`. "
        "Run training first to generate diagnostic data."
    )

# ── Build lookup key ──────────────────────────────────────────────────────────
diag_key = f"{sample_type}__{tp_label}__{model}__{search}"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Confusion Matrix")

if cm_data is None:
    st.info("Run training first to generate diagnostic data (cm_data.json).")
else:
    entry = cm_data.get(diag_key)
    if entry is None:
        st.warning(f"No confusion matrix entry found for key: `{diag_key}`")
    else:
        cm_array = np.array(entry["matrix"])
        class_labels = entry.get("labels", [str(i) for i in range(cm_array.shape[0])])

        fig_cm = px.imshow(
            cm_array,
            x=class_labels,
            y=class_labels,
            text_auto=True,
            color_continuous_scale="RdYlGn",
            labels={"x": "Predicted label", "y": "True label", "color": "Count"},
            title=f"Confusion matrix — {sample_type} | {tp_label} | {model}",
            aspect="auto",
        )
        fig_cm.update_layout(height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Sensitivity (recall) per class
        sensitivities = {}
        for i, cls in enumerate(class_labels):
            tp_val = cm_array[i, i]
            fn_val = cm_array[i, :].sum() - tp_val
            sens = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else float("nan")
            sensitivities[cls] = sens

        cols = st.columns(len(class_labels))
        for col, (cls, sens) in zip(cols, sensitivities.items()):
            col.metric(f"Sensitivity — {cls}", f"{sens:.3f}" if not np.isnan(sens) else "N/A")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("ROC Curves — One-vs-Rest")

if roc_data is None:
    st.info("Run training first to generate diagnostic data (roc_data.json).")
else:
    entry = roc_data.get(diag_key)
    if entry is None:
        st.warning(f"No ROC data entry found for key: `{diag_key}`")
    else:
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize

            y_test = np.array(entry["y_test"])
            y_prob = np.array(entry["y_prob"])
            classes = entry.get("classes", sorted(set(y_test.tolist())))

            y_bin = label_binarize(y_test, classes=classes)
            if y_bin.shape[1] == 1:
                # Binary case — sklearn returns single column
                y_bin = np.hstack([1 - y_bin, y_bin])

            fig_roc = go.Figure()
            palette = px.colors.qualitative.Set1

            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                auc_val = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode="lines",
                    name=f"Class {cls} (AUC = {auc_val:.3f})",
                    line=dict(color=palette[i % len(palette)], width=2),
                ))

            # Diagonal reference
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random classifier",
                line=dict(color="grey", dash="dash", width=1),
                showlegend=True,
            ))

            fig_roc.update_layout(
                title=f"ROC curves (one-vs-rest) — {sample_type} | {tp_label} | {model}",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                height=450,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.02]),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        except ImportError:
            st.error("scikit-learn is required for ROC curve computation.")
        except Exception as exc:
            st.error(f"Could not render ROC curves: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Calibration Plot
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Probability Calibration")

if roc_data is None:
    st.info("Run training first to generate diagnostic data (roc_data.json).")
else:
    entry = roc_data.get(diag_key)
    if entry is None:
        st.warning(f"No calibration data entry found for key: `{diag_key}`")
    else:
        try:
            from sklearn.calibration import calibration_curve
            from sklearn.preprocessing import label_binarize

            y_test = np.array(entry["y_test"])
            y_prob = np.array(entry["y_prob"])
            classes = entry.get("classes", sorted(set(y_test.tolist())))

            y_bin = label_binarize(y_test, classes=classes)
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])

            fig_cal = go.Figure()
            palette = px.colors.qualitative.Set1

            for i, cls in enumerate(classes):
                frac_pos, mean_pred = calibration_curve(
                    y_bin[:, i], y_prob[:, i], n_bins=10, strategy="uniform"
                )
                fig_cal.add_trace(go.Scatter(
                    x=mean_pred, y=frac_pos,
                    mode="lines+markers",
                    name=f"Class {cls}",
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=7),
                ))

            # Perfect calibration diagonal
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line=dict(color="black", dash="dash", width=1),
            ))

            fig_cal.update_layout(
                title=f"Calibration plot — {sample_type} | {tp_label} | {model}",
                xaxis_title="Mean predicted probability",
                yaxis_title="Observed fraction of positives",
                height=420,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05]),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
        except ImportError:
            st.error("scikit-learn is required for calibration curve computation.")
        except Exception as exc:
            st.error(f"Could not render calibration plot: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Aggregated VIP Scores
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Aggregated VIP Scores — All Runs")

if vip_data is None:
    st.info("Run training first to generate diagnostic data (vip_data.json).")
else:
    try:
        # vip_data is expected to be a dict of {run_key: {"wavenumbers": [...], "vip": [...]}}
        all_wn_arrays = []
        all_vip_arrays = []

        for run_key, v in vip_data.items():
            wn = np.array(v.get("wavenumbers", []))
            vip = np.array(v.get("vip", []))
            if len(wn) > 0 and len(vip) == len(wn):
                all_wn_arrays.append(wn)
                all_vip_arrays.append(vip)

        n_runs = len(all_vip_arrays)

        if n_runs == 0:
            st.warning("No valid VIP entries found in vip_data.json.")
        else:
            # Use wavenumbers from first entry as reference
            ref_wn = all_wn_arrays[0]

            # Align all arrays to reference wavenumber grid
            aligned = []
            for wn, vip in zip(all_wn_arrays, all_vip_arrays):
                if np.array_equal(wn, ref_wn):
                    aligned.append(vip)
                else:
                    # Interpolate to reference grid
                    aligned.append(np.interp(ref_wn, np.sort(wn), vip[np.argsort(wn)]))

            stacked = np.vstack(aligned)
            mean_vip = stacked.mean(axis=0)
            std_vip = stacked.std(axis=0)

            wn_arr = ref_wn.astype(float)
            water_lo, water_hi = float(WATER_REGION[0]), float(WATER_REGION[1])

            fig_vip = go.Figure()

            # Shaded std band
            fig_vip.add_trace(go.Scatter(
                x=np.concatenate([wn_arr, wn_arr[::-1]]),
                y=np.concatenate([mean_vip + std_vip, (mean_vip - std_vip)[::-1]]),
                fill="toself",
                fillcolor="rgba(70, 130, 180, 0.20)",
                line=dict(color="rgba(255,255,255,0)"),
                name="± 1 SD",
                hoverinfo="skip",
            ))

            # Mean VIP line
            fig_vip.add_trace(go.Scatter(
                x=wn_arr,
                y=mean_vip,
                mode="lines",
                name="Mean VIP",
                line=dict(color="steelblue", width=2),
            ))

            # VIP > 1 threshold line
            fig_vip.add_hline(
                y=1.0,
                line_dash="dot",
                line_color="red",
                annotation_text="VIP = 1 threshold",
                annotation_position="bottom right",
            )

            # Red shading for VIP > 1 region
            above_one_mask = mean_vip >= 1.0
            if above_one_mask.any():
                fig_vip.add_trace(go.Scatter(
                    x=np.concatenate([wn_arr[above_one_mask], wn_arr[above_one_mask][::-1]]),
                    y=np.concatenate([
                        mean_vip[above_one_mask],
                        np.ones(above_one_mask.sum()),
                    ]),
                    fill="toself",
                    fillcolor="rgba(220, 50, 50, 0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="VIP > 1",
                    hoverinfo="skip",
                ))

            # Water region grey shading
            fig_vip.add_vrect(
                x0=water_lo,
                x1=water_hi,
                fillcolor="rgba(180, 180, 180, 0.40)",
                line_width=0,
                annotation_text="Water vapour / CO₂",
                annotation_position="top left",
                annotation_font_size=11,
            )

            fig_vip.update_layout(
                title=f"Mean VIP score across all runs ± SD (n={n_runs} runs)",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="VIP score",
                height=480,
                xaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_vip, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not render VIP plot: {exc}")
