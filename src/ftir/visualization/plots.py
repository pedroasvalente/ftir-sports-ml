import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

WATER_REGION = (1850, 2500)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def plot_confusion_matrix(
    y_test,
    label_encoder: list,
    accuracy: float,
    sample_type: str,
    train_pct: float,
    run_name: str,
    target: str,
    save_dir: str,
):
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        y_test, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, cbar=False,
        xticklabels=label_encoder, yticklabels=label_encoder, ax=ax,
    )
    ax.set_title(f"{target} — {sample_type} | {int(train_pct*100)}%\n{run_name}")
    plt.text(0.5, 1.04, f"Balanced Acc: {accuracy*100:.1f}%", fontsize=9,
             ha="center", transform=ax.transAxes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fname = f"{target}_{sample_type}_{int(train_pct*100)}pct_{run_name}_cm.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    try:
        import mlflow
        mlflow.log_figure(fig, fname)
    except Exception:
        pass
    plt.close()


def plot_roc_curve(
    y_test_encoded,
    y_prob: np.ndarray,
    label_encoder: list,
    sample_type: str,
    train_pct: float,
    run_name: str,
    target: str,
    roc_auc: float | None,
    save_dir: str,
):
    _ensure_dir(save_dir)
    fig, ax = plt.subplots()
    has_curves = False
    if y_prob is not None:
        for i, cls in enumerate(label_encoder):
            try:
                y_bin = (y_prob.argmax(axis=1) == i).astype(int) if y_test_encoded is None \
                    else (y_test_encoded == i).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
                ax.plot(fpr, tpr, label=f"{cls}")
                has_curves = True
            except Exception:
                pass
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    auc_str = f" (AUC={roc_auc:.3f})" if roc_auc else ""
    ax.set_title(f"{target} — ROC {sample_type}{auc_str}")
    if has_curves:
        ax.legend()
    plt.tight_layout()
    fname = f"{target}_{sample_type}_{int(train_pct*100)}pct_{run_name}_roc.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    try:
        import mlflow
        mlflow.log_figure(fig, fname)
    except Exception:
        pass
    plt.close()


def plot_vip_scores(
    vip: np.ndarray,
    wavenumbers: np.ndarray,
    sample_type: str,
    target: str,
    run_name: str,
    save_dir: str,
    top_n: int = 20,
):
    """Bar chart of top-N VIP scores with axis break at water region."""
    _ensure_dir(save_dir)

    mask_left = wavenumbers > WATER_REGION[1]
    mask_right = wavenumbers < WATER_REGION[0]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True,
        gridspec_kw={"width_ratios": [1, 3], "wspace": 0.02},
    )

    for ax, mask in ((ax_l, mask_left), (ax_r, mask_right)):
        wn = wavenumbers[mask]
        vi = vip[mask]
        ax.plot(wn, vi, color="#2c7bb6", linewidth=1)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="VIP=1")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    ax_l.set_xlim(ax_l.get_xlim())
    fig.text(0.04, 0.5, "VIP score", va="center", rotation="vertical")
    fig.text(0.5, 0.01, r"Wavenumber (cm$^{-1}$)", ha="center")
    fig.suptitle(f"{target} — VIP scores [{sample_type}]")
    ax_l.legend(fontsize=8)

    fname = f"{target}_{sample_type}_{run_name}_vip.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    try:
        import mlflow
        mlflow.log_figure(fig, fname)
    except Exception:
        pass
    plt.close()
    return fpath


def plot_mean_spectra(
    df: pd.DataFrame,
    ftir_cols: list[str],
    group_col: str = "group_fam",
    sample_type: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Mean ± SD spectra per group with axis break at water absorption region.
    Each group gets its own colour; shaded band is ±1 SD.
    """
    wavenumbers = np.array([float(c) for c in ftir_cols])
    groups = df[group_col].unique()
    palette = sns.color_palette("Set1", len(groups))

    mask_left = wavenumbers > WATER_REGION[1]
    mask_right = wavenumbers < WATER_REGION[0]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True,
        gridspec_kw={"width_ratios": [1, 3], "wspace": 0.02},
    )

    for ax, mask in ((ax_l, mask_left), (ax_r, mask_right)):
        wn = wavenumbers[mask]
        for grp, color in zip(groups, palette):
            sub = df[df[group_col] == grp][ftir_cols].values[:, mask].astype(float)
            mean = sub.mean(axis=0)
            sd = sub.std(axis=0)
            ax.plot(wn, mean, color=color, label=str(grp), linewidth=1.2)
            ax.fill_between(wn, mean - sd, mean + sd, alpha=0.15, color=color)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    ax_l.legend(title=group_col, fontsize=8)
    fig.text(0.04, 0.5, "Absorbance (a.u.)", va="center", rotation="vertical")
    fig.text(0.5, 0.01, r"Wavenumber (cm$^{-1}$)", ha="center")
    fig.suptitle(f"Mean ± SD spectra — {sample_type}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
