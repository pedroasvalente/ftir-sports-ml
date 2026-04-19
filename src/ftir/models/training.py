import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV

from ftir.config import (
    EXPERIMENTS_DIR, RESULTS_DIR, WATER_REGION,
    global_threshold_acc, init_mlflow, logger, random_seed,
)
from ftir.data.loader import get_ftir_columns, load_data
from ftir.data.splits import has_repeated_measures
from ftir.models.configs import CV_SEARCH_ARGS, MODEL_REGISTRY
from ftir.models.evaluation import evaluate
from ftir.preprocessing.pipeline import preprocess
from ftir.visualization.plots import plot_confusion_matrix, plot_roc_curve, plot_vip_scores

mlflow = init_mlflow()

# Lazy-import MLflow helpers — broken protobuf / missing install won't crash training
try:
    from mlflow.models import infer_signature as _infer_signature
    from mlflow.tracking import MlflowClient
    _mlflow_client = MlflowClient()
except Exception as _mlflow_import_err:
    logger.warning(f"MLflow tracking helpers unavailable: {_mlflow_import_err}")
    _infer_signature = None
    _mlflow_client = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_key(sample_type, tp_label, model_name, search_label):
    return f"{sample_type}__{tp_label}__{model_name}__{search_label}"


def _save_detail_files(run_slug: str, detail_records: list[dict]):
    """
    Write per-run CM, ROC and VIP data to JSON files alongside results_summary.csv.
    These are consumed by the Streamlit app for interactive plots.
    """
    out_dir = Path(RESULTS_DIR) / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_data, roc_data, vip_data = {}, {}, {}
    for rec in detail_records:
        key = rec["key"]
        if rec.get("cm") is not None:
            cm_data[key] = rec["cm"].tolist()
        if rec.get("y_prob") is not None and rec.get("y_test") is not None:
            roc_data[key] = {
                "y_prob":  rec["y_prob"].tolist(),
                "y_test":  rec["y_test"].tolist(),
                "classes": rec["classes"],
            }
        if rec.get("vip") is not None and rec.get("wavenumbers") is not None:
            vip_data[key] = {
                "vip":         rec["vip"].tolist(),
                "wavenumbers": rec["wavenumbers"].tolist(),
            }

    (out_dir / "cm_data.json").write_text(json.dumps(cm_data))
    (out_dir / "roc_data.json").write_text(json.dumps(roc_data))
    (out_dir / "vip_data.json").write_text(json.dumps(vip_data))
    logger.info(f"Detail files saved to {out_dir}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_experiment(config_path: str):
    """Load a JSON experiment config and run all combinations.

    MLflow structure:
      Experiment : {experiment_name} | {sample_type}   (one per matrix)
        Parent run : {tp_label}                         (one per timepoints config)
          Child run  : {model} | {search}               (one per model fit)
    """
    with open(config_path) as f:
        cfg = json.load(f)

    base_name   = cfg["experiment_name"]
    run_slug    = cfg.get("run_name", "run")
    apply_pls_values   = cfg.get("apply_pls", [True])
    apply_smote_values = cfg.get("apply_smote_resampling", [True])
    scale_values       = cfg.get("scale", [True])

    df = load_data()
    ftir_cols   = get_ftir_columns(df)
    wavenumbers = np.array([float(c) for c in ftir_cols])

    results_all   = []
    detail_records = []

    for sample_type in cfg["sample_types"]:
        mlflow.set_experiment(f"{base_name} | {sample_type}")

        for target in cfg["targets_to_predict"]:
            for timepoints in cfg.get("timepoints", [None]):
                tp_label = (
                    "tp" + "_".join(str(t) for t in timepoints)
                    if timepoints else "all_tp"
                )

                with mlflow.start_run(run_name=tp_label) as tp_run:
                    mlflow.set_tags({
                        "sample_type": sample_type,
                        "target":      target,
                        "timepoints":  tp_label,
                        "config":      Path(config_path).name,
                    })

                    for train_pct in cfg["train_percentages"]:
                        for model_name in cfg["model_types_to_train"]:
                            for search_type in cfg["searchs_hipermetrics"]:
                                for apply_pls in apply_pls_values:
                                    for apply_smote in apply_smote_values:
                                        for scale in scale_values:
                                            result, detail = _train_single(
                                                df=df,
                                                ftir_cols=ftir_cols,
                                                wavenumbers=wavenumbers,
                                                sample_type=sample_type,
                                                target=target,
                                                timepoints=timepoints,
                                                tp_label=tp_label,
                                                train_pct=train_pct,
                                                model_name=model_name,
                                                search_type=search_type,
                                                apply_pls=apply_pls,
                                                apply_smote=apply_smote,
                                                scale=scale,
                                                n_components=cfg.get("n_components", [3])[0],
                                                num_classes=cfg.get("num_classes", [3])[0],
                                                parent_run_id=tp_run.info.run_id,
                                                config_name=Path(config_path).name,
                                                run_slug=run_slug,
                                                group_fam=cfg.get("selected_group_fam"),
                                            )
                                            if result:
                                                results_all.append(result)
                                            if detail:
                                                detail_records.append(detail)

    if results_all:
        summary = pd.DataFrame(results_all)
        out_dir = Path(RESULTS_DIR) / run_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "results_summary.csv", index=False)
        logger.info(f"Results saved to {out_dir / 'results_summary.csv'}")
        _save_detail_files(run_slug, detail_records)


# ── Single training run ───────────────────────────────────────────────────────

def _train_single(
    df, ftir_cols, wavenumbers,
    sample_type, target, timepoints, tp_label, train_pct, model_name, search_type,
    apply_pls, apply_smote, scale, n_components, num_classes,
    parent_run_id, config_name, run_slug="", group_fam=None,
) -> tuple[dict | None, dict | None]:
    from ftir.data.loader import filter_sample_data
    from sklearn.preprocessing import LabelEncoder

    try:
        data_df, X, y_raw = filter_sample_data(
            df=df,
            target=target,
            sample_type=sample_type,
            ftir_columns=ftir_cols,
            group_fam=group_fam,
            timepoints=timepoints,
        )
    except ValueError as e:
        logger.warning(str(e))
        return None, None

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    groups = data_df["person_code"].values if has_repeated_measures(timepoints) else None

    prep = preprocess(
        X=X,
        y=y,
        groups=groups,
        timepoints=timepoints,
        test_size=1 - train_pct,
        scale=scale,
        apply_pls=apply_pls,
        n_components=n_components,
        num_classes=num_classes,
        random_state=random_seed,
    )

    X_train      = prep["X_train"]
    X_test       = prep["X_test"]
    y_train      = prep["y_train"]
    y_test       = prep["y_test"]
    pls          = prep["pls"]
    groups_train = prep["groups_train"]
    cv_splitter  = prep["cv_splitter"]

    config      = MODEL_REGISTRY[model_name]
    search_label = "grid"

    # ── Build estimator: SMOTE inside CV pipeline if requested ────────────────
    if apply_smote and num_classes > 1:
        estimator  = ImbPipeline([
            ("smote",      SMOTE(random_state=random_seed)),
            ("classifier", config.get_model()),
        ])
        param_grid = config.get_pipeline_params("classifier")
        n_synthetic_note = "inside_CV"
    else:
        estimator  = config.get_model()
        param_grid = config.get_params("grid")
        n_synthetic_note = "none"

    search_args = {**CV_SEARCH_ARGS["GridSearchCV"], "cv": cv_splitter}
    search = GridSearchCV(estimator, param_grid, **search_args)

    # Pass groups to fit for StratifiedGroupKFold
    fit_kwargs = {}
    if groups_train is not None:
        fit_kwargs["groups"] = groups_train

    search.fit(X_train, y_train, **fit_kwargs)
    best_model = search.best_estimator_

    metrics = evaluate(best_model, X_test, y_test, X_train, model_name)

    # ── VIP scores (original wavenumber space) ────────────────────────────────
    vip        = pls.vip_scores() if pls is not None else None
    valid_mask = (wavenumbers < WATER_REGION[0]) | (wavenumbers > WATER_REGION[1])

    run_name = f"{config.desc_name} | {search_label}"

    # ── MLflow logging ────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name, nested=True, parent_run_id=parent_run_id):
        mlflow.set_tags({
            "sample_type": sample_type,
            "target":      target,
            "timepoints":  tp_label,
            "model":       config.desc_name,
            "search":      search_label,
            "config":      config_name,
            "run_slug":    run_slug,
            # SMOTE is now applied inside CV: cv_best_score is uncontaminated
            "cv_note":     f"SMOTE_{n_synthetic_note}",
        })
        mlflow.log_params({
            "train_pct":     train_pct,
            "scale":         scale,
            "apply_pls":     apply_pls,
            "n_components":  n_components,
            "apply_smote":   apply_smote,
            "n_train":       len(y_train),
            "n_test":        len(y_test),
            **{k.replace("classifier__", ""): v for k, v in search.best_params_.items()},
        })
        for k, v in metrics.items():
            if k not in ("cm", "y_pred", "y_prob", "feature_importances") and v is not None:
                mlflow.log_metric(k, float(v))
        mlflow.log_metric("cv_best_score", search.best_score_)
        mlflow.log_metric(
            "cv_best_score_std",
            search.cv_results_["std_test_score"][search.best_index_],
        )

        try:
            if _infer_signature is not None:
                sig = _infer_signature(X_test, metrics["y_pred"])
                mlflow.sklearn.log_model(best_model, "model", signature=sig)
        except Exception as e:
            logger.warning(f"MLflow model logging failed: {e}")

        if metrics["balanced_accuracy"] >= global_threshold_acc / 100:
            _save_plots(
                metrics, le, sample_type, train_pct, run_name, target,
                wavenumbers, vip, valid_mask,
            )

    # ── Build result row for CSV ──────────────────────────────────────────────
    ci_cols = {k: metrics.get(k) for k in metrics if k.endswith(("_ci_low", "_ci_high"))}
    sens_cols = {k: metrics.get(k) for k in metrics if k.startswith("sensitivity_class")}

    result = {
        "sample_type":        sample_type,
        "target":             target,
        "timepoints":         str(timepoints),
        "model":              config.desc_name,
        "search":             search_label,
        "train_pct":          train_pct,
        "scale":              scale,
        "apply_pls":          apply_pls,
        "apply_smote":        apply_smote,
        "balanced_accuracy":  metrics["balanced_accuracy"],
        "mcc":                metrics["mcc"],
        "cohen_kappa":        metrics["cohen_kappa"],
        "f1_weighted":        metrics["f1_weighted"],
        "f1_macro":           metrics["f1_macro"],
        "roc_auc":            metrics["roc_auc"],
        "cv_best_score":      search.best_score_,
        **ci_cols,
        **sens_cols,
    }

    # ── Detail record for JSON files ──────────────────────────────────────────
    detail = {
        "key":         _run_key(sample_type, tp_label, config.desc_name, search_label),
        "cm":          metrics["cm"],
        "y_prob":      metrics["y_prob"],
        "y_test":      y_test,
        "classes":     list(le.classes_),
        "vip":         vip,
        "wavenumbers": wavenumbers,
    }

    return result, detail


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save_plots(metrics, le, sample_type, train_pct, run_name, target,
                wavenumbers, vip, valid_mask):
    import os
    from ftir.config import FIGURES_DIR

    fig_dir = Path(FIGURES_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        y_test=metrics["cm"],
        label_encoder=list(le.classes_),
        accuracy=metrics["balanced_accuracy"],
        sample_type=sample_type,
        train_pct=train_pct,
        run_name=run_name,
        target=target,
        save_dir=str(fig_dir / "confusion_matrices"),
    )

    if metrics["y_prob"] is not None:
        plot_roc_curve(
            y_test_encoded=None,
            y_prob=metrics["y_prob"],
            label_encoder=list(le.classes_),
            sample_type=sample_type,
            train_pct=train_pct,
            run_name=run_name,
            target=target,
            roc_auc=metrics["roc_auc"],
            save_dir=str(fig_dir / "roc_curves"),
        )

    if vip is not None and valid_mask is not None:
        plot_vip_scores(
            vip=vip[valid_mask],
            wavenumbers=wavenumbers[valid_mask],
            sample_type=sample_type,
            target=target,
            run_name=run_name,
            save_dir=str(fig_dir / "vip_scores"),
        )
