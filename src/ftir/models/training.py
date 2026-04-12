import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

from ftir.config import EXPERIMENTS_DIR, RESULTS_DIR, global_threshold_acc, init_mlflow, logger, random_seed
from ftir.data.loader import get_ftir_columns, load_data
from ftir.data.splits import has_repeated_measures
from ftir.models.configs import CV_SEARCH_ARGS, MODEL_REGISTRY
from ftir.models.evaluation import evaluate
from ftir.preprocessing.pipeline import preprocess
from ftir.visualization.plots import plot_confusion_matrix, plot_roc_curve, plot_vip_scores

mlflow = init_mlflow()
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
client = MlflowClient()


def run_experiment(config_path: str):
    """Entry point: load a JSON experiment config and run all combinations.

    MLflow structure:
      Experiment : {experiment_name} | {sample_type}   (one per matrix)
        Run       : {model} | {search} | {tp_label}    (one per model fit)

    Tags on each run allow filtering by timepoints, target, config, etc.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    base_name = cfg["experiment_name"]

    df = load_data()
    ftir_cols = get_ftir_columns(df)
    wavenumbers = np.array([float(c) for c in ftir_cols])

    results_all = []

    for sample_type in cfg["sample_types"]:
        # One MLflow experiment per biological matrix
        mlflow.set_experiment(f"{base_name} | {sample_type}")

        for target in cfg["targets_to_predict"]:
            for timepoints in cfg.get("timepoints", [None]):
                tp_label = (
                    "tp" + "_".join(str(t) for t in timepoints)
                    if timepoints else "all_tp"
                )

                # Parent run per timepoints config
                with mlflow.start_run(run_name=tp_label) as tp_run:
                    mlflow.set_tags({
                        "sample_type": sample_type,
                        "target": target,
                        "timepoints": tp_label,
                        "config": Path(config_path).name,
                    })

                    for train_pct in cfg["train_percentages"]:
                        for model_name in cfg["model_types_to_train"]:
                            for search_type in cfg["searchs_hipermetrics"]:
                                for apply_pls in cfg.get("apply_pls", [True]):
                                    for apply_smote in cfg.get("apply_smote_resampling", [True]):
                                        result = _train_single(
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
                                            n_components=cfg.get("n_components", [10])[0],
                                            num_classes=cfg.get("num_classes", [3])[0],
                                            parent_run_id=tp_run.info.run_id,
                                            config_name=Path(config_path).name,
                                            group_fam=cfg.get("selected_group_fam"),
                                        )
                                        if result:
                                            results_all.append(result)

    if results_all:
        summary = pd.DataFrame(results_all)
        run_slug = cfg.get("run_name", "run")
        out_dir = Path(RESULTS_DIR) / run_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "results_summary.csv", index=False)
        logger.info(f"Results saved to {out_dir / 'results_summary.csv'}")


def _train_single(
    df, ftir_cols, wavenumbers,
    sample_type, target, timepoints, tp_label, train_pct, model_name, search_type,
    apply_pls, apply_smote, n_components, num_classes,
    parent_run_id, config_name, group_fam=None,
) -> dict | None:
    from ftir.data.loader import filter_sample_data
    from sklearn.preprocessing import LabelEncoder

    search_cls_name = "GridSearchCV" if search_type == "grid" else "BayesSearchCV"

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
        return None

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    groups = data_df["person_code"].values if has_repeated_measures(timepoints) else None

    prep = preprocess(
        X=X,
        y=y,
        groups=groups,
        timepoints=timepoints,
        test_size=1 - train_pct,
        apply_pls=apply_pls,
        n_components=n_components,
        apply_smote=apply_smote,
        num_classes=num_classes,
        random_state=random_seed,
    )

    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    pls = prep["pls"]
    n_synthetic = prep["n_synthetic"]
    cv_splitter = prep["cv_splitter"]

    config = MODEL_REGISTRY[model_name]
    search_args = {
        **CV_SEARCH_ARGS[search_cls_name],
        "cv": cv_splitter,
    }

    if search_cls_name == "GridSearchCV":
        search = GridSearchCV(
            config.get_model(), config.get_params("grid"), **search_args
        )
    else:
        search = BayesSearchCV(
            config.get_model(), config.get_params("bayes"), **search_args
        )

    # Pass groups to fit if using GroupKFold — sklearn CV objects accept it via fit params
    fit_kwargs = {}
    if prep["groups_train"] is not None and not apply_smote:
        fit_kwargs["groups"] = prep["groups_train"]

    search.fit(X_train, y_train, **fit_kwargs)
    best_model = search.best_estimator_

    metrics = evaluate(best_model, X_test, y_test, X_train, model_name)

    # VIP scores for wavenumber importance (replaces back-projection)
    vip = None
    if pls is not None:
        vip = pls.vip_scores()
        valid_mask = (wavenumbers < 1850) | (wavenumbers > 2500)

    search_label = "grid" if search_type == "grid" else "bayes"
    run_name = f"{config.desc_name} | {search_label}"

    with mlflow.start_run(run_name=run_name, nested=True, parent_run_id=parent_run_id):
        mlflow.set_tags({
            "sample_type": sample_type,
            "target": target,
            "timepoints": tp_label,
            "model": config.desc_name,
            "search": search_label,
            "config": config_name,
        })
        mlflow.log_params({
            "train_pct": train_pct,
            "apply_pls": apply_pls,
            "n_components": n_components,
            "apply_smote": apply_smote,
            "n_synthetic": n_synthetic,
            "n_train": len(y_train),
            "n_test": len(y_test),
            **search.best_params_,
        })
        for k, v in metrics.items():
            if k not in ("cm", "y_pred", "y_prob", "feature_importances") and v is not None:
                mlflow.log_metric(k, float(v))
        mlflow.log_metric("cv_best_score", search.best_score_)
        mlflow.log_metric("cv_best_score_std", search.cv_results_["std_test_score"][search.best_index_])

        try:
            sig = infer_signature(X_test, metrics["y_pred"])
            mlflow.sklearn.log_model(best_model, "model", signature=sig)
        except Exception:
            pass

        balanced_acc = metrics["balanced_accuracy"]
        if balanced_acc >= global_threshold_acc / 100:
            _save_plots(
                metrics, le, sample_type, train_pct, run_name, target,
                wavenumbers, vip, valid_mask if vip is not None else None,
            )

    return {
        "sample_type": sample_type,
        "target": target,
        "timepoints": str(timepoints),
        "model": config.desc_name,
        "search": search_cls_name,
        "train_pct": train_pct,
        "apply_pls": apply_pls,
        "apply_smote": apply_smote,
        "n_synthetic": n_synthetic,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "mcc": metrics["mcc"],
        "cohen_kappa": metrics["cohen_kappa"],
        "f1_weighted": metrics["f1_weighted"],
        "f1_macro": metrics["f1_macro"],
        "roc_auc": metrics["roc_auc"],
    }


def _save_plots(metrics, le, sample_type, train_pct, run_name, target, wavenumbers, vip, valid_mask):
    from ftir.config import FIGURES_DIR
    import os

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
