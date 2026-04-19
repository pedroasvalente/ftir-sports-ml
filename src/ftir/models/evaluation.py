import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

from ftir.config import logger, random_seed

N_BOOTSTRAP = 500  # bootstrap resamples for 95% CI on test metrics


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    model_name: str,
) -> dict:
    """
    Evaluate a fitted model (or imblearn Pipeline) on the test set.

    Returns scalar metrics, confusion matrix, predictions, probabilities,
    bootstrap 95% CIs for key metrics, and feature importances.
    """
    # ── Predictions ───────────────────────────────────────────────────────────
    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=-1)

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    roc_auc = None
    if y_prob is not None and len(np.unique(y_test)) > 1:
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed: {e}")

    # ── Point estimates ────────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    per_class_sens = _per_class_sensitivity(cm)

    metrics = {
        "balanced_accuracy":  balanced_accuracy_score(y_test, y_pred),
        "mcc":                matthews_corrcoef(y_test, y_pred),
        "cohen_kappa":        cohen_kappa_score(y_test, y_pred),
        "f1_weighted":        f1_score(y_test, y_pred, average="weighted"),
        "f1_macro":           f1_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc":            roc_auc,
        "cm":                 cm,
        "y_pred":             y_pred,
        "y_prob":             y_prob,
        **per_class_sens,
    }

    # ── Bootstrap 95% CIs ─────────────────────────────────────────────────────
    ci = _bootstrap_ci(y_test, y_pred, y_prob)
    metrics.update(ci)

    # ── Feature importances ───────────────────────────────────────────────────
    metrics["feature_importances"] = _feature_importances(
        model, X_train, X_test, y_test, model_name
    )

    return metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _per_class_sensitivity(cm: np.ndarray) -> dict:
    """Sensitivity (recall) per class from confusion matrix."""
    result = {}
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        result[f"sensitivity_class{i}"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return result


def _bootstrap_ci(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    rng_seed: int = random_seed,
) -> dict:
    """
    Bootstrap 95% confidence intervals for balanced_accuracy, MCC, and ROC-AUC.

    Resamples (y_test, y_pred) with replacement N_BOOTSTRAP times.
    Returns {metric}_ci_low and {metric}_ci_high for each metric.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(y_test)
    ba_boot, mcc_boot, auc_boot = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_test[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate bootstrap samples
        ba_boot.append(balanced_accuracy_score(yt, yp))
        mcc_boot.append(matthews_corrcoef(yt, yp))
        if y_prob is not None:
            ypr = y_prob[idx]
            try:
                auc_boot.append(roc_auc_score(yt, ypr, multi_class="ovr", average="macro"))
            except Exception:
                pass

    lo, hi = alpha / 2, 1 - alpha / 2
    ci = {}
    if ba_boot:
        ci["balanced_accuracy_ci_low"]  = float(np.quantile(ba_boot, lo))
        ci["balanced_accuracy_ci_high"] = float(np.quantile(ba_boot, hi))
    if mcc_boot:
        ci["mcc_ci_low"]  = float(np.quantile(mcc_boot, lo))
        ci["mcc_ci_high"] = float(np.quantile(mcc_boot, hi))
    if auc_boot:
        ci["roc_auc_ci_low"]  = float(np.quantile(auc_boot, lo))
        ci["roc_auc_ci_high"] = float(np.quantile(auc_boot, hi))

    return ci


def _feature_importances(model, X_train, X_test, y_test, model_name: str) -> np.ndarray:
    """Extract feature importances, handling both raw estimators and imblearn Pipelines."""
    # Unwrap imblearn / sklearn Pipeline to get the actual classifier
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("classifier", model)

    name = model_name.lower()
    try:
        if name == "xgboost":
            booster = estimator.get_booster()
            imp_dict = booster.get_score(importance_type="gain")
            lv = np.zeros(X_train.shape[1])
            for i in range(len(lv)):
                lv[i] = imp_dict.get(f"f{i}", 0.0)
            total = lv.sum()
            if total > 0:
                lv /= total
        elif name in ("mlp_classifier", "mlp"):
            # Permutation importance on test set: measures generalisation contribution
            perm = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=random_seed
            )
            lv = perm.importances_mean
        else:
            lv = estimator.feature_importances_
    except Exception as e:
        logger.warning(f"Feature importance extraction failed for {model_name}: {e}")
        lv = np.zeros(X_train.shape[1])

    return lv
