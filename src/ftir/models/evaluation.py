import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ftir.config import random_seed


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    model_name: str,
) -> dict:
    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=-1)

    roc_auc = None
    if y_prob is not None and len(np.unique(y_test)) > 1:
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"ROC-AUC computation failed: {e}")

    cm = confusion_matrix(y_test, y_pred)
    per_class_sensitivity = _per_class_sensitivity(cm)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc,
        "cm": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
        **per_class_sensitivity,
    }
    metrics["feature_importances"] = _feature_importances(
        model, X_train, X_test, y_test, model_name
    )
    return metrics


def _per_class_sensitivity(cm: np.ndarray) -> dict:
    """Sensitivity (recall) per class from confusion matrix."""
    result = {}
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        result[f"sensitivity_class{i}"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return result


def _feature_importances(model, X_train, X_test, y_test, model_name: str) -> np.ndarray:
    name = model_name.lower()
    if name == "xgboost":
        booster = model.get_booster()
        imp_dict = booster.get_score(importance_type="gain")
        lv = np.zeros(X_train.shape[1])
        for i in range(len(lv)):
            lv[i] = imp_dict.get(f"f{i}", 0.0)
        total = lv.sum()
        if total > 0:
            lv /= total
    elif name in ("mlp_classifier", "mlp"):
        perm = permutation_importance(model, X_test, y_test, random_state=random_seed)
        lv = perm.importances_mean
    else:
        lv = model.feature_importances_
    return lv
