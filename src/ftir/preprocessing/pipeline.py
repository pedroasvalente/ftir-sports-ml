import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ftir.config import logger, random_seed as _seed
from ftir.data.splits import (
    has_repeated_measures,
    make_cv_splitter,
    stratified_split,
    train_test_split_by_person,
)
from ftir.reduction.pls_da import PLSDA


def preprocess(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    timepoints: list[int] | None = None,
    test_size: float = 0.3,
    scale: bool = True,
    apply_pls: bool = True,
    n_components: int = 3,
    num_classes: int = 3,
    random_state: int = _seed,
) -> dict:
    """
    Preprocessing pipeline: split → scale → (optional) PLS-DA.

    SMOTE is intentionally NOT applied here. It is applied inside the CV loop
    via an imblearn Pipeline in training.py, so that synthetic samples never
    appear in CV validation folds (avoids inflated CV estimates).

    When multiple timepoints are used AND groups (person_code) is provided,
    splits by person so the same individual never appears in both train and test.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test,
        pls (fitted PLSDA or None),
        scaler (fitted StandardScaler or None),
        groups_train, groups_test,
        cv_splitter (for use in GridSearchCV)
    """
    repeated = has_repeated_measures(timepoints) and groups is not None

    if repeated:
        logger.info("Using person-aware split (repeated measures detected)")
        from sklearn.model_selection import train_test_split as _tts
        unique_persons = np.unique(groups)
        persons_train, persons_test = _tts(
            unique_persons, test_size=test_size, random_state=random_state
        )
        mask_train = np.isin(groups, persons_train)
        mask_test  = np.isin(groups, persons_test)
        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
        groups_train = groups[mask_train]
        groups_test  = groups[mask_test]
    else:
        logger.info("Using stratified split")
        X_train, X_test, y_train, y_test = stratified_split(
            X, y, test_size=test_size, random_state=random_state
        )
        groups_train = None
        groups_test  = None

    logger.info(f"Split — Train: {len(y_train)} | Test: {len(y_test)}")

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    pls = None
    if apply_pls:
        pls = PLSDA(n_components=n_components)
        X_train = pls.fit_transform(X_train, y_train)
        X_test  = pls.transform(X_test)
        logger.info(f"PLS-DA applied ({n_components} components)")

    cv_splitter = make_cv_splitter(groups=groups_train)

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "pls":          pls,
        "scaler":       scaler,
        "groups_train": groups_train,
        "groups_test":  groups_test,
        "cv_splitter":  cv_splitter,
    }
