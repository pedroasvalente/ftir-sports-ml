import numpy as np
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from ftir.config import random_seed as _seed


def train_test_split_by_person(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.3,
    random_state: int = _seed,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split so that all samples from the same person land in the same partition.

    This prevents data leakage when the sedentary group contributes samples
    from multiple timepoints (T1, T2, T3) for the same individual.
    """
    unique_persons = np.unique(groups)
    persons_train, persons_test = train_test_split(
        unique_persons, test_size=test_size, random_state=random_state
    )
    mask_train = np.isin(groups, persons_train)
    mask_test = np.isin(groups, persons_test)
    return X[mask_train], X[mask_test], y[mask_train], y[mask_test]


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = _seed,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard stratified split for single-timepoint data."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def make_cv_splitter(
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = _seed,
):
    """
    Return the appropriate CV splitter based on whether repeated measures exist.

    If groups is provided (person_code array), returns StratifiedGroupKFold
    to ensure all samples from the same person stay in the same fold.
    Otherwise returns StratifiedKFold.
    """
    if groups is not None:
        return StratifiedGroupKFold(n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def has_repeated_measures(timepoints: list[int] | None) -> bool:
    """True when multiple timepoints are used (sedentary group has repeated measures)."""
    return timepoints is not None and len(timepoints) > 1
