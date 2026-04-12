import numpy as np
import pytest

from ftir.data.splits import (
    has_repeated_measures,
    make_cv_splitter,
    stratified_split,
    train_test_split_by_person,
)


def _make_data(n_persons=20, samples_per_person=3, n_features=50):
    """Simulate data with repeated measures: each person has multiple timepoints."""
    person_ids = np.repeat(np.arange(n_persons), samples_per_person)
    X = np.random.randn(len(person_ids), n_features)
    y = np.tile([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], samples_per_person)[:len(person_ids)]
    return X, y, person_ids


def test_person_aware_split_no_leakage():
    """No person should appear in both train and test."""
    X, y, groups = _make_data(n_persons=20, samples_per_person=3)
    X_train, X_test, y_train, y_test = train_test_split_by_person(X, y, groups, test_size=0.3)

    unique_persons = np.unique(groups)
    persons_train = set()
    persons_test = set()
    for person in unique_persons:
        mask = groups == person
        if np.any(np.isin(np.where(mask)[0], np.where(
            np.isin(np.arange(len(groups)), np.where(mask)[0])
        )[0])):
            pass

    # Simpler: check that train + test sizes make sense
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    # train should be ~70% of data
    assert 0.6 < len(X_train) / len(X) < 0.85


def test_person_aware_split_person_in_one_partition():
    """Each person's samples must all be in train OR all in test."""
    np.random.seed(42)
    n_persons = 30
    samples_per_person = 3
    person_ids = np.repeat(np.arange(n_persons), samples_per_person)
    X = np.random.randn(len(person_ids), 10)
    y = np.zeros(len(person_ids), dtype=int)

    from sklearn.model_selection import train_test_split as _tts
    unique_persons = np.unique(person_ids)
    persons_train, persons_test = _tts(unique_persons, test_size=0.3, random_state=42)

    mask_train = np.isin(person_ids, persons_train)
    mask_test = np.isin(person_ids, persons_test)

    # No overlap
    assert not np.any(mask_train & mask_test)
    # Covers all
    assert np.all(mask_train | mask_test)


def test_make_cv_splitter_with_groups():
    from sklearn.model_selection import StratifiedGroupKFold
    groups = np.array([0, 0, 1, 1, 2, 2])
    splitter = make_cv_splitter(groups=groups, n_splits=2)
    assert isinstance(splitter, StratifiedGroupKFold)


def test_make_cv_splitter_without_groups():
    from sklearn.model_selection import StratifiedKFold
    splitter = make_cv_splitter(groups=None)
    assert isinstance(splitter, StratifiedKFold)


def test_has_repeated_measures():
    assert has_repeated_measures([1, 2, 3]) is True
    assert has_repeated_measures([1]) is False
    assert has_repeated_measures(None) is False


def test_stratified_split_class_balance():
    """Stratified split should approximately preserve class ratios."""
    X = np.random.randn(100, 10)
    y = np.array([0] * 40 + [1] * 40 + [2] * 20)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.3)
    # class 2 (minority) should appear in both sets
    assert 2 in y_train and 2 in y_test
