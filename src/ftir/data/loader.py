import numpy as np
import pandas as pd

from ftir.config import TRAINING_DATA_PATH, logger
from ftir.data.config import DATA_COLS


def load_data(path: str | None = None) -> pd.DataFrame:
    path = path or TRAINING_DATA_PATH
    return pd.read_csv(path)


def get_ftir_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DATA_COLS]


def filter_sample_data(
    df: pd.DataFrame,
    target: str,
    sample_type: str,
    ftir_columns: list[str],
    group_fam: str | list[str] | None = None,
    timepoints: list[int] | int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Filter data by sample type, optionally by group and timepoint.

    Returns the filtered DataFrame (needed downstream for person_code),
    the feature matrix X, and target array y.
    """
    logger.info(f"Filtering: {sample_type}")
    data = df[df["sample_type"] == sample_type].copy()

    if group_fam is not None:
        if not isinstance(group_fam, list):
            group_fam = [group_fam]
        data = data[data["group_fam"].isin(group_fam)]

    if timepoints is not None:
        if not isinstance(timepoints, list):
            timepoints = [timepoints]
        data = data[data["timepoint"].isin(timepoints)]
        logger.info(f"Samples after timepoint filter: {len(data)}")

    if target not in data.columns:
        y = np.ones(len(data))
    else:
        if data[target].dropna().empty:
            raise ValueError(f"No data for target '{target}' in {sample_type}")
        y = data[target]

    y_valid = y.notna()
    X = data[ftir_columns]
    x_valid = X.notna().all(axis=1) & (X != 0).any(axis=1)
    mask = y_valid & x_valid

    data = data[mask]
    X = X[mask]
    y = y[mask]

    logger.info(f"Samples after filtering: {len(data)}")
    return data, X.values, y.values
