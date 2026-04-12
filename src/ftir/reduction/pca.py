import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    X: np.ndarray,
    n_components: int | None = None,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run PCA on spectral data.

    Returns
    -------
    scores : (n_samples, n_components)
    loadings : (n_components, n_features) — contribution of each wavenumber per PC
    explained_variance_ratio : (n_components,)
    """
    if scale:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    loadings = pca.components_
    return scores, loadings, pca.explained_variance_ratio_


def pca_results_df(
    scores: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int | None = None,
) -> pd.DataFrame:
    """
    Combine PCA scores with sample metadata for plotting.

    metadata must have the same index as the original filtered dataframe.
    """
    n = scores.shape[1] if n_components is None else n_components
    pc_cols = {f"PC{i+1}": scores[:, i] for i in range(n)}
    result = pd.DataFrame(pc_cols, index=metadata.index)
    return pd.concat([metadata.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
