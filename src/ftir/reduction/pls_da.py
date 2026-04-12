import numpy as np
from sklearn.cross_decomposition import PLSRegression


class PLSDA:
    """
    PLS-DA wrapper with proper VIP score calculation.

    VIP (Variable Importance in Projection) scores are the standard metric
    for identifying important wavenumbers in PLS-DA — preferred over the
    back-projection approach (model importance × loadings) used previously.

    A VIP score > 1 is conventionally considered important.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.model = PLSRegression(n_components=n_components)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PLSDA":
        self.model.fit(X_train, y_train)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.transform(X_train)

    def vip_scores(self) -> np.ndarray:
        """
        Compute VIP scores for all input variables (wavenumbers).

        Formula:
            VIP_j = sqrt(p * sum_h(W*_jh^2 * SSY_h) / SSY_total)

        where SSY_h = variance of Y explained by the h-th component,
        W*_jh = normalised weight of variable j in component h.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before vip_scores()")

        T = self.model.x_scores_     # (n_samples, n_components)
        W = self.model.x_weights_    # (n_features, n_components)
        Q = self.model.y_loadings_   # (n_targets, n_components)

        p = W.shape[0]
        s = np.diag(T.T @ T @ Q.T @ Q)  # SSY explained per component
        total_s = s.sum()

        w_norm = W / np.linalg.norm(W, axis=0, keepdims=True)  # normalised weights
        vips = np.sqrt(p * (w_norm**2 @ s) / total_s)
        return vips

    @property
    def x_weights_(self) -> np.ndarray:
        return self.model.x_weights_
