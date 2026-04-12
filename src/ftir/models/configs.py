from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ftir.config import random_seed


@dataclass
class ModelConfig:
    name: str
    desc_name: str
    model_cls: Any
    param_grid: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=dict)

    def get_model(self, **kwargs):
        args = {**self.model_kwargs, **kwargs}
        if "random_state" not in args and "random_state" in self.model_cls().get_params():
            args["random_state"] = random_seed
        return self.model_cls(**args)

    def get_params(self, search_type: str = "grid") -> dict:
        if search_type in ("grid", "GridSearchCV"):
            return self.param_grid
        raise ValueError(f"Unknown search_type: {search_type}. Only 'grid' is supported.")

    def get_pipeline_params(self, step_name: str = "classifier") -> dict:
        """Return param_grid with keys prefixed for use inside an imblearn Pipeline."""
        return {f"{step_name}__{k}": v for k, v in self.param_grid.items()}


random_forest = ModelConfig(
    name="random_forest",
    desc_name="Random Forest",
    model_cls=RandomForestClassifier,
    param_grid={
        "n_estimators":      [100, 200, 300],
        "max_depth":         [4, 8, 12, None],
        "max_features":      ["sqrt", "log2"],
        "min_samples_leaf":  [1, 2, 4],
        "criterion":         ["gini", "entropy"],
    },
)

mlp_classifier = ModelConfig(
    name="mlp_classifier",
    desc_name="MLP Classifier",
    model_cls=MLPClassifier,
    model_kwargs={
        "max_iter": 3000,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": random_seed,
    },
    param_grid={
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (64, 64)],
        "activation":         ["tanh", "relu"],
        "solver":             ["adam"],
        "alpha":              [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    },
)

decision_tree = ModelConfig(
    name="decision_tree",
    desc_name="Decision Tree",
    model_cls=DecisionTreeClassifier,
    param_grid={
        "criterion":         ["gini", "entropy"],
        "max_depth":         [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      [None, "sqrt", "log2"],
    },
)

xgboost = ModelConfig(
    name="xgboost",
    desc_name="XGBoost",
    model_cls=XGBClassifier,
    model_kwargs={
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "use_label_encoder": False,
    },
    param_grid={
        "n_estimators":    [100, 200, 300],
        "max_depth":       [3, 6, 9],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.7, 0.85, 1.0],
        "colsample_bytree":[0.7, 0.85, 1.0],
        "min_child_weight":[1, 3, 5],
    },
)

MODEL_REGISTRY = {
    "random_forest":   random_forest,
    "mlp_classifier":  mlp_classifier,
    "decision_tree":   decision_tree,
    "xgboost":         xgboost,
}

CV_SEARCH_ARGS = {
    "GridSearchCV": {
        "scoring":   "balanced_accuracy",
        "n_jobs":    -1,
        "refit":     True,
        "verbose":   0,
    },
}
