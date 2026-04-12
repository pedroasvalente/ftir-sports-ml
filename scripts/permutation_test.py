#!/usr/bin/env python3
"""
Permutation significance test for FTIR-ML classifiers.

For each (matrix, model, timepoints) combination in results_summary.csv,
tests whether the observed balanced accuracy is significantly better than
chance by running the model N times with randomly permuted labels.

Usage (inside Docker):
    docker compose run --rm --entrypoint python3 train \
        /app/scripts/permutation_test.py \
        --config experiments/configs/study1_group_fam_v2.json \
        --n-permutations 100

Output:
    results/<run_name>/permutation_test.csv
    columns: sample_type, timepoints, model, observed_ba, perm_mean_ba,
             perm_std_ba, p_value, significant_0.05
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src importable
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ftir.config import RESULTS_DIR, random_seed
from ftir.data.loader import filter_sample_data, get_ftir_columns, load_data
from ftir.data.splits import has_repeated_measures
from ftir.models.configs import MODEL_REGISTRY
from ftir.models.evaluation import evaluate
from ftir.preprocessing.pipeline import preprocess
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score


def _single_permutation(
    X_train, y_train, X_test, y_test,
    model_name, cv_splitter, groups_train, apply_smote, n_classes, rng,
):
    """Run one permutation: shuffle y_train, fit, evaluate."""
    from ftir.models.configs import MODEL_REGISTRY, CV_SEARCH_ARGS
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    y_perm = rng.permutation(y_train)
    config = MODEL_REGISTRY[model_name]

    if apply_smote and n_classes > 1:
        estimator  = ImbPipeline([
            ("smote",      SMOTE(random_state=int(rng.integers(0, 9999)))),
            ("classifier", config.get_model()),
        ])
        param_grid = config.get_pipeline_params("classifier")
    else:
        estimator  = config.get_model()
        param_grid = config.get_params("grid")

    search_args = {**CV_SEARCH_ARGS["GridSearchCV"], "cv": cv_splitter}
    search = GridSearchCV(estimator, param_grid, **search_args)

    fit_kwargs = {}
    if groups_train is not None:
        fit_kwargs["groups"] = groups_train

    search.fit(X_train, y_perm, **fit_kwargs)
    y_pred = search.best_estimator_.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)


def run_permutation_test(config_path: str, n_permutations: int = 100):
    with open(config_path) as f:
        cfg = json.load(f)

    run_slug = cfg.get("run_name", "run")
    results_csv = Path(RESULTS_DIR) / run_slug / "results_summary.csv"
    if not results_csv.exists():
        sys.exit(f"❌  {results_csv} not found. Run training first.")

    results = pd.read_csv(results_csv)
    df = load_data()
    ftir_cols   = get_ftir_columns(df)
    wavenumbers = np.array([float(c) for c in ftir_cols])

    rng = np.random.default_rng(random_seed)
    rows = []

    for _, row in results.iterrows():
        sample_type  = row["sample_type"]
        model_name   = next(k for k, v in MODEL_REGISTRY.items() if v.desc_name == row["model"])
        timepoints   = eval(row["timepoints"]) if row["timepoints"] not in ("None", "nan") else None
        apply_pls    = bool(row.get("apply_pls", True))
        apply_smote  = bool(row.get("apply_smote", True))
        scale        = bool(row.get("scale", True))
        n_components = int(cfg.get("n_components", [3])[0])
        num_classes  = int(cfg.get("num_classes", [3])[0])
        observed_ba  = float(row["balanced_accuracy"])

        print(f"  Permutation test: {sample_type} | {row['model']} | tp={timepoints} "
              f"({n_permutations} permutations)...")

        try:
            data_df, X, y_raw = filter_sample_data(
                df=df, target="group_fam", sample_type=sample_type,
                ftir_columns=ftir_cols, timepoints=timepoints,
            )
        except ValueError as e:
            print(f"    ⚠️  Skipped: {e}")
            continue

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y  = le.fit_transform(y_raw)
        groups = data_df["person_code"].values if has_repeated_measures(timepoints) else None

        prep = preprocess(
            X=X, y=y, groups=groups, timepoints=timepoints,
            test_size=1 - float(row.get("train_pct", 0.7)),
            scale=scale, apply_pls=apply_pls, n_components=n_components,
            num_classes=num_classes, random_state=random_seed,
        )

        perm_bas = []
        for i in range(n_permutations):
            ba = _single_permutation(
                prep["X_train"], prep["y_train"], prep["X_test"], prep["y_test"],
                model_name, prep["cv_splitter"], prep["groups_train"],
                apply_smote, num_classes, rng,
            )
            perm_bas.append(ba)

        perm_bas = np.array(perm_bas)
        p_value  = (perm_bas >= observed_ba).mean()

        rows.append({
            "sample_type":    sample_type,
            "timepoints":     str(timepoints),
            "model":          row["model"],
            "observed_ba":    observed_ba,
            "perm_mean_ba":   perm_bas.mean(),
            "perm_std_ba":    perm_bas.std(),
            "p_value":        p_value,
            "significant_0.05": p_value < 0.05,
        })
        print(f"    observed={observed_ba:.3f}  perm_mean={perm_bas.mean():.3f}  "
              f"p={p_value:.3f}  {'✅' if p_value < 0.05 else '❌'}")

    out = Path(RESULTS_DIR) / run_slug / "permutation_test.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n✅  Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--n-permutations", type=int, default=100)
    args = parser.parse_args()
    run_permutation_test(args.config, args.n_permutations)
