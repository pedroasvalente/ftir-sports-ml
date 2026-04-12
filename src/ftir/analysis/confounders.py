"""
Confounder analysis: test whether age and body composition drive
the PCA/ML separation rather than sport group membership.

Addresses Reviewer 2, Major Point 1.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def partial_eta_squared(groups: np.ndarray, values: np.ndarray) -> float:
    """
    Partial eta-squared: variance explained by group membership.
    Uses one-way ANOVA decomposition.
    """
    grand_mean = values.mean()
    unique_groups = np.unique(groups)
    ss_between = sum(
        np.sum(groups == g) * (values[groups == g].mean() - grand_mean) ** 2
        for g in unique_groups
    )
    ss_total = np.sum((values - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def variance_explained_by_covariates(
    df: pd.DataFrame,
    pc_scores: np.ndarray,
    activity_col: str = "group_fam",
    covariates: list[str] | None = None,
    n_pcs: int = 4,
) -> pd.DataFrame:
    """
    For each of the first n_pcs PCs, fit two OLS models:
      1. PC ~ activity_group only
      2. PC ~ activity_group + covariates
    Report partial eta-squared and change in R² when covariates are added.

    Addresses the reviewer's request to quantify how much variance is explained
    by age/body composition vs. activity group.
    """
    if covariates is None:
        covariates = ["age_years", "ffm_kg", "bodyfat_perc"]

    rows = []
    for i in range(min(n_pcs, pc_scores.shape[1])):
        col = f"PC{i+1}"
        tmp = df[[activity_col] + [c for c in covariates if c in df.columns]].copy()
        tmp[col] = pc_scores[:, i]
        tmp = tmp.dropna()

        formula_base = f"{col} ~ C({activity_col})"
        cov_cols = [c for c in covariates if c in tmp.columns]
        formula_full = f"{col} ~ C({activity_col}) + {' + '.join(cov_cols)}" if cov_cols else formula_base

        m_base = smf.ols(formula_base, data=tmp).fit()
        m_full = smf.ols(formula_full, data=tmp).fit()

        eta2 = partial_eta_squared(tmp[activity_col].values, tmp[col].values)
        rows.append({
            "PC": col,
            "R2_activity_only": m_base.rsquared,
            "R2_activity_plus_covariates": m_full.rsquared,
            "delta_R2": m_full.rsquared - m_base.rsquared,
            "partial_eta2_activity": eta2,
            "p_activity": m_base.f_pvalue,
        })

    return pd.DataFrame(rows)


def stratified_analysis_by_covariate(
    df: pd.DataFrame,
    covariate: str,
    activity_col: str = "group_fam",
    n_strata: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Split into n_strata quantile groups of the covariate and return
    a sub-dataframe per stratum. Used to check separation persists
    across age or body-fat strata.
    """
    df = df.copy()
    df["_stratum"] = pd.qcut(df[covariate], q=n_strata, labels=False, duplicates="drop")
    return {f"stratum_{k}": grp.drop(columns="_stratum") for k, grp in df.groupby("_stratum")}


def ancova_test(
    df: pd.DataFrame,
    pc_score: np.ndarray,
    activity_col: str = "group_fam",
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """
    ANCOVA: test group differences in PC scores after controlling for covariates.
    Returns the ANOVA table from statsmodels.
    """
    if covariates is None:
        covariates = ["age_years", "ffm_kg"]

    tmp = df[[activity_col] + [c for c in covariates if c in df.columns]].copy()
    tmp["score"] = pc_score
    tmp = tmp.dropna()

    cov_cols = [c for c in covariates if c in tmp.columns]
    formula = f"score ~ C({activity_col}) + {' + '.join(cov_cols)}" if cov_cols else f"score ~ C({activity_col})"
    model = smf.ols(formula, data=tmp).fit()

    from statsmodels.stats.anova import anova_lm
    return anova_lm(model, typ=2)
