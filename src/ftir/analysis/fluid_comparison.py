"""
Cross-biofluid comparison: effect sizes and variability.

Addresses Reviewer 2, Major Point 3:
"compare differences between fluids, magnitude of inter-fluid differences
vs differences between activity groups"
"""

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between two groups."""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def pca_centroid_distances(
    scores_per_fluid: dict[str, np.ndarray],
    groups_per_fluid: dict[str, np.ndarray],
    group_col: str = "group_fam",
) -> pd.DataFrame:
    """
    Euclidean distance between group centroids in PC space, per fluid.

    Returns a DataFrame with columns: fluid, group_a, group_b, distance_pc1pc2
    """
    rows = []
    for fluid, scores in scores_per_fluid.items():
        groups = groups_per_fluid[fluid]
        unique = np.unique(groups)
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                g_a, g_b = unique[i], unique[j]
                c_a = scores[groups == g_a, :2].mean(axis=0)
                c_b = scores[groups == g_b, :2].mean(axis=0)
                dist = float(np.linalg.norm(c_a - c_b))
                rows.append({
                    "fluid": fluid,
                    "group_a": g_a,
                    "group_b": g_b,
                    "centroid_distance_pc1pc2": dist,
                })
    return pd.DataFrame(rows)


def within_vs_between_variability(
    scores_per_fluid: dict[str, np.ndarray],
    groups_per_fluid: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Ratio of within-group to between-group variance in PC1 scores.
    Lower ratio = better discrimination.
    """
    rows = []
    for fluid, scores in scores_per_fluid.items():
        groups = groups_per_fluid[fluid]
        pc1 = scores[:, 0]
        grand_mean = pc1.mean()
        unique = np.unique(groups)

        ss_between = sum(
            np.sum(groups == g) * (pc1[groups == g].mean() - grand_mean) ** 2
            for g in unique
        )
        ss_within = sum(
            np.sum((pc1[groups == g] - pc1[groups == g].mean()) ** 2)
            for g in unique
        )
        rows.append({
            "fluid": fluid,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "ratio_within_between": ss_within / ss_between if ss_between > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def effect_sizes_across_fluids(
    scores_per_fluid: dict[str, np.ndarray],
    groups_per_fluid: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Cohen's d for PC1 between each pair of activity groups, per fluid.
    Allows semi-quantitative comparison across biofluids.
    """
    rows = []
    for fluid, scores in scores_per_fluid.items():
        groups = groups_per_fluid[fluid]
        unique = np.unique(groups)
        pc1 = scores[:, 0]
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                g_a, g_b = unique[i], unique[j]
                d = cohens_d(pc1[groups == g_a], pc1[groups == g_b])
                rows.append({
                    "fluid": fluid,
                    "group_a": g_a,
                    "group_b": g_b,
                    "cohens_d_pc1": d,
                })
    return pd.DataFrame(rows)
