"""Descriptive statistics and bivariate tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean, SD, quartiles, skewness and kurtosis for numeric columns."""
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurt()
    return desc.round(3)


def describe_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Counts and shares for each level of each categorical column."""
    rows = []
    for col in cols:
        counts = df[col].value_counts(sort=False)
        for level, n in counts.items():
            rows.append({"variable": col, "level": level, "n": int(n), "share": n / len(df)})
    return pd.DataFrame(rows).round({"share": 4})


def charges_by_group(df: pd.DataFrame, group_cols: list[str], target: str = "charges") -> pd.DataFrame:
    """Target summary per level of each grouping column."""
    frames = []
    for col in group_cols:
        g = df.groupby(col, observed=True)[target].agg(["count", "mean", "median", "std"])
        g.insert(0, "variable", col)
        g.index.name = "level"
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True).round(2)


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def chi_square_tests(df: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
    """Chi-square test of independence for each categorical pair, with Cramér's V."""
    rows = []
    for a, b in pairs:
        table = pd.crosstab(df[a], df[b])
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        rows.append(
            {
                "pair": f"{a} x {b}",
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
                "cramers_v": cramers_v(table),
                "min_expected": expected.min(),
            }
        )
    return pd.DataFrame(rows).round(4)


def group_difference_tests(df: pd.DataFrame, target: str = "charges") -> pd.DataFrame:
    """Non-parametric tests of charges across groups (charges is strongly right-skewed).

    Two groups: Mann-Whitney U, with rank-biserial correlation as effect size.
    More than two groups: Kruskal-Wallis H, with epsilon-squared as effect size.
    """
    rows = []
    for col in ["smoker", "sex"]:
        levels = list(df[col].cat.categories)
        x = df.loc[df[col] == levels[0], target]
        y = df.loc[df[col] == levels[1], target]
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rbc = 1 - 2 * u / (len(x) * len(y))
        rows.append({"variable": col, "test": "Mann-Whitney U", "statistic": u,
                     "p_value": p, "effect_size": rbc, "effect_measure": "rank-biserial r"})
    n = len(df)
    for col in ["region", "children"]:
        groups = [g[target].to_numpy() for _, g in df.groupby(col, observed=True)]
        h, p = stats.kruskal(*groups)
        eps2 = (h - len(groups) + 1) / (n - len(groups))
        rows.append({"variable": col, "test": "Kruskal-Wallis H", "statistic": h,
                     "p_value": p, "effect_size": eps2, "effect_measure": "epsilon-squared"})
    return pd.DataFrame(rows).round(4)
