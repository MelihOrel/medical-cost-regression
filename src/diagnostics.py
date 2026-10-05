"""Assumption diagnostics and inference remediation for an OLS fit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


def assumption_tests(result) -> pd.DataFrame:
    """Normality, homoscedasticity, independence and functional-form tests."""
    resid = result.resid
    exog = result.model.exog
    jb_stat, jb_p = stats.jarque_bera(resid)[:2]
    sw_stat, sw_p = stats.shapiro(resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)
    w_stat, w_p = white_special(result)
    reset = linear_reset(result, power=2, use_f=True)
    rows = [
        ("Normality", "Jarque-Bera", jb_stat, jb_p),
        ("Normality", "Shapiro-Wilk", sw_stat, sw_p),
        ("Homoscedasticity", "Breusch-Pagan", bp_stat, bp_p),
        ("Homoscedasticity", "White (special form)", w_stat, w_p),
        ("Functional form", "Ramsey RESET (power 2)", float(reset.fvalue), float(reset.pvalue)),
        ("Independence", "Durbin-Watson", durbin_watson(resid), np.nan),
    ]
    out = pd.DataFrame(rows, columns=["assumption", "test", "statistic", "p_value"])
    out["skew_resid"] = np.nan
    out.loc[0, "skew_resid"] = stats.skew(resid)
    return out.round(4)


def white_special(result) -> tuple[float, float]:
    """White's test in its special form: regress e^2 on fitted and fitted^2.

    The full White test squares every regressor; with dummy variables (obese^2 == obese)
    the auxiliary design becomes rank-deficient, so the special form is used instead.
    """
    yhat = np.asarray(result.fittedvalues)
    aux = sm.OLS(np.asarray(result.resid) ** 2, sm.add_constant(np.column_stack([yhat, yhat**2]))).fit()
    lm = result.nobs * aux.rsquared
    return float(lm), float(stats.chi2.sf(lm, 2))


def vif_table(result) -> pd.DataFrame:
    """Variance inflation factors for every non-intercept column of the design matrix."""
    exog = result.model.exog
    names = result.model.exog_names
    rows = [
        {"term": names[i], "vif": variance_inflation_factor(exog, i)}
        for i in range(exog.shape[1])
        if names[i] != "Intercept"
    ]
    return pd.DataFrame(rows).round(3)


def influence_table(result, data: pd.DataFrame, threshold_factor: float = 4.0) -> tuple[pd.DataFrame, float]:
    """Cook's distance, leverage and studentized residuals; rows above 4/n are flagged."""
    infl = OLSInfluence(result)
    n = int(result.nobs)
    threshold = threshold_factor / n
    out = data.copy()
    out["cooks_d"] = infl.cooks_distance[0]
    out["leverage"] = infl.hat_matrix_diag
    out["student_resid"] = infl.resid_studentized_external
    out["flag_influential"] = out["cooks_d"] > threshold
    return out, threshold


def high_residual_profile(result, data: pd.DataFrame, threshold_usd: float) -> pd.DataFrame:
    """Who are the people the model under-predicts by more than `threshold_usd`?"""
    df = data.copy()
    df["resid"] = result.resid.to_numpy()
    df["high_resid"] = df["resid"] > threshold_usd
    prof = (
        df.groupby(["smoker", "high_resid"], observed=True)
        .agg(n=("charges", "size"), mean_charges=("charges", "mean"),
             mean_age=("age", "mean"), mean_bmi=("bmi", "mean"), mean_resid=("resid", "mean"))
        .reset_index()
    )
    return prof.round(2)


def bootstrap_coefs(result, reps: int, seed: int) -> pd.DataFrame:
    """Case-resampling (pairs) bootstrap of OLS coefficients. Robust to non-normal errors."""
    rng = np.random.default_rng(seed)
    X = result.model.exog
    y = result.model.endog
    n = len(y)
    draws = np.empty((reps, X.shape[1]))
    for b in range(reps):
        idx = rng.integers(0, n, n)
        draws[b] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    return pd.DataFrame(draws, columns=result.model.exog_names)


def coefficient_table(result, robust_cov: str, boot: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Side-by-side CIs: classical OLS, heteroskedasticity-robust, and bootstrap percentile."""
    robust = result.get_robustcov_results(cov_type=robust_cov)
    ci_naive = result.conf_int(alpha=alpha)
    ci_rob = pd.DataFrame(robust.conf_int(alpha=alpha), index=result.params.index)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = pd.DataFrame(
        {
            "coef": result.params,
            "se_ols": result.bse,
            "p_ols": result.pvalues,
            "ci_low_ols": ci_naive[0],
            "ci_high_ols": ci_naive[1],
            f"se_{robust_cov.lower()}": pd.Series(robust.bse, index=result.params.index),
            f"p_{robust_cov.lower()}": pd.Series(robust.pvalues, index=result.params.index),
            f"ci_low_{robust_cov.lower()}": ci_rob[0],
            f"ci_high_{robust_cov.lower()}": ci_rob[1],
            "se_boot": boot.std(ddof=1),
            "ci_low_boot": boot.quantile(lo / 100),
            "ci_high_boot": boot.quantile(hi / 100),
        }
    )
    out.index.name = "term"
    return out.reset_index().round(4)
