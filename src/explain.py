"""SHAP explanations for the final OLS model.

For a linear model with an independent masker, SHAP values are exactly
coef_j * (x_ij - mean(x_j)). So SHAP adds no new information over the coefficients;
its value here is presentation: it shows, person by person, how many dollars each
term moves the prediction away from the average prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

PRETTY = {
    "sex[T.male]": "sex = male",
    "smoker[T.yes]": "smoker = yes",
    "region[T.northwest]": "region = northwest",
    "region[T.southeast]": "region = southeast",
    "region[T.southwest]": "region = southwest",
    "age_c": "age (centred)",
    "I(age_c ** 2)": "age squared (centred)",
    "obese:smoker[T.yes]": "obese x smoker",
}


def shap_for_ols(result, background_size: int, seed: int) -> shap.Explanation:
    names = [n for n in result.model.exog_names if n != "Intercept"]
    X = pd.DataFrame(result.model.exog, columns=result.model.exog_names)[names]
    coef = result.params[names].to_numpy()
    intercept = float(result.params["Intercept"])
    rng = np.random.default_rng(seed)
    bg = X.iloc[rng.choice(len(X), size=min(background_size, len(X)), replace=False)]
    explainer = shap.LinearExplainer((coef, intercept), shap.maskers.Independent(bg, max_samples=len(bg)))
    sv = explainer(X)
    sv.feature_names = [PRETTY.get(n, n) for n in names]
    return sv


def mean_abs_shap(sv: shap.Explanation) -> pd.DataFrame:
    vals = np.abs(sv.values).mean(axis=0)
    return (
        pd.DataFrame({"term": sv.feature_names, "mean_abs_shap_usd": vals})
        .sort_values("mean_abs_shap_usd", ascending=False)
        .round(2)
        .reset_index(drop=True)
    )
