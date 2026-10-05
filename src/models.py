"""Fit the model ladder, compare on a common dollar scale, cross-validate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold


@dataclass
class FittedModel:
    key: str
    label: str
    formula: str
    scale: str
    result: object          # statsmodels RegressionResults
    smearing: float = 1.0   # Duan smearing factor, only used for log models

    def predict_dollars(self, data: pd.DataFrame) -> np.ndarray:
        pred = np.asarray(self.result.predict(data))
        if self.scale == "log":
            return np.exp(pred) * self.smearing
        return pred


def fit_one(key: str, spec: dict, data: pd.DataFrame) -> FittedModel:
    res = smf.ols(spec["formula"], data=data).fit()
    smear = float(np.mean(np.exp(res.resid))) if spec["scale"] == "log" else 1.0
    return FittedModel(key, spec["label"], spec["formula"], spec["scale"], res, smear)


def fit_ladder(model_specs: dict, data: pd.DataFrame) -> dict[str, FittedModel]:
    return {k: fit_one(k, v, data) for k, v in model_specs.items()}


def _dollar_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "r2_dollars": 1 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
    }


def compare_in_sample(models: dict[str, FittedModel], data: pd.DataFrame, target: str) -> pd.DataFrame:
    """In-sample fit. R2/adj-R2/AIC are on each model's own scale; *_dollars are comparable."""
    y = data[target].to_numpy()
    rows = []
    for m in models.values():
        r = m.result
        row = {
            "model": m.label,
            "scale": m.scale,
            "n_params": int(r.df_model) + 1,
            "r2_own_scale": r.rsquared,
            "adj_r2_own_scale": r.rsquared_adj,
            "aic_own_scale": r.aic,
        }
        row.update(_dollar_metrics(y, m.predict_dollars(data)))
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def cross_validate(model_specs: dict, data: pd.DataFrame, target: str,
                   n_splits: int, shuffle: bool, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-fold CV, every model scored in dollars on the same folds."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    fold_rows = []
    for fold, (tr, te) in enumerate(kf.split(data), start=1):
        train, test = data.iloc[tr], data.iloc[te]
        y = test[target].to_numpy()
        for key, spec in model_specs.items():
            m = fit_one(key, spec, train)
            metrics = _dollar_metrics(y, m.predict_dollars(test))
            fold_rows.append({"fold": fold, "model": spec["label"], **metrics})
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby("model", sort=False)[["r2_dollars", "rmse", "mae"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"cv_{a}_{b}" for a, b in summary.columns]
    return summary.reset_index(), folds.round(4)
