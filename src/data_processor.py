"""Load, validate and prepare the Medical Cost Personal dataset."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
EXPECTED_LEVELS = {
    "sex": {"female", "male"},
    "smoker": {"no", "yes"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}


def load_raw(path: str | Path, source_url: str | None = None) -> pd.DataFrame:
    """Read the CSV, downloading it first if it is not on disk."""
    path = Path(path)
    if not path.exists():
        if source_url is None:
            raise FileNotFoundError(f"{path} not found and no source_url given")
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading dataset from %s", source_url)
        urllib.request.urlretrieve(source_url, path)
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    """Check schema and value ranges. Raises on hard failures, returns a report dict."""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    for col, levels in EXPECTED_LEVELS.items():
        unexpected = set(df[col].unique()) - levels
        if unexpected:
            raise ValueError(f"Unexpected levels in {col}: {sorted(unexpected)}")

    if (df["charges"] <= 0).any():
        raise ValueError("Non-positive charges found; log transform would fail")
    if not df["bmi"].between(10, 80).all():
        raise ValueError("BMI outside plausible range [10, 80]")
    if not df["age"].between(0, 120).all():
        raise ValueError("Age outside plausible range [0, 120]")

    dup_mask = df.duplicated(keep=False)
    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows": df[dup_mask].to_dict(orient="records"),
    }
    return report


def prepare(df: pd.DataFrame, drop_duplicates: bool = True, obesity_threshold: float = 30.0,
            center_age: bool = True) -> pd.DataFrame:
    """Drop exact duplicates, add engineered columns, set categorical order."""
    out = df.copy()
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        log.info("Dropped %d exact duplicate row(s)", before - len(out))

    out["obese"] = (out["bmi"] >= obesity_threshold).astype(int)
    age_mean = out["age"].mean() if center_age else 0.0
    out["age_c"] = out["age"] - age_mean
    out.attrs["age_mean"] = float(age_mean)
    out["log_charges"] = np.log(out["charges"].astype(float))

    # Fixed reference levels so coefficients always read against the same baseline
    out["sex"] = pd.Categorical(out["sex"], categories=["female", "male"])
    out["smoker"] = pd.Categorical(out["smoker"], categories=["no", "yes"])
    out["region"] = pd.Categorical(
        out["region"], categories=["northeast", "northwest", "southeast", "southwest"]
    )
    return out
