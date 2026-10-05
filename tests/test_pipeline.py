"""Tests against the real dataset (no synthetic fixtures)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import data_processor, diagnostics, eda, models

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


@pytest.fixture(scope="module")
def raw():
    return data_processor.load_raw(ROOT / CFG["data"]["raw_path"], CFG["data"]["source_url"])


@pytest.fixture(scope="module")
def df(raw):
    return data_processor.prepare(raw, True, CFG["data"]["obesity_threshold"])


def test_raw_schema_and_known_shape(raw):
    report = data_processor.validate(raw)
    assert report["n_rows"] == 1338
    assert report["n_missing"] == 0
    assert report["n_duplicate_rows"] == 1


def test_prepare_drops_duplicate_and_adds_features(df):
    assert len(df) == 1337
    assert set(df["obese"].unique()) == {0, 1}
    assert (df.loc[df["obese"] == 1, "bmi"] >= 30).all()
    assert np.allclose(np.exp(df["log_charges"]), df["charges"])
    assert list(df["smoker"].cat.categories) == ["no", "yes"]


def test_validate_rejects_bad_levels(raw):
    bad = raw.copy()
    bad.loc[0, "smoker"] = "sometimes"
    with pytest.raises(ValueError):
        data_processor.validate(bad)


def test_chi_square_outputs_valid(df):
    out = eda.chi_square_tests(df, CFG["eda"]["chi_square_pairs"])
    assert out["p_value"].between(0, 1).all()
    assert out["cramers_v"].between(0, 1).all()


def test_log_model_smearing_above_one(df):
    m = models.fit_one("M3_log", CFG["models"]["M3_log"], df)
    assert m.smearing > 1.0          # Jensen: E[exp(e)] > exp(E[e]) = 1
    assert (m.predict_dollars(df) > 0).all()


def test_interaction_model_beats_baseline_out_of_sample(df):
    specs = {k: CFG["models"][k] for k in ["M0_baseline", "M3_interaction"]}
    summary, _ = models.cross_validate(specs, df, "charges", 5, True, CFG["seed"])
    rmse = summary.set_index("model")["cv_rmse_mean"]
    assert rmse[CFG["models"]["M3_interaction"]["label"]] < rmse[CFG["models"]["M0_baseline"]["label"]]


def test_cv_is_reproducible(df):
    specs = {"M0_baseline": CFG["models"]["M0_baseline"]}
    a, _ = models.cross_validate(specs, df, "charges", 5, True, 42)
    b, _ = models.cross_validate(specs, df, "charges", 5, True, 42)
    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_reproducible_and_centered(df):
    res = models.fit_one("M3_interaction", CFG["models"]["M3_interaction"], df).result
    b1 = diagnostics.bootstrap_coefs(res, 200, 7)
    b2 = diagnostics.bootstrap_coefs(res, 200, 7)
    pd.testing.assert_frame_equal(b1, b2)
    rel = (b1.mean() - res.params).abs() / res.bse
    assert (rel < 1.0).all()
