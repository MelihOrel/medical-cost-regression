"""Run the full pipeline: data -> EDA -> model ladder -> diagnostics -> inference -> SHAP.

Usage:
    python main.py                 # uses config.yaml
    python main.py --config other.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import yaml

from src import data_processor, diagnostics, eda, explain, models, visualize

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")


def main(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)

    fig_dir = Path(cfg["outputs"]["figures_dir"])
    tab_dir = Path(cfg["outputs"]["tables_dir"])
    tab_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg["outputs"]["dpi"]
    dc = cfg["data"]
    target = dc["target"]

    # 1. Data -----------------------------------------------------------------
    raw = data_processor.load_raw(dc["raw_path"], dc["source_url"])
    report = data_processor.validate(raw)
    log.info("Raw data: %d rows, %d missing, %d duplicate", report["n_rows"], report["n_missing"],
             report["n_duplicate_rows"])
    df = data_processor.prepare(raw, dc["drop_duplicates"], dc["obesity_threshold"], dc["center_age"])
    log.info("Analysis set: %d rows", len(df))

    # 2. Descriptive statistics and bivariate tests ----------------------------
    eda.describe_numeric(df, dc["numeric"] + [target]).to_csv(tab_dir / "01_descriptive_numeric.csv")
    eda.describe_categorical(df, dc["categorical"] + ["obese"]).to_csv(tab_dir / "02_descriptive_categorical.csv", index=False)
    eda.charges_by_group(df, ["smoker", "sex", "region", "children", "obese"], target).to_csv(
        tab_dir / "03_charges_by_group.csv", index=False)
    chi = eda.chi_square_tests(df, cfg["eda"]["chi_square_pairs"])
    chi.to_csv(tab_dir / "04_chi_square_tests.csv", index=False)
    grp = eda.group_difference_tests(df, target)
    grp.to_csv(tab_dir / "05_group_difference_tests.csv", index=False)

    visualize.set_style()
    visualize.charges_distribution(df, fig_dir, dpi)
    visualize.charges_by_smoker(df, fig_dir, dpi)
    visualize.age_vs_charges(df, fig_dir, dpi)
    visualize.bmi_vs_charges(df, fig_dir, dpi, dc["obesity_threshold"])
    visualize.charges_by_other_groups(df, fig_dir, dpi)
    visualize.correlation_heatmap(df, fig_dir, dpi)
    log.info("EDA tables and figures written")

    # 3. Model ladder ---------------------------------------------------------
    specs = cfg["models"]
    fitted = models.fit_ladder(specs, df)
    in_sample = models.compare_in_sample(fitted, df, target)
    in_sample.to_csv(tab_dir / "06_model_comparison_in_sample.csv", index=False)
    cv_sum, cv_folds = models.cross_validate(specs, df, target, cfg["cv"]["n_splits"],
                                             cfg["cv"]["shuffle"], seed)
    cv_sum.to_csv(tab_dir / "07_model_comparison_cv.csv", index=False)
    cv_folds.to_csv(tab_dir / "07b_cv_folds.csv", index=False)
    visualize.model_ladder(cv_sum, fig_dir, dpi)

    final = fitted[cfg["final_model"]]
    for k in ["M0_baseline", cfg["final_model"]]:
        (tab_dir / f"08_summary_{k}.txt").write_text(str(fitted[k].result.summary()))
    log.info("Final model %s: adj-R² %.4f", final.key, final.result.rsquared_adj)

    # 4. Diagnostics ----------------------------------------------------------
    dg = cfg["diagnostics"]
    tests = diagnostics.assumption_tests(final.result)
    tests.to_csv(tab_dir / "09_assumption_tests.csv", index=False)
    vif_base = diagnostics.vif_table(fitted["M0_baseline"].result)
    vif_base.to_csv(tab_dir / "10_vif_baseline.csv", index=False)
    diagnostics.vif_table(final.result).to_csv(tab_dir / "10b_vif_final.csv", index=False)
    infl, cooks_thr = diagnostics.influence_table(final.result, df, dg["cooks_threshold_factor"])
    infl[infl["flag_influential"]].sort_values("cooks_d", ascending=False).to_csv(
        tab_dir / "11_influential_points.csv", index=False)
    hr = diagnostics.high_residual_profile(final.result, df, dg["high_residual_usd"])
    hr.to_csv(tab_dir / "12_high_residual_profile.csv", index=False)
    visualize.diagnostics_panel(final.result, df, cooks_thr, fig_dir, dpi)
    visualize.residuals_by_smoker(final.result, df, dg["high_residual_usd"], fig_dir, dpi)

    # 5. Inference: classical vs robust vs bootstrap ---------------------------
    inf = cfg["inference"]
    boot = diagnostics.bootstrap_coefs(final.result, inf["bootstrap_reps"], seed)
    coef_tab = diagnostics.coefficient_table(final.result, inf["robust_cov"], boot, inf["alpha"])
    coef_tab.to_csv(tab_dir / "13_coefficients_three_ci.csv", index=False)
    visualize.coefficient_ci_grid(coef_tab, inf["robust_cov"], fig_dir, dpi)
    visualize.actual_vs_predicted(fitted, ["M0_baseline", cfg["final_model"], "M3_log"], df, fig_dir, dpi)

    # 6. SHAP -----------------------------------------------------------------
    sv = explain.shap_for_ols(final.result, cfg["shap"]["background_size"], seed)
    explain.mean_abs_shap(sv).to_csv(tab_dir / "14_mean_abs_shap.csv", index=False)
    visualize.shap_beeswarm(sv, fig_dir, dpi)

    # 7. Run summary ----------------------------------------------------------
    summary = {
        "n_raw": report["n_rows"],
        "n_duplicates_dropped": report["n_duplicate_rows"],
        "n_analysis": int(len(df)),
        "age_mean_used_for_centering": round(df.attrs["age_mean"], 4),
        "final_model": final.key,
        "final_r2": round(float(final.result.rsquared), 4),
        "final_adj_r2": round(float(final.result.rsquared_adj), 4),
        "cv": cv_sum.to_dict(orient="records"),
        "in_sample": in_sample.to_dict(orient="records"),
        "assumption_tests": tests.to_dict(orient="records"),
        "n_influential": int(infl["flag_influential"].sum()),
        "cooks_threshold": round(cooks_thr, 5),
        "chi_square": chi.to_dict(orient="records"),
        "group_tests": grp.to_dict(orient="records"),
    }
    (tab_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Done. Figures -> %s, tables -> %s", fig_dir, tab_dir)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
