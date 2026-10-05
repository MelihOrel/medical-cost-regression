"""All figures. One consistent style, 300 dpi, colour follows the entity (non-smoker / smoker)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# Palette (validated categorical slots 1-2, ink and surface tokens)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
NEUTRAL = "#8a8984"
BLUE = "#2a78d6"     # slot 1: non-smoker
ORANGE = "#eb6834"   # slot 2: smoker
VIOLET = "#4a3aa7"   # slot 7: used only as a third CI method in the coefficient plot
SMOKER_COLORS = {"no": BLUE, "yes": ORANGE}
SMOKER_LABELS = {"no": "Non-smoker", "yes": "Smoker"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b42f2f"])


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def _usd(ax, axis: str = "y") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _save(fig, out_dir: Path, name: str, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _note(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, color=INK_2, wrap=True)


# EDA -------------------------------------------------------------------------

def charges_distribution(df, out_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title, xl in [
        (axes[0], "charges", "Charges are strongly right-skewed", "Charges (USD)"),
        (axes[1], "log_charges", "log(charges) is close to symmetric", "log(charges)"),
    ]:
        ax.hist(df[col], bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel("Number of people")
        ax.text(0.98, 0.95, f"skew = {stats.skew(df[col]):.2f}", transform=ax.transAxes,
                ha="right", va="top", color=INK, fontsize=9)
    _usd(axes[0], "x")
    _note(fig, "A symmetric target does not make a log model better. That is tested, not assumed (see model ladder).")
    return _save(fig, out_dir, "01_charges_distribution", dpi)


def charges_by_smoker(df, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    for i, lvl in enumerate(["no", "yes"]):
        y = df.loc[df["smoker"] == lvl, "charges"].to_numpy()
        ax.boxplot(y, positions=[i], widths=0.45, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=INK_2, linewidth=1.2),
                   medianprops=dict(color=INK, linewidth=2), whiskerprops=dict(color=INK_2),
                   capprops=dict(color=INK_2))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(y)), y, s=9, alpha=0.45,
                   color=SMOKER_COLORS[lvl], linewidths=0)
        ax.text(i + 0.3, np.median(y), f"median ${np.median(y):,.0f}\nn = {len(y)}",
                va="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1], [SMOKER_LABELS["no"], SMOKER_LABELS["yes"]])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel("Charges (USD)")
    _usd(ax)
    ratio = df.loc[df["smoker"] == "yes", "charges"].median() / df.loc[df["smoker"] == "no", "charges"].median()
    ax.set_title(f"Smokers' median charges are {ratio:.1f}x non-smokers'")
    return _save(fig, out_dir, "02_charges_by_smoker", dpi)


def _scatter_by_smoker(ax, df, x):
    for lvl in ["no", "yes"]:
        d = df[df["smoker"] == lvl]
        ax.scatter(d[x], d["charges"], s=14, alpha=0.6, color=SMOKER_COLORS[lvl],
                   edgecolor=SURFACE, linewidth=0.4, label=SMOKER_LABELS[lvl])
    _usd(ax)
    ax.set_ylabel("Charges (USD)")
    ax.legend(loc="upper left")


def age_vs_charges(df, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(8, 5))
    _scatter_by_smoker(ax, df, "age")
    ax.set_xlabel("Age (years)")
    ax.set_title("Three parallel bands: cost rises with age at every level")
    _note(fig, "Bottom band: mostly non-smokers. Top band: smokers. The middle band mixes both, "
               "which is the first hint that BMI interacts with smoking.")
    return _save(fig, out_dir, "03_age_vs_charges_by_smoker", dpi)


def bmi_vs_charges(df, out_dir, dpi, threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    _scatter_by_smoker(ax, df, "bmi")
    ax.axvline(threshold, color=INK_2, linestyle="--", linewidth=1)
    ax.text(threshold + 0.4, ax.get_ylim()[1] * 0.97, f"BMI = {threshold:g}", color=INK_2,
            fontsize=9, va="top")
    ax.set_xlabel("BMI (kg/m²)")
    ax.set_title("For smokers, crossing BMI 30 is a step up, not a slope")
    _note(fig, "Non-smokers: BMI barely moves cost. Smokers: two clusters split at BMI 30. "
               "This is the obese x smoker interaction the model needs.")
    return _save(fig, out_dir, "04_bmi_vs_charges_by_smoker", dpi)


def charges_by_other_groups(df, out_dir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, col in zip(axes, ["sex", "region", "children"]):
        levels = list(df[col].cat.categories) if hasattr(df[col], "cat") else sorted(df[col].unique())
        data = [df.loc[df[col] == lvl, "charges"].to_numpy() for lvl in levels]
        ax.boxplot(data, widths=0.5, showfliers=True, patch_artist=True,
                   boxprops=dict(facecolor="#cde2fb", edgecolor=INK_2),
                   medianprops=dict(color=INK, linewidth=2), whiskerprops=dict(color=INK_2),
                   capprops=dict(color=INK_2),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor=NEUTRAL,
                                   markeredgecolor="none", alpha=0.5))
        ax.set_xticks(range(1, len(levels) + 1), [str(l) for l in levels])
        ax.set_title(f"by {col}")
        ax.set_xlabel(col)
    axes[0].set_ylabel("Charges (USD)")
    _usd(axes[0])
    fig.suptitle("Sex, region and number of children shift costs far less than smoking",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "05_charges_by_sex_region_children", dpi)


def correlation_heatmap(df, out_dir, dpi):
    m = pd.DataFrame({
        "age": df["age"], "bmi": df["bmi"], "children": df["children"],
        "smoker (yes=1)": (df["smoker"] == "yes").astype(int),
        "sex (male=1)": (df["sex"] == "male").astype(int),
        "obese (BMI≥30)": df["obese"], "charges": df["charges"],
    }).corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.grid(False)
    im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(m)), m.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(m)), m.columns)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    ax.set_title("Pearson correlations (smoking dominates charges)")
    return _save(fig, out_dir, "06_correlation_heatmap", dpi)


# Modelling -------------------------------------------------------------------

def model_ladder(cv_summary, out_dir, dpi):
    d = cv_summary.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    y = np.arange(len(d))
    for ax, col, title, fmt in [
        (axes[0], "cv_rmse", "5-fold CV RMSE (lower is better)", lambda v: f"${v:,.0f}"),
        (axes[1], "cv_r2_dollars", "5-fold CV R², dollar scale", lambda v: f"{v:.3f}"),
    ]:
        mean, sd = d[f"{col}_mean"], d[f"{col}_std"]
        ax.barh(y, mean, xerr=sd, height=0.55, color=BLUE, edgecolor=SURFACE,
                error_kw=dict(ecolor=INK_2, lw=1, capsize=3))
        for yi, v, s in zip(y, mean, sd):
            ax.text(v + s + (mean.max() * 0.02), yi, fmt(v), va="center", fontsize=9, color=INK)
        ax.set_title(title)
        ax.set_xlim(0, mean.max() * 1.25 if col == "cv_rmse" else 1.0)
        ax.grid(axis="y", visible=False)
    if (d["cv_r2_dollars_mean"] < 0).any():
        axes[1].set_xlim(min(0, d["cv_r2_dollars_mean"].min() - 0.1), 1.0)
    axes[0].set_yticks(y, d["model"])
    _note(fig, "Error bars: SD across folds. Log model predictions are back-transformed with Duan's smearing estimator.")
    return _save(fig, out_dir, "07_model_ladder_cv", dpi)


def diagnostics_panel(result, data, cooks_threshold, out_dir, dpi):
    infl = result.get_influence()
    fitted, resid = result.fittedvalues, result.resid
    stud = infl.resid_studentized_internal
    colors = data["smoker"].map(SMOKER_COLORS).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    ax.scatter(fitted, resid, s=10, c=colors, alpha=0.6, linewidths=0)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set(title="Residuals vs fitted", xlabel="Fitted charges (USD)", ylabel="Residual (USD)")
    _usd(ax), _usd(ax, "x")

    ax = axes[0, 1]
    (osm, osr), (slope, inter, _) = stats.probplot(stud, dist="norm")
    ax.scatter(osm, osr, s=10, color=BLUE, alpha=0.6, linewidths=0)
    ax.plot(osm, slope * osm + inter, color=INK_2, lw=1)
    ax.set(title="Normal Q-Q of studentized residuals", xlabel="Theoretical quantile", ylabel="Sample quantile")

    ax = axes[1, 0]
    ax.scatter(fitted, np.sqrt(np.abs(stud)), s=10, c=colors, alpha=0.6, linewidths=0)
    ax.set(title="Scale-location", xlabel="Fitted charges (USD)", ylabel="√|studentized residual|")
    _usd(ax, "x")

    ax = axes[1, 1]
    cooks = infl.cooks_distance[0]
    ax.vlines(np.arange(len(cooks)), 0, cooks, color=BLUE, lw=0.8)
    ax.axhline(cooks_threshold, color=ORANGE, lw=1, ls="--")
    n_flag = int((cooks > cooks_threshold).sum())
    ax.text(0.02, 0.96, f"Dashed line: 4/n = {cooks_threshold:.4f}\n{n_flag} points above it",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK_2)
    ax.set(title="Cook's distance", xlabel="Observation index", ylabel="Cook's D")

    handles = [plt.Line2D([], [], marker="o", ls="", color=SMOKER_COLORS[k], label=SMOKER_LABELS[k])
               for k in ["no", "yes"]]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Final model (M3) diagnostics", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "08_diagnostics_final_model", dpi)


def residuals_by_smoker(result, data, threshold, out_dir, dpi):
    resid = result.resid.to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bins = np.linspace(resid.min(), resid.max(), 60)
    for lvl in ["no", "yes"]:
        r = resid[(data["smoker"] == lvl).to_numpy()]
        ax.hist(r, bins=bins, alpha=0.75, color=SMOKER_COLORS[lvl], edgecolor=SURFACE,
                linewidth=0.5, label=f"{SMOKER_LABELS[lvl]} (n = {len(r)})")
    ax.axvline(threshold, color=INK_2, ls="--", lw=1)
    n_hi = int(((resid > threshold) & (data["smoker"] == "no").to_numpy()).sum())
    ax.text(threshold * 1.05, ax.get_ylim()[1] * 0.9,
            f"{n_hi} non-smokers under-predicted\nby more than ${threshold:,.0f}",
            fontsize=9, color=INK, va="top")
    ax.set(xlabel="Residual (actual minus predicted, USD)", ylabel="Number of people")
    _usd(ax, "x")
    ax.legend(loc="upper left")
    ax.set_title("The long right tail is people the data cannot explain")
    _note(fig, "These are the 'middle band' non-smokers from the age plot. No variable in the dataset separates them, "
               "so the residuals stay non-normal no matter which terms are added.")
    return _save(fig, out_dir, "09_residuals_by_smoker", dpi)


def coefficient_ci_grid(coef_table, robust_label, out_dir, dpi):
    terms = coef_table[coef_table["term"] != "Intercept"].reset_index(drop=True)
    n = len(terms)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    rl = robust_label.lower()
    methods = [
        ("ols", "Classical OLS", BLUE),
        (rl, f"Robust ({robust_label})", ORANGE),
        ("boot", "Pairs bootstrap", VIOLET),
    ]
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.0 * nrows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        row = terms.iloc[k]
        for j, (key, _, color) in enumerate(methods):
            lo, hi = row[f"ci_low_{key}"], row[f"ci_high_{key}"]
            ax.plot([lo, hi], [j, j], color=color, lw=2.2, solid_capstyle="round")
            ax.plot(row["coef"], j, "o", color=color, markersize=6, markeredgecolor=SURFACE)
        ax.axvline(0, color=INK_2, lw=0.8, ls=":")
        ax.set_yticks([])
        ax.set_ylim(-0.7, 2.7)
        ax.set_title(f"{row['term']}   ({row['coef']:,.1f})", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 10 else f"{v:g}"))
    handles = [plt.Line2D([], [], color=c, lw=2.2, marker="o", label=lab) for _, lab, c in methods]
    fig.suptitle("95% confidence intervals: three ways (USD per unit; each panel has its own scale)",
                 x=0.01, y=1.06, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 1.035))
    fig.tight_layout()
    return _save(fig, out_dir, "10_coefficient_ci_comparison", dpi)


def actual_vs_predicted(models: dict, keys: list[str], data, out_dir, dpi):
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.8), sharex=True, sharey=True)
    y = data["charges"].to_numpy()
    colors = data["smoker"].map(SMOKER_COLORS).to_numpy()
    lim = (0, y.max() * 1.05)
    for ax, k in zip(np.atleast_1d(axes), keys):
        m = models[k]
        p = m.predict_dollars(data)
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        ax.scatter(y, p, s=9, c=colors, alpha=0.55, linewidths=0)
        ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
        ax.set(xlim=lim, ylim=(min(0, p.min()), max(lim[1], p.max() * 1.02)),
               xlabel="Actual charges (USD)")
        ax.set_title(f"{m.label.split('  ')[0]}   R² (USD) = {r2:.3f}", fontsize=11)
        _usd(ax), _usd(ax, "x")
    np.atleast_1d(axes)[0].set_ylabel("Predicted charges (USD)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SMOKER_COLORS[k], label=SMOKER_LABELS[k])
               for k in ["no", "yes"]]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.04))
    fig.suptitle("Actual vs predicted (in-sample, dollar scale)", x=0.01, y=1.04, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "11_actual_vs_predicted", dpi)


def shap_beeswarm(sv, out_dir, dpi):
    plt.figure()
    shap.plots.beeswarm(sv, max_display=len(sv.feature_names), show=False, plot_size=(9, 5.5))
    fig = plt.gcf()
    fig.patch.set_facecolor(SURFACE)
    ax = plt.gca()
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("SHAP value: dollars added to this person's predicted charges", color=INK_2)
    ax.set_title("SHAP: what drives each person's prediction (final model)", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    return _save(fig, out_dir, "12_shap_beeswarm", dpi)
