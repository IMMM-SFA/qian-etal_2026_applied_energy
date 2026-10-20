#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper-package script. Output scope is documented in PAPER_OUTPUTS.md at the package root."""

import os
from pathlib import Path

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
PACKAGE_ROOT = DATA_ROOT / "datasets_and_scripts"
REPRODUCED_ROOT = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve()
SECTION_DIR = PACKAGE_ROOT / "3.5_ml_coverage"
import re
import argparse
import numpy as np
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Batch figure generation; no desktop GUI required.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import matplotlib.colors as mcolors


# ---------------- CLI -----------------

def parse_args():
    p = argparse.ArgumentParser(description="Layer1: Top-k coverage vs α (pooled + per-model) + top-k feature list")
    p.add_argument("--metrics-root", default=None,
                   help="Directory containing alpha_* subfolders. If omitted, defaults to the packaged section data directory.")
    p.add_argument("--outdir", default=None,
                   help="Figure output directory (default: <package>/reproduced/3.5/topk).")
    p.add_argument("--coverage", type=float, default=0.80,
                   help="Coverage target (default: 0.80).")
    p.add_argument("--boot-n", type=int, default=1000,
                   help="Bootstrap resamples per group for CI (default: 1000).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--style", default="", help="Optional matplotlib style name.")
    p.add_argument("--hot-models", default="log_en,log_l1,rf,gb",
                   help="Comma list of models to plot for HOT (default: log_en,rf).")
    p.add_argument("--cold-models", default="log_en,log_l1,rf,gb",
                   help="Comma list of models to plot for COLD (default: log_en,rf).")
    p.add_argument("--xtick-rotation", type=float, default=90.0, help="Rotation for x tick labels.")
    p.add_argument("--font-size", type=int, default=14, help="Base font size.")
    p.add_argument("--labelpad", type=float, default=16.0, help="Axis label padding.")
    p.add_argument("--titlepad", type=float, default=22.0, help="Title padding.")
    p.add_argument("--no-topk-features", action="store_true",
                   help="If set, do not write the top-k feature list CSV.")


    return p.parse_args()


# ---------------- IO -----------------

ALPHA_RE = re.compile(r"alpha_(?P<val>[0-9]+(?:_[0-9])?)$")


def find_alpha_dirs(root: str):
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        m = ALPHA_RE.match(name)
        if not m:
            continue
        label = m.group("val")
        try:
            numeric = float(label.replace("_", "."))
        except Exception:
            continue
        out.append((numeric, label, p))
    out.sort(key=lambda x: x[0])
    return out


def read_shap_summaries(alpha_dirs):
    frames = []
    for a_num, a_label, path in alpha_dirs:
        csv_path = os.path.join(path, "layer1_nestedcv_shap_summary.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        needed = {"subset", "alpha", "model", "outer_fold", "feature", "mean_abs_shap"}
        if not needed.issubset(df.columns):
            continue
        df = df.copy()
        df["alpha_numeric"] = float(a_num)
        df["model"] = df["model"].astype(str).str.lower()
        # normalize common model aliases
        df.loc[df["model"].isin(["logreg_en","elasticnet","logistic_en","logistic (elastic net)"]), "model"] = "log_en"
        df.loc[df["model"].isin(["svm"]), "model"] = "svc"
        df.loc[df["model"].isin(["logreg_l1","logistic_l1","logistic (l1)","lasso","l1"]), "model"] = "log_l1"
        df.loc[df["model"].isin(["randomforest","random_forest","random forest","rf_classifier","rfmodel"]), "model"] = "rf"
        df.loc[df["model"].isin(["gradientboosting","gradient_boosting","gradient boosting","gradientboostingclassifier","gbc"]), "model"] = "gb"
        df["subset"] = df["subset"].astype(str).str.lower()
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No layer1_nestedcv_shap_summary.csv found under alpha_* folders.")
    return pd.concat(frames, ignore_index=True)


# --------------- optional: drop anomalous folds (rare SHAP blow-ups) ---------------

def _summarize_folds(shap_df):
    def _stats(g):
        vals = np.asarray(g["mean_abs_shap"], float)
        tot = float(vals.sum())
        nz = int((vals > 0).sum())
        v = np.sort(vals)[::-1]
        top3 = float(v[:3].sum()) if v.size else 0.0
        return pd.Series({"total": tot, "nonzero": nz, "top3_share": (top3 / tot) if tot > 0 else 0.0})

    return (
        shap_df.groupby(["subset", "alpha_numeric", "model", "outer_fold"]).apply(_stats).reset_index()
    )


def filter_bad_folds(shap_df: pd.DataFrame, outdir=None, share_thresh=0.98, min_nonzero=5, iqr_mult=25.0):
    stats = _summarize_folds(shap_df)
    med = stats["total"].median()
    iqr = stats["total"].quantile(0.75) - stats["total"].quantile(0.25)
    big = stats["total"] > (med + iqr_mult * iqr)
    few = stats["nonzero"] < min_nonzero
    spk = stats["top3_share"] >= share_thresh
    stats["is_outlier"] = big | few | spk
    key = ["subset", "alpha_numeric", "model", "outer_fold"]
    merged = shap_df.merge(stats[key + ["is_outlier"]], on=key, how="left")
    filtered = merged[~merged["is_outlier"].fillna(False)].drop(columns=["is_outlier"])
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        stats.to_csv(os.path.join(outdir, "shap_fold_stats.csv"), index=False)
        stats[stats["is_outlier"]].to_csv(os.path.join(outdir, "shap_outlier_folds.csv"), index=False)
    return filtered


# ---------------- Top-k coverage + feature list -----------------

def topk_for_coverage(vals: np.ndarray, target: float) -> int:
    s = float(np.nansum(vals))
    if s <= 0:
        return 0
    shares = np.sort(vals / s)[::-1]
    c = 0.0
    for k, v in enumerate(shares, start=1):
        c += float(v)
        if c >= target:
            return k
    return int(len(shares))


def topk_features_for_coverage(vals: np.ndarray, features: np.ndarray, target: float):
    vals = np.asarray(vals, dtype=float)
    s = float(np.nansum(vals))
    if s <= 0 or len(vals) == 0:
        return 0, np.array([], dtype=object), np.array([], dtype=float), np.array([], dtype=float)

    idx = np.argsort(vals)[::-1]
    vals_s = vals[idx]
    feats_s = np.asarray(features, dtype=object)[idx]
    shares = vals_s / s
    cum = np.cumsum(shares)
    k = int(np.searchsorted(cum, target, side="left") + 1)
    k = min(k, len(vals_s))
    return k, feats_s[:k], shares[:k], cum[:k]


def compute_topk_feature_frequency(shap_df: pd.DataFrame, coverage_target: float) -> pd.DataFrame:
    rows = []
    group_cols = ["subset", "alpha_numeric", "model", "outer_fold"]
    for (subset, alpha, model, fold), g in shap_df.groupby(group_cols, sort=False):
        feats = g["feature"].astype(str).to_numpy()
        vals = g["mean_abs_shap"].to_numpy(dtype=float)
        k, top_feats, top_shares, top_cum = topk_features_for_coverage(vals, feats, coverage_target)
        for r, (f, sh, cs) in enumerate(zip(top_feats, top_shares, top_cum), start=1):
            rows.append(
                {
                    "subset": subset,
                    "alpha_numeric": float(alpha),
                    "model": model,
                    "outer_fold": fold,
                    "coverage_target": float(coverage_target),
                    "k_coverage": int(k),
                    "rank_in_fold": int(r),
                    "feature": str(f),
                    "share_in_fold": float(sh),
                    "cum_share_in_fold": float(cs),
                }
            )
    return pd.DataFrame(rows)


def compute_topk_feature_summary(shap_df: pd.DataFrame, coverage_target: float) -> pd.DataFrame:
    agg = (
        shap_df.groupby(["subset", "alpha_numeric", "model", "feature"], as_index=False)["mean_abs_shap"]
        .mean()
        .rename(columns={"mean_abs_shap": "mean_abs_shap_mean"})
    )

    fold_topk = compute_topk_feature_frequency(shap_df, coverage_target)

    n_folds = (
        shap_df.groupby(["subset", "alpha_numeric", "model"])["outer_fold"]
        .nunique()
        .reset_index(name="n_outer_folds")
    )
    fold_counts = (
        fold_topk.groupby(["subset", "alpha_numeric", "model", "feature"])["outer_fold"]
        .nunique()
        .reset_index(name="topk_fold_count")
        .merge(n_folds, on=["subset", "alpha_numeric", "model"], how="left")
    )
    fold_counts["topk_fold_frac"] = fold_counts["topk_fold_count"] / fold_counts["n_outer_folds"]

    mean_rank = (
        fold_topk.groupby(["subset", "alpha_numeric", "model", "feature"])["rank_in_fold"]
        .mean()
        .reset_index(name="mean_rank_in_fold")
    )

    out_parts = []
    for (subset, alpha, model), g in agg.groupby(["subset", "alpha_numeric", "model"], sort=True):
        g = g.sort_values("mean_abs_shap_mean", ascending=False).copy()
        total = float(g["mean_abs_shap_mean"].sum())
        if total <= 0 or g.empty:
            continue
        g["share"] = g["mean_abs_shap_mean"] / total
        g["cum_share"] = g["share"].cumsum()
        g["rank"] = np.arange(1, len(g) + 1, dtype=int)
        k = int(np.searchsorted(g["cum_share"].to_numpy(dtype=float), coverage_target, side="left") + 1)
        k = min(k, len(g))
        g["k_coverage"] = int(k)
        g["coverage_target"] = float(coverage_target)
        out_parts.append(g.iloc[:k].copy())

    if not out_parts:
        return pd.DataFrame(
            columns=[
                "subset",
                "alpha_numeric",
                "model",
                "coverage_target",
                "k_coverage",
                "rank",
                "feature",
                "mean_abs_shap_mean",
                "share",
                "cum_share",
                "topk_fold_count",
                "n_outer_folds",
                "topk_fold_frac",
                "mean_rank_in_fold",
            ]
        )

    out = pd.concat(out_parts, ignore_index=True)
    out = (
        out.merge(fold_counts, on=["subset", "alpha_numeric", "model", "feature"], how="left")
        .merge(mean_rank, on=["subset", "alpha_numeric", "model", "feature"], how="left")
    )
    out["topk_fold_count"] = out["topk_fold_count"].fillna(0).astype(int)
    out["n_outer_folds"] = out["n_outer_folds"].fillna(0).astype(int)
    out["topk_fold_frac"] = out["topk_fold_frac"].fillna(0.0).astype(float)
    out["mean_rank_in_fold"] = out["mean_rank_in_fold"].astype(float)
    return out.sort_values(["subset", "alpha_numeric", "model", "rank"]).reset_index(drop=True)


def compute_unit_k_coverage(shap_df: pd.DataFrame, coverage_target: float) -> pd.DataFrame:
    rows = []
    for (subset, alpha, model, fold), g in shap_df.groupby(["subset", "alpha_numeric", "model", "outer_fold"], sort=False):
        vals = g["mean_abs_shap"].to_numpy(dtype=float)
        k = topk_for_coverage(vals, coverage_target)
        rows.append(
            {
                "subset": subset,
                "alpha_numeric": float(alpha),
                "model": model,
                "outer_fold": fold,
                "coverage_target": float(coverage_target),
                "k_coverage": int(k),
                "n_features": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(df: pd.DataFrame, group_cols, value_col: str, boot_n: int, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for key, g in df.groupby(group_cols, sort=True):
        arr = g[value_col].to_numpy(dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n == 0:
            continue
        point = float(np.mean(arr))
        boots = np.empty(boot_n, dtype=float)
        for b in range(boot_n):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(np.mean(arr[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])

        rec = {value_col + "_mean": point, value_col + "_ci95_low": float(lo), value_col + "_ci95_high": float(hi)}
        if isinstance(key, tuple):
            for c, v in zip(group_cols, key):
                rec[c] = v
        else:
            rec[group_cols[0]] = key
        rec["n_units"] = int(n)
        out.append(rec)
    return pd.DataFrame(out)


# ---------------- Plotting -----------------

SUBSET_COLORS = {"hot": "#B22222", "cold": "#1E90FF"}
SUBSET_LIGHTS = {"hot": "#E9967A", "cold": "#87CEFA"}


def apply_plot_style(font_size: int, labelpad: float, titlepad: float):
    mpl.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": int(font_size * 1.45),
            "axes.labelsize": int(font_size * 1.45),
            "xtick.labelsize": int(font_size * 1.05),
            "ytick.labelsize": int(font_size * 1.05),
            "legend.fontsize": int(font_size * 1.20),
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.grid": False,
            "axes.labelpad": labelpad,
            "axes.titlepad": titlepad,
        }
    )


def _beautify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _pretty_model_name(s: str) -> str:
    s = (s or "").lower().strip()
    if s in ("log_en", "logreg_en", "elasticnet", "logistic_en", "logistic (elastic net)"):
        return "Logistic (Elastic Net)"
    if s in ("log_l1", "logreg_l1", "lasso", "logistic_l1", "logistic (l1)"):
        return "Logistic (L1)"
    if s in ("svc", "svm"):
        return "SVM"
    if s == "rf":
        return "Random Forest"
    if s == "gb":
        return "Gradient Boosting"
    return s




def _model_family(model: str) -> str:
    m = (model or "").lower().strip()
    if m in ("log_en", "log_l1", "logreg_en", "logreg_l1"):
        return "logistic"
    if m in ("rf", "gb"):
        return "tree"
    return "other"


# =============================================================================
# def _is_selected_model(model: str) -> bool:
#     # Selected representatives in each family
#     return (model or "").lower().strip() in ("log_en", "logreg_en", "rf")
# =============================================================================


def _is_primary_model(subset: str, model: str) -> bool:
    """Primary models are regime-specific:
    - hot:  EN and GB
    - cold: L1 and RF
    """
    subset = (subset or "").lower().strip()
    model = (model or "").lower().strip()

    primary_by_subset = {
        "hot": {"log_en", "gb"},
        "cold": {"log_l1", "rf"},
    }
    return model in primary_by_subset.get(subset, set())


# =============================================================================
# def _style_for_model(subset: str, model: str):
#     """Return (color, marker, linestyle) for subset/model.
# 
#     - Hot uses red shades; Cold uses blue shades.
#     - Logistic family: circle marker; Tree family: triangle marker.
#     - Selected (log_en, rf): darker shade; Sensitivity (log_l1, gb): lighter shade.
#     """
#     subset = (subset or "").lower().strip()
#     model = (model or "").lower().strip()
# 
#     dark = SUBSET_COLORS.get(subset, "gray")
#     light = SUBSET_LIGHTS.get(subset, "silver")
# 
#     color = dark if _is_selected_model(model) else light
#     linestyle = "-" if _is_selected_model(model) else "--"
#     fam = _model_family(model)
#     marker = "o" if fam == "logistic" else ("^" if fam == "tree" else "s")
#     return color, marker, linestyle
# =============================================================================


def _style_for_model(subset: str, model: str):
    """Return (color, marker, linestyle) for subset/model.

    - Hot uses red shades; Cold uses blue shades.
    - Logistic family: circle marker; Tree family: triangle marker.
    - Primary models (subset-specific): darker shade + solid line.
    - Support models: lighter shade + dashed line.
    """
    subset = (subset or "").lower().strip()
    model = (model or "").lower().strip()

    dark = SUBSET_COLORS.get(subset, "gray")
    light = SUBSET_LIGHTS.get(subset, "silver")

    is_primary = _is_primary_model(subset, model)

    color = dark if is_primary else light
    linestyle = "-" if is_primary else "--"

    fam = _model_family(model)
    marker = "o" if fam == "logistic" else ("^" if fam == "tree" else "s")
    return color, marker, linestyle


def plot_kcoverage_by_model(boot_df_m: pd.DataFrame, outdir: str, model_lists, rot=90, labelpad=16, titlepad=22, dpi=200):
    """Per-subset figure(s), showing requested models with family markers and selected-vs-sensitivity styling.

    - No CI shading (fill_between) to reduce visual clutter.
    - HOT legend split into two columns: Logistic on the left, Tree-based on the right.
    """
    by_model_dir = os.path.join(outdir, "by_model")
    os.makedirs(by_model_dir, exist_ok=True)

    for subset, models in model_lists.items():
        sub = boot_df_m[(boot_df_m["subset"] == subset) & (boot_df_m["model"].isin(models))]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        line_by_model = {}

        for model in models:
            seg = sub[sub["model"] == model].sort_values("alpha_numeric")
            if seg.empty:
                continue
            x = seg["alpha_numeric"].to_numpy(dtype=float)
            y = seg["k_coverage_mean"].to_numpy(dtype=float)

            color, marker, ls = _style_for_model(subset, model)
            ln = ax.plot(
                x, y,
                marker=marker,
                linestyle=ls,
                linewidth=2.2,
                markersize=9,
                color=color,
                label=_pretty_model_name(model),
            )[0]
            line_by_model[model] = ln

        xticks = sorted(sub["alpha_numeric"].unique())
        ax.set_xlabel("Stress Level α", labelpad=labelpad)
        ax.set_ylabel("Top-k", labelpad=labelpad)
        ax.set_title(f"{subset.capitalize()}", pad=titlepad)

        # Use the same y-scale for both hot and cold panels
        _, ymax = ax.get_ylim()
        ymax_target = max(37, ymax)
        ax.set_ylim(0, 40)

        # Force ticks to start at 0 and keep clean 5-unit spacing
        ax.set_yticks(np.arange(0, 41, 5))


        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in xticks])
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(rot)
            lbl.set_ha("center")
            lbl.set_va("top")

        # Legend formatting
        if subset == "hot":
            # Split legend into 2 columns: Logistic (left) vs Tree-based (right)
            logi = [m for m in models if (m in line_by_model) and (_model_family(m) == "logistic")]
            tree = [m for m in models if (m in line_by_model) and (_model_family(m) == "tree")]

# =============================================================================
#             logi_sel = [m for m in logi if _is_selected_model(m)]
#             logi_sen = [m for m in logi if not _is_selected_model(m)]
#             tree_sel = [m for m in tree if _is_selected_model(m)]
#             tree_sen = [m for m in tree if not _is_selected_model(m)]
# =============================================================================

            logi_sel = [m for m in logi if _is_primary_model(subset, m)]
            logi_sen = [m for m in logi if not _is_primary_model(subset, m)]
            tree_sel = [m for m in tree if _is_primary_model(subset, m)]
            tree_sen = [m for m in tree if not _is_primary_model(subset, m)]

            # Arrange row-wise: (logistic selected, tree selected) then (logistic sensitivity, tree sensitivity)
            order = []
            if logi_sel:
                order.append(logi_sel[0])
            if tree_sel:
                order.append(tree_sel[0])
            if logi_sen:
                order.append(logi_sen[0])
            if tree_sen:
                order.append(tree_sen[0])

            # Append any remaining (rare)
            for m0 in (logi_sel[1:] + tree_sel[1:] + logi_sen[1:] + tree_sen[1:]):
                if m0 not in order:
                    order.append(m0)

            handles = [line_by_model[m0] for m0 in order]
            labels = [_pretty_model_name(m0) for m0 in order]
            ax.legend(handles, labels, frameon=False, loc="upper left", ncol=2,
                      columnspacing=1.6, handlelength=2.6, handletextpad=0.8)
        else:
            ax.legend(frameon=False, loc="upper left")

        _beautify_axes(ax)
        fig.tight_layout()

        base = os.path.join(by_model_dir, f"k_coverage_by_alpha_{subset}_by_model")
        for ext in ("png",):
            fig.savefig(f"{base}.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ---------------- Category-share bars (CDD/HDD vs Wind vs Solar)-----------------

# Colors (match Layer-2 catshares style, but with requested categories)
TEMP_BASE = {"hot": "#8B0000", "cold": "#00008B"}   # dark red / dark blue
WIND_BASE = "#006400"   # dark green
SOLAR_BASE = "#B8860B"  # dark goldenrod


def _lighten(hex_color: str, factor: float = 0.5) -> str:
    """Blend a color with white. factor in (0,1); higher -> lighter."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex(rgb + (1 - rgb) * factor)


def _is_temp_feature(feat: str, subset: str) -> bool:
    f = str(feat).strip().lower()
    tokens = re.split(r"[_\W]+", f)
    if subset == "hot":
        return f.startswith("cdd") or ("cdd" in tokens)
    if subset == "cold":
        return f.startswith("hdd") or ("hdd" in tokens)
    return False



def _is_wind_feature(feat: str) -> bool:
    f = str(feat).strip().lower()
    # renewables wind: wind, windfrac, etc. Exclude wind speed / components
    if f.startswith(("wspd", "u10", "v10")):
        return False
    tokens = re.split(r"[_\W]+", f)
    return f.startswith("wind") or ("wind" in tokens) or ("windfrac" in tokens) or ("wind_fraction" in tokens)



def _is_solar_feature(feat: str) -> bool:
    f = str(feat).strip().lower()
    tokens = re.split(r"[_\W]+", f)
    return f.startswith("solar") or ("solar" in tokens) or ("solarfrac" in tokens) or ("solar_fraction" in tokens)



def _select_topk_rows_by_coverage(g: pd.DataFrame, coverage_target: float) -> pd.DataFrame:
    """Return rows restricted to the minimal Top-k features reaching coverage_target."""
    vals = g["mean_abs_shap"].to_numpy(dtype=float)
    s = float(np.nansum(vals))
    if s <= 0 or len(vals) == 0:
        return g.iloc[0:0]
    idx = np.argsort(vals)[::-1]
    vals_s = vals[idx]
    shares = vals_s / s
    cum = np.cumsum(shares)
    k = int(np.searchsorted(cum, coverage_target, side="left") + 1)
    k = min(k, len(vals_s))
    keep_idx = idx[:k]
    return g.iloc[keep_idx]


def compute_temp_wind_solar_shares(
    shap_df: pd.DataFrame,
    coverage_target: float,
    topk_only: bool = False,
) -> pd.DataFrame:
    """Compute unit-level shares of |SHAP| mass for:
    - temp: CDD (HOT) / HDD (COLD)
    - wind: Wind / WindFrac / ... (feature name starts with 'wind', excluding WSPD/U10/V10)
    - solar: Solar / SolarFrac / ... (feature name starts with 'solar')

    Shares are normalized within {temp, wind, solar} so stacked bars sum to 1.
    """
    rows = []
    group_cols = ["subset", "alpha_numeric", "model", "outer_fold"]
    for (subset, alpha, model, fold), g in shap_df.groupby(group_cols, sort=False):
        gg = g
        if topk_only:
            gg = _select_topk_rows_by_coverage(g, coverage_target=coverage_target)
            if gg.empty:
                continue

        temp_sum = float(gg.loc[gg["feature"].apply(lambda x: _is_temp_feature(x, subset)), "mean_abs_shap"].sum())
        wind_sum = float(gg.loc[gg["feature"].apply(_is_wind_feature), "mean_abs_shap"].sum())
        solar_sum = float(gg.loc[gg["feature"].apply(_is_solar_feature), "mean_abs_shap"].sum())

        total = temp_sum + wind_sum + solar_sum
        if total <= 0:
            continue

        rows.append({
            "subset": subset,
            "alpha_numeric": float(alpha),
            "model": model,
            "outer_fold": fold,
            "share_temp": temp_sum / total,
            "share_wind": wind_sum / total,
            "share_solar": solar_sum / total,
        })

    return pd.DataFrame(rows)


def bootstrap_mean_shares_cat(df: pd.DataFrame, model: str, boot_n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (subset, alpha), g in df[df["model"] == model].groupby(["subset", "alpha_numeric"], sort=True):
        # one row per fold
        agg = g.groupby("outer_fold")[["share_temp", "share_wind", "share_solar"]].mean()
        arr = agg.to_numpy(dtype=float)
        m = arr.shape[0]
        if m == 0:
            continue
        point = np.nanmean(arr, axis=0)
        boots = np.zeros((boot_n, 3), dtype=float)
        for b in range(boot_n):
            idx = rng.integers(0, m, size=m)
            boots[b, :] = np.nanmean(arr[idx, :], axis=0)
        lo = np.nanpercentile(boots, 2.5, axis=0)
        hi = np.nanpercentile(boots, 97.5, axis=0)
        rows.append({
            "subset": subset,
            "alpha_numeric": float(alpha),
            "model": model,
            "n_units": int(m),
            "share_temp_mean": float(point[0]),
            "share_temp_lo": float(lo[0]),
            "share_temp_hi": float(hi[0]),
            "share_wind_mean": float(point[1]),
            "share_wind_lo": float(lo[1]),
            "share_wind_hi": float(hi[1]),
            "share_solar_mean": float(point[2]),
            "share_solar_lo": float(lo[2]),
            "share_solar_hi": float(hi[2]),
        })
    return pd.DataFrame(rows).sort_values(["subset", "alpha_numeric"]) if rows else pd.DataFrame()




# ---------------- main -----------------

def main():
    args = parse_args()

    apply_plot_style(args.font_size, args.labelpad, args.titlepad)
    if args.style:
        try:
            plt.style.use(args.style)
        except Exception:
            pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_root = os.path.abspath(args.metrics_root) if args.metrics_root else str(SECTION_DIR / "data")
    outdir_figs = os.path.abspath(args.outdir) if args.outdir else str(REPRODUCED_ROOT / "3.5" / "topk")
    outdir_tables = os.path.join(outdir_figs, "tables")
    os.makedirs(outdir_figs, exist_ok=True)
    os.makedirs(outdir_tables, exist_ok=True)

    # model lists
    hot_models = [m.strip().lower() for m in args.hot_models.split(",") if m.strip()]
    cold_models = [m.strip().lower() for m in args.cold_models.split(",") if m.strip()]
    hot_models = ["svc" if m == "svm" else m for m in hot_models]
    cold_models = ["svc" if m == "svm" else m for m in cold_models]
    model_lists = {"hot": hot_models, "cold": cold_models}

    alpha_dirs = find_alpha_dirs(metrics_root)
    if not alpha_dirs:
        raise SystemExit(f"No alpha_* folders found under: {metrics_root}")

    shap_df = read_shap_summaries(alpha_dirs)
    shap_df = filter_bad_folds(shap_df, outdir=outdir_tables)

    # ---- unit-level k coverage ----
    unit_df = compute_unit_k_coverage(shap_df, coverage_target=args.coverage)
    unit_path = os.path.join(outdir_tables, "topk_coverage_unit_metrics.csv")
    unit_df.to_csv(unit_path, index=False)

    # ---- bootstrap summaries ----
    boot_df = bootstrap_mean_ci(
        unit_df,
        group_cols=["subset", "alpha_numeric"],
        value_col="k_coverage",
        boot_n=args.boot_n,
        seed=args.seed,
    ).sort_values(["subset", "alpha_numeric"])
    boot_path = os.path.join(outdir_tables, "topk_coverage_bootstrap_summary.csv")
    boot_df.to_csv(boot_path, index=False)

    boot_df_m = bootstrap_mean_ci(
        unit_df,
        group_cols=["subset", "alpha_numeric", "model"],
        value_col="k_coverage",
        boot_n=args.boot_n,
        seed=args.seed,
    ).sort_values(["subset", "alpha_numeric", "model"])
    boot_path_m = os.path.join(outdir_tables, "topk_coverage_bootstrap_by_model.csv")
    boot_df_m.to_csv(boot_path_m, index=False)

    # ---- top-k feature list (optional) ----
    topk_feat_path = None
    if not args.no_topk_features:
        topk_feat_df = compute_topk_feature_summary(shap_df, coverage_target=args.coverage)
        topk_feat_path = os.path.join(outdir_tables, f"topk_features_{int(args.coverage*100)}pct_by_alpha_model.csv")
        topk_feat_df.to_csv(topk_feat_path, index=False)

    # ---- plots (TOP-K ONLY) ----
    rot, lp, tp = args.xtick_rotation, args.labelpad, args.titlepad
    plot_kcoverage_by_model(boot_df_m, outdir=outdir_figs, model_lists=model_lists, rot=rot, labelpad=lp, titlepad=tp)

    # ---- explicit paths for troubleshooting ----
    print("\n[INFO] metrics_root :", metrics_root)
    print("[INFO] figs outdir   :", outdir_figs)
    print("[INFO] tables outdir :", outdir_tables)
    print("[SAVE]", unit_path)
    print("[SAVE]", boot_path)
    print("[SAVE]", boot_path_m)
    if topk_feat_path:
        print("[SAVE]", topk_feat_path)
    print("[DONE] by-model top-k coverage and supporting tables")


if __name__ == "__main__":
    import sys

    main()
