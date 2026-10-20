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
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Batch figure generation; no desktop GUI required.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Layer1 metrics visualizer")
    p.add_argument("--metrics-root", required=False, default=None,
                   help="Directory that contains per-alpha subfolders (e.g., alpha_80, alpha_82_0).")
    p.add_argument("--outdir", default=None,
                   help="Output figures directory. Default: <package>/reproduced/3.5/metrics")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--style", default="",
                   help="Optional matplotlib style name (e.g., 'seaborn-v0_8-paper').")

    # SHAP stacked bars

    # Built-in importance stacked bars (RF/GB)
    
    p.add_argument("--save-pr-tables", default="on", choices=["on","off"],
               help="Write PR-AUC tables to CSV (default: on)")
    
    # Defaults
    p.set_defaults(stacked_normalize=True)
    p.set_defaults(imp_normalize=True)
    return p.parse_args()

# ---------------------- IO helpers ----------------------

ALPHA_RE = re.compile(r"alpha_(?P<val>[0-9]+(?:_[0-9])?)$")

def _safe_to_csv(df: pd.DataFrame, path: str, *, index: bool):
    """Write CSV atomically; if target is locked on Windows, fall back to timestamped file."""
    import time
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=index)
    try:
        os.replace(tmp, path)  # atomic replace if not locked
        print("[save]", path)
    except PermissionError:
        alt = os.path.splitext(path)[0] + f"_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        os.replace(tmp, alt)
        print(f"[warn] Locked: {path} — wrote '{alt}' instead.")

def find_alpha_dirs(root: str) -> List[Tuple[float, str, str]]:
    """Return list of (alpha_numeric, alpha_label, path). Accepts alpha_80, alpha_99_5, etc."""
    results: List[Tuple[float, str, str]] = []
    if not os.path.isdir(root):
        return results
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        m = ALPHA_RE.match(name)
        if not m:
            continue
        label = m.group("val")  # e.g., "80", "99_5"
        try:
            numeric = float(label.replace("_", "."))
        except Exception:
            continue
        results.append((numeric, label, p))
    results.sort(key=lambda x: x[0])
    return results

def _parse_alpha_list(arg: str):
    """Return None for 'max', else a list of float alphas. Accepts 99_5 too."""
    if arg is None or str(arg).strip().lower() == "max":
        return None
    out = []
    for tok in str(arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        tok = tok.replace("_", ".")
        out.append(float(tok))
    return out

def _alpha_label(a: float) -> str:
    """File/label-friendly α text (99.5 -> '99_5', 97.0 -> '97')."""
    s = f"{a:g}"
    return s.replace(".", "_")

def read_metrics_summaries(alpha_dirs):
    """Concatenate layer1_nestedcv_metrics_summary.csv across alphas; add alpha columns."""
    frames = []
    for a_num, a_label, path in alpha_dirs:
        csv_path = os.path.join(path, "layer1_nestedcv_metrics_summary.csv")
        if not os.path.exists(csv_path):
            alt = os.path.abspath(os.path.join(path, "..", "layer1_nestedcv_metrics_summary.csv"))
            if os.path.exists(alt):
                csv_path = alt
            else:
                print(f"[warn] Missing: {csv_path}")
                continue
        try:
            df = pd.read_csv(csv_path)
            df["alpha_numeric"] = a_num
            df["alpha_label"] = a_label
            frames.append(df)
        except Exception as e:
            print(f"[warn] Failed to read {csv_path}: {e}")
    if not frames:
        raise FileNotFoundError("No metrics_summary CSVs found under metrics-root.")
    all_df = pd.concat(frames, ignore_index=True)
    # normalize column names lightly
    all_df.columns = [c.replace("PR_AUC", "PR_AUC")
                        .replace("Precision", "precision")
                        .replace("Recall", "recall")
                        .replace("F1", "f1")
                        .replace("Accuracy", "accuracy")
                        for c in all_df.columns]
    return all_df

def save_pr_auc_tables(df: pd.DataFrame, outdir: str):
    """
    Save PR-AUC values per model across alphas, for each subset (hot/cold).

    Outputs (per subset):
      - PR_AUC_table_<subset>_wide.csv  : rows = alpha, columns = models, values = PR_AUC_mean
      - PR_AUC_table_<subset>_long.csv  : tidy table (alpha, model, PR_AUC_mean, ci95_low, ci95_high)
    """
    needed = {"subset", "alpha_numeric", "model", "PR_AUC_mean"}
    if not needed.issubset(df.columns):
        print("[info] PR-AUC columns not found; skipping PR-AUC tables.")
        return

    # keep only the models we chart (preserves consistent order)
    if "model" in df.columns:
        df = df[df["model"].isin(MODEL_ORDER)].copy()

    for subset_name, subdf in _facet_by_subset(df):
        if subdf.empty:
            continue

        # --- WIDE: alpha × model
        wide = (subdf
                .pivot_table(index="alpha_numeric",
                             columns="model",
                             values="PR_AUC_mean",
                             aggfunc="mean")
                .sort_index())
        # nice display names as headers
        wide.columns = [MODEL_LABELS.get(m, m) for m in wide.columns]
        wide.index.name = "alpha"

        wide_path = os.path.join(outdir, f"PR_AUC_table_{subset_name}_wide.csv")
        #wide.to_csv(wide_path)
        _safe_to_csv(wide, wide_path, index=True)

        # --- LONG / tidy: include CIs if present
        cols = ["subset", "alpha_numeric", "model", "PR_AUC_mean"]
        if {"PR_AUC_ci95_low", "PR_AUC_ci95_high"}.issubset(subdf.columns):
            cols += ["PR_AUC_ci95_low", "PR_AUC_ci95_high"]

        long_df = (subdf[cols]
                   .sort_values(["alpha_numeric", "model"])
                   .rename(columns={"alpha_numeric": "alpha"}))
        # pretty model names in a separate column for readability
        long_df["model_pretty"] = long_df["model"].map(lambda m: MODEL_LABELS.get(m, m))

        long_path = os.path.join(outdir, f"PR_AUC_table_{subset_name}_long.csv")
        #long_df.to_csv(long_path, index=False)
        _safe_to_csv(long_df, long_path, index=False)

        print("[save]", wide_path)
        print("[save]", long_path)

def read_shap_summaries(alpha_dirs):
    """Read per-alpha SHAP summaries; return concatenated DataFrame."""
    frames = []
    for a_num, a_label, path in alpha_dirs:
        csv_path = os.path.join(path, "layer1_nestedcv_shap_summary.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            needed = {"subset","alpha","model","outer_fold","feature","mean_abs_shap"}
            if not needed.issubset(set(df.columns)):
                continue
            df = df.copy()
            df["alpha_numeric"] = float(a_num)
            df["model"] = df["model"].astype(str).str.lower()
            df["subset"] = df["subset"].astype(str).str.lower()
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["subset","alpha","alpha_numeric","model","outer_fold","feature","mean_abs_shap"])
    return pd.concat(frames, ignore_index=True)

def read_builtin_importances(alpha_dirs):
    """Read per-alpha RF/GB built-in importances; return concatenated DataFrame."""
    frames = []
    for a_num, a_label, path in alpha_dirs:
        csv_path = os.path.join(path, "layer1_nestedcv_builtin_importances.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            needed = {"subset","alpha","model","outer_fold","feature","importance"}
            if not needed.issubset(set(df.columns)):
                continue
            df = df.copy()
            df["alpha_numeric"] = float(a_num)
            df["model"] = df["model"].astype(str).str.lower()
            df["subset"] = df["subset"].astype(str).str.lower()
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["subset","alpha","alpha_numeric","model","outer_fold","feature","importance","importance_norm","rank"])
    return pd.concat(frames, ignore_index=True)

# ---------------------- Plot helpers ----------------------

MODEL_ORDER = ["logreg_l1", "logreg_en", "rf", "gb", "svc", "knn", "mlp"]
MODEL_LABELS = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "logreg_en": "Logistic (Elastic Net)",
    "logreg_l1": "Logistic (L1)",
    "svc": "SVM",
    "knn": "k-NN",
    "mlp": "MLP",
#    "group_lasso": "Group Lasso",
}
MODEL_COLORS = {
    "rf":  "#CD5C5C",
    "gb":  "#9b59b6",
    "logreg_en": "#2ca02c",
    "logreg_l1": "#228B22",
    "svc": "#4169E1",
    "knn": "#7f7f7f",
    "mlp": "#e67e22",
#    "group_lasso": "#8e44ad",
}

METRICS = [
    ("PR_AUC",   "PR_AUC_mean",   "PR_AUC_ci95_low",   "PR_AUC_ci95_high",   "PR-AUC"),
    ("precision","precision_mean","precision_ci95_low","precision_ci95_high","Precision"),
    ("recall",   "recall_mean",   "recall_ci95_low",   "recall_ci95_high",   "Recall"),
    ("f1",       "f1_mean",       "f1_ci95_low",       "f1_ci95_high",       "F1-score"),
    ("accuracy", "accuracy_mean", "accuracy_ci95_low", "accuracy_ci95_high", "Accuracy"),
]

def ci_to_err(low: np.ndarray, high: np.ndarray, mean: np.ndarray) -> np.ndarray:
    err = np.maximum(np.asarray(mean, float) - np.asarray(low, float), np.asarray(high, float) - np.asarray(mean, float))
    return np.where(np.isfinite(err), err, 0.0)

def _facet_by_subset(df: pd.DataFrame):
    hot  = df[df["subset"].str.lower() == "hot"].copy() if "subset" in df.columns else df.copy()
    cold = df[df["subset"].str.lower() == "cold"].copy() if "subset" in df.columns else df.iloc[0:0].copy()
    return [("hot", hot), ("cold", cold)]

# ---------- Performance plots ----------

def plot_metric_ribbons(df: pd.DataFrame, metric_key: str, mean_col: str, lo_col: str, hi_col: str,
                        title_base: str, outdir: str, dpi: int = 200):
    """
    Ribbons over α with model lines.
    For PR_AUC plots only (i.e., PR_AUC_by_alpha_hot/cold):
      • HOT  : Primary = Logistic (Elastic Net), Support = RF
      • COLD : Primary = Logistic (Elastic Net),          Support = RF
      • De-emphasize other models.
      • Legend order: Primary, Support, then others grouped in 2 rows.
    Other metrics keep default styling.
    """

    # --- central font size knobs ---
    title_fs   = 22
    label_fs   = 20
    tick_fs    = 18
    legend_fs  = 16

    for subset_name, subdf in _facet_by_subset(df):
        if subdf.empty:
            continue

        # -------- Primary/Support by subset --------
        if subset_name.lower() == "hot":
            primary_key   = "logreg_en"   # Logistic (Elastic Net)
            support_key   = "gb"          # Gradient Boosting
            primary_label = "Selected: Logistic (Elastic Net)"
            support_label = "Selected: Gradient Boosting"
        else:  # cold
            primary_key   = "logreg_l1"   # Logistic (L1)
            support_key   = "rf"          # Random Forest
            primary_label = "Selected: Logistic (L1)"
            support_label = "Selected: Random Forest"

        # Plot order: Primary, Support, then the rest
        others = [m for m in MODEL_ORDER if m not in (primary_key, support_key)]
        loop_order = [m for m in [primary_key, support_key] if m in MODEL_ORDER] + others

        fig, ax = plt.subplots(figsize=(8.8, 5.2))

        for m in loop_order:
            ms = subdf[subdf["model"] == m].sort_values("alpha_numeric")
            if ms.empty:
                continue

            is_pr      = (metric_key == "PR_AUC")
            is_primary = (m == primary_key)
            is_support = (m == support_key)

            base_color = MODEL_COLORS.get(m, "gray")

            # Styling: emphasize primary/support only for PR_AUC; others lighter & hollow
            if is_pr and (is_primary or is_support):
                lw          = 3.0
                alpha_line  = 1.0
                alpha_fill  = 0.22
                marker      = "o"
                ms_mark     = 8.5
                z           = 5 if is_primary else 4   # Primary over Support
                mew         = 1.8
                mfc         = base_color             # FILLED markers for Primary/Support
                mec         = base_color
            elif is_pr:
                lw          = 1.6
                alpha_line  = 0.38
                alpha_fill  = 0.08
                marker      = "o"
                ms_mark     = 5.0
                z           = 2
                mew         = 1.4
                mfc         = "none"                 # HOLLOW markers for other models
                mec         = base_color
            else:
                lw          = 2.4
                alpha_line  = 0.95
                alpha_fill  = 0.15
                marker      = "o"
                ms_mark     = 5.0
                z           = 3
                mew         = 1.4
                mfc         = "none"
                mec         = base_color

            # Data vectors
            y  = ms[mean_col].to_numpy()
            lo = ms[lo_col].to_numpy() if lo_col in ms.columns else y
            hi = ms[hi_col].to_numpy() if hi_col in ms.columns else y

            # Legend label
            if is_pr:
                if is_primary:
                    label = primary_label
                elif is_support:
                    label = support_label
                else:
                    label = MODEL_LABELS.get(m, m)
            else:
                label = MODEL_LABELS.get(m, m)

            ax.plot(
                ms["alpha_numeric"], y,
                marker=marker, markersize=ms_mark,
                markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=mew,
                linewidth=lw, label=label, color=base_color,
                alpha=alpha_line, zorder=z
            )
            ax.fill_between(
                ms["alpha_numeric"], lo, hi,
                alpha=alpha_fill, color=base_color,
                linewidth=0, zorder=z-1
            )

        # --- Titles / labels with larger fonts ---
        ax.set_title(
            f"{subset_name.capitalize()}",
            fontsize=title_fs, fontweight="bold", pad=14
        )
        ax.set_xlabel(
            "Stress Level α",
            fontsize=label_fs, fontweight="bold", labelpad=10
        )
        ax.set_ylabel(
            title_base,
            fontsize=label_fs, fontweight="bold", labelpad=10
        )

        # --- X ticks: rotate 90 deg and enlarge font ---
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        xticks = sorted(subdf["alpha_numeric"].unique())
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
            rotation=90,
            ha="center",
            fontsize=tick_fs
        )
        # Y ticks fontsize
        ax.tick_params(axis="y", labelsize=tick_fs)

        # kill top/right frame
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_axisbelow(True)

        # ----------------------------
        # Legend (PR_AUC: 5 rows by spec)
        # ----------------------------
        if metric_key == "PR_AUC":
            model_pretty = {
                "rf":        "Random Forest",
                "gb":        "Gradient Boosting",
                "logreg_en": "Logistic (Elastic Net)",
                "logreg_l1": "Logistic (L1)",
                "svc":       "Support Vector Classifier",
                "knn":       "K-Nearest Neighbors",
                "mlp":       "Multilayer Perceptron",
            }

            def h(model_key, lw=2.2, ms=7, filled=False):
                color = MODEL_COLORS.get(model_key, "gray")
                return Line2D(
                    [0], [0],
                    color=color, lw=lw, marker="o", markersize=ms,
                    markerfacecolor=(color if filled else "none"),
                    markeredgecolor=color, markeredgewidth=1.8, alpha=1.0
                )

            # Selected models
            row1_handles = [h(primary_key, lw=3.0, ms=8, filled=True)]
            row1_labels  = [primary_label]

            row2_handles = [h(support_key, lw=3.0, ms=8, filled=True)]
            row2_labels  = [support_label]

            # Remaining models (exclude the selected ones)
            remaining_models = [m for m in MODEL_ORDER if m not in (primary_key, support_key)]

            # Put the non-selected logistic first if present
            remaining_models = sorted(
                remaining_models,
                key=lambda m: (
                    0 if m in ("logreg_l1", "logreg_en") else 1,
                    MODEL_ORDER.index(m)
                )
            )

            # Split remaining models into up to 3 legend rows
            row3_models = remaining_models[:1]
            row4_models = remaining_models[1:3]
            row5_models = remaining_models[3:5]

            row3_handles = [h(m, lw=2.2, ms=7, filled=False) for m in row3_models]
            row3_labels  = [model_pretty[m] for m in row3_models]

            row4_handles = [h(m, lw=2.2, ms=7, filled=False) for m in row4_models]
            row4_labels  = [model_pretty[m] for m in row4_models]

            row5_handles = [h(m, lw=2.2, ms=7, filled=False) for m in row5_models]
            row5_labels  = [model_pretty[m] for m in row5_models]

            common_kws = dict(
                frameon=True, fancybox=True, framealpha=0.85, edgecolor="none",
                fontsize=legend_fs, bbox_transform=ax.transAxes, loc="lower left",
                labelspacing=0.6, borderpad=0.6, handlelength=2.0, handletextpad=0.6,
            )

            x_anchor  = 0.02
            y_bottom  = 0.02
            row_gap   = 0.10

            if row5_handles:
                leg5 = ax.legend(row5_handles, row5_labels, ncol=len(row5_handles),
                                 bbox_to_anchor=(x_anchor, y_bottom), **common_kws)
                ax.add_artist(leg5)

            if row4_handles:
                leg4 = ax.legend(row4_handles, row4_labels, ncol=len(row4_handles),
                                 bbox_to_anchor=(x_anchor, y_bottom + 1*row_gap), **common_kws)
                ax.add_artist(leg4)

            if row3_handles:
                leg3 = ax.legend(row3_handles, row3_labels, ncol=len(row3_handles),
                                 bbox_to_anchor=(x_anchor, y_bottom + 2*row_gap), **common_kws)
                ax.add_artist(leg3)

            leg2 = ax.legend(row2_handles, row2_labels, ncol=1,
                             bbox_to_anchor=(x_anchor, y_bottom + 3*row_gap), **common_kws)
            for t in leg2.get_texts():
                t.set_fontweight("bold")
            ax.add_artist(leg2)

            leg1 = ax.legend(row1_handles, row1_labels, ncol=1,
                             bbox_to_anchor=(x_anchor, y_bottom + 4*row_gap), **common_kws)
            for t in leg1.get_texts():
                t.set_fontweight("bold")
            ax.add_artist(leg1)

# =============================================================================
# 
#         # ----------------------------
#         # Legend (PR_AUC special layout)
#         # ----------------------------
#         if metric_key == "PR_AUC":
#             model_pretty = {
#                 "rf":        "Random Forest",
#                 "gb":        "Gradient Boosting",
#                 "logreg_en": "Logistic (Elastic Net)",
#                 "logreg_l1": "Logistic (L1)",
#                 "svc":       "SVM",
#                 "knn":       "k-NN",
#                 "mlp":       "MLP",
#             }
# 
#             def h(model_key, lw=2.2, ms=6, filled=False):
#                 color = MODEL_COLORS.get(model_key, "gray")
#                 return Line2D(
#                     [0], [0],
#                     color=color, lw=lw, marker="o", markersize=ms,
#                     markerfacecolor=(color if filled else "none"),
#                     markeredgecolor=color, markeredgewidth=1.6, alpha=1.0
#                 )
# 
#             # Row 1: Primary/Support (filled)
#             prim_handle = h(primary_key, lw=3.0, ms=7, filled=True)
#             supp_handle = h(support_key, lw=3.0, ms=7, filled=True)
# 
#             # Row 2: the *other* Logistic, RF, GB — hollow
#             other_logistic = "logreg_en" if primary_key == "logreg_l1" else "logreg_l1"
#             row2_keys    = [other_logistic, "rf", "gb"]
#             row2_handles = [h(k) for k in row2_keys]
#             row2_labels  = [model_pretty[k] for k in row2_keys]
# 
#             # Row 3: k-NN, MLP — hollow
#             row3_keys    = ["knn", "mlp"]
#             row3_handles = [h(k) for k in row3_keys]
#             row3_labels  = [model_pretty[k] for k in row3_keys]
# 
#             # Anchor inside axes
#             x_anchor  = 0.018
#             y_bottom  = 0.05
#             row_gap   = 0.090  # a touch more vertical gap for bigger font
# 
#             common_kws = dict(
#                 frameon=True,
#                 fancybox=True,
#                 framealpha=0.85,
#                 edgecolor="none",
#                 fontsize=legend_fs,
#                 bbox_transform=ax.transAxes,
#                 labelspacing=0.6,
#                 borderpad=0.5,
#                 handlelength=2.0,
#                 handletextpad=0.6,
#                 loc="lower left"
#             )
# 
#             # Bottom-up stacking so row1 sits highest
#             leg3 = ax.legend(
#                 row3_handles, row3_labels, ncol=2,
#                 bbox_to_anchor=(x_anchor, y_bottom),
#                 **common_kws
#             )
#             ax.add_artist(leg3)
# 
#             leg2 = ax.legend(
#                 row2_handles, row2_labels, ncol=3,
#                 bbox_to_anchor=(x_anchor, y_bottom + row_gap),
#                 **common_kws
#             )
#             ax.add_artist(leg2)
# 
#             leg1 = ax.legend(
#                 [prim_handle, supp_handle],
#                 [primary_label, support_label],
#                 ncol=2,
#                 bbox_to_anchor=(x_anchor, y_bottom + 2*row_gap),
#                 **common_kws
#             )
#             ax.add_artist(leg1)
# 
#             # bold Primary/Support text
#             for t in leg1.get_texts():
#                 t.set_fontweight("bold")
# 
#         else:
#             # Non-PR_AUC case
#             handles, labels = ax.get_legend_handles_labels()
#             leg = ax.legend(
#                 handles, labels,
#                 ncol=3,
#                 frameon=True, fancybox=True,
#                 framealpha=0.85, edgecolor="none",
#                 loc="lower left",
#                 bbox_to_anchor=(0.018, 0.05),
#                 bbox_transform=ax.transAxes,
#                 fontsize=legend_fs,
#                 labelspacing=0.6,
#                 borderpad=0.5,
#                 handlelength=2.0,
#                 handletextpad=0.6
#             )
#             # (no bold tweak here)
# 
# 
# =============================================================================

        fig.tight_layout()

        base = f"{metric_key}_by_alpha_{subset_name}"
        for ext in ("png",):
            fig.savefig(os.path.join(outdir, f"{base}.{ext}"),
                        dpi=dpi, bbox_inches="tight")
        plt.close(fig)





# ---------- Stacked bars (SHAP) ----------

def _gradient(start_hex, end_hex, n):
    start = np.array(mcolors.to_rgb(start_hex))
    end   = np.array(mcolors.to_rgb(end_hex))
    if n <= 1:
        return [mcolors.to_hex(start)]
    return [mcolors.to_hex(start + (end - start) * i/(n-1)) for i in range(n)]

def _pivot_shap(df, top_feats, normalize_rows=True):
    g = (df.groupby(["alpha_numeric","feature"], as_index=False)["mean_abs_shap"].mean())
    pivot = g.pivot(index="alpha_numeric", columns="feature", values="mean_abs_shap").fillna(0.0)
    for f in top_feats:
        if f not in pivot.columns:
            pivot[f] = 0.0
    out = pivot[top_feats].copy()
    other_cols = [c for c in pivot.columns if c not in top_feats]
    out["Other"] = pivot[other_cols].sum(axis=1) if other_cols else 0.0
    if normalize_rows:
        s = out.sum(axis=1).replace(0, np.nan)
        out = out.div(s, axis=0).fillna(0.0)
    return out.sort_index()

def _pick_top_feats_all_models(shap_df_subset, top_n=10):
    agg = (shap_df_subset.groupby("feature", as_index=False)["mean_abs_shap"].mean()
           .sort_values("mean_abs_shap", ascending=False))
    return agg["feature"].head(top_n).tolist()



def _summarize_shap_folds(shap_df: pd.DataFrame, topk: int = 3) -> pd.DataFrame:
    """Compute fold-level SHAP diagnostics: total mass, top1/topk share, nonzero count."""
    def _stats(g: pd.DataFrame) -> pd.Series:
        s = g["mean_abs_shap"].astype(float).to_numpy()
        tot = float(np.nansum(s))
        s_sorted = np.sort(s)[::-1]
        top1 = float(s_sorted[0]) if s_sorted.size else 0.0
        topk_sum = float(np.nansum(s_sorted[:topk])) if s_sorted.size else 0.0
        nz = int(np.sum(s > 0))
        return pd.Series({"total": tot,
                          "top1_share": (top1 / tot) if tot > 0 else np.nan,
                          "topk_share": (topk_sum / tot) if tot > 0 else np.nan,
                          "nonzero_feats": nz})
    stats = (shap_df
             .groupby(["subset","alpha_numeric","model","outer_fold"])
             .apply(_stats).reset_index())
    return stats

def filter_bad_shap_folds(shap_df: pd.DataFrame,
                          share_thresh: float = 0.98,
                          min_nonzero: int = 5,
                          iqr_mult: float = 25.0,
                          outdir: str | None = None):
    """Flag and drop anomalous SHAP folds.

    Rules:
      • topk_share >= share_thresh (spiky / dominated by a few features)
      • nonzero_feats < min_nonzero
      • total > median + iqr_mult * IQR   (ridiculously large totals)

    Writes shap_fold_stats.csv and shap_outlier_folds.csv to outdir if provided.
    Returns: (filtered_df, stats_df, outliers_df)
    """
    stats = _summarize_shap_folds(shap_df, topk=3)
    med = stats["total"].median()
    iqr = stats["total"].quantile(0.75) - stats["total"].quantile(0.25)
    big = stats["total"] > (med + iqr_mult * iqr)
    few = stats["nonzero_feats"] < min_nonzero
    spiky = stats["topk_share"] >= share_thresh
    stats["is_outlier"] = big | few | spiky
    bad = stats[stats["is_outlier"]].copy()
    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            stats.to_csv(os.path.join(outdir, "shap_fold_stats.csv"), index=False)
            bad.to_csv(os.path.join(outdir, "shap_outlier_folds.csv"), index=False)
        except Exception as e:
            print(f"[warn] Could not write SHAP stats: {e}")
    key = ["subset","alpha_numeric","model","outer_fold"]
    merged = shap_df.merge(stats[key + ["is_outlier"]], on=key, how="left")
    filtered = merged[~merged["is_outlier"].fillna(False)].drop(columns=["is_outlier"])
    return filtered, stats, bad


# ---------- Stacked bars (Built‑in Importances for RF/GB) ----------

def _pivot_importance(df, top_feats, normalize_rows=True, use_norm=False):
    val_col = "importance_norm" if (use_norm and "importance_norm" in df.columns) else "importance"
    g = (df.groupby(["alpha_numeric","feature"], as_index=False)[val_col].mean())
    pivot = g.pivot(index="alpha_numeric", columns="feature", values=val_col).fillna(0.0)
    for f in top_feats:
        if f not in pivot.columns:
            pivot[f] = 0.0
    out = pivot[top_feats].copy()
    other_cols = [c for c in pivot.columns if c not in top_feats]
    out["Other"] = pivot[other_cols].sum(axis=1) if other_cols else 0.0
    if normalize_rows:
        s = out.sum(axis=1).replace(0, np.nan)
        out = out.div(s, axis=0).fillna(0.0)
    return out.sort_index()


# ---------------------- main ----------------------

def main():
    args = parse_args()

    if args.style:
        try:
            plt.style.use(args.style)
        except Exception:
            print(f"[warn] matplotlib style '{args.style}' not found; proceeding with defaults.")

    # metrics root
    if args.metrics_root:
        metrics_root = os.path.abspath(args.metrics_root)
    else:
        metrics_root = str(SECTION_DIR / "data")

    outdir = os.path.abspath(args.outdir) if args.outdir else str(REPRODUCED_ROOT / "3.5" / "metrics")
    os.makedirs(outdir, exist_ok=True)

    alpha_dirs = find_alpha_dirs(metrics_root)
    if not alpha_dirs:
        raise SystemExit(f"No alpha_* folders found under: {metrics_root}")
    
    
    # ---- Performance metrics ----
    df = read_metrics_summaries(alpha_dirs)
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.lower()
        df = df[df["model"].isin(MODEL_ORDER)]
    
    # Save PR-AUC tables (works even if you skip the ribbons plots)
    if args.save_pr_tables == "on" and "PR_AUC_mean" in df.columns:
        save_pr_auc_tables(df, outdir)

    plot_metric_ribbons(df, "PR_AUC", "PR_AUC_mean", "PR_AUC_ci95_low", "PR_AUC_ci95_high", "PR-AUC", outdir, dpi=args.dpi)


# =============================================================================
# 
#     # ---- SHAP stacked bars ----
#     try:
#         shap_df = read_shap_summaries(alpha_dirs)
#         if not shap_df.empty and (getattr(args, "shap_figs", "on") == "on"):
#             # Filter out anomalous folds before plotting
#             shap_df, shap_stats, shap_bad = filter_bad_shap_folds(
#                 shap_df,
#                 share_thresh=0.98,
#                 min_nonzero=5,
#                 iqr_mult=25.0,
#                 outdir=outdir,
#             )
#             if not shap_bad.empty:
#                 print(f"[info] Dropped {len(shap_bad)} outlier fold(s) for SHAP; see 'shap_outlier_folds.csv'.")
# 
#             topn = getattr(args, "stacked_topn", 10)
#             norm = getattr(args, "stacked_normalize", True)
#             for subset_name in ["hot", "cold"]:
#                 sub = shap_df[shap_df["subset"] == subset_name]
#                 if sub.empty:
#                     continue
#                 top_feats = _pick_top_feats_all_models(sub, top_n=topn)
#                 _plot_stacked_grid(
#                     shap_df=shap_df,
#                     subset_name=subset_name,
#                     model_order=MODEL_ORDER,
#                     top_feats=top_feats,
#                     outdir=outdir,
#                     normalize_rows=norm,
#                     dpi=args.dpi
#                 )
#     except Exception as e:
#         print(f"[warn] SHAP stacked-bars skipped: {e}")
# 
# 
#     # ---- RF/GB built-in importance stacked bars ----
#     try:
#         imp_df = read_builtin_importances(alpha_dirs)
#         if not imp_df.empty and (getattr(args, "imp_figs", "on") == "on"):
#             topn = getattr(args, "imp_topn", 10)
#             norm = getattr(args, "imp_normalize", True)
#             for subset_name in ["hot", "cold"]:
#                 sub = imp_df[(imp_df["subset"] == subset_name) & (imp_df["model"].isin(["rf","gb"]))]
#                 if sub.empty:
#                     continue
#                 agg = (sub.groupby("feature", as_index=False)["importance"].mean()
#                          .sort_values("importance", ascending=False))
#                 top_feats = agg["feature"].head(topn).tolist()
#                 _plot_importance_stacked_grid(
#                     imp_df=imp_df,
#                     subset_name=subset_name,
#                     top_feats=top_feats,
#                     outdir=outdir,
#                     normalize_rows=norm,
#                     use_norm=True,
#                     dpi=args.dpi
#                 )
#     except Exception as e:
#         print(f"[warn] Importance stacked-bars skipped: {e}")
# 
#     print(f"[done] Figures saved to: {outdir}")
# 
# 
# =============================================================================


if __name__ == "__main__":
    import sys
    main()
