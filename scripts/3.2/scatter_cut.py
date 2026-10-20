"""Paper-package script. Output scope is documented in PAPER_OUTPUTS.md at the package root."""

from __future__ import annotations

import os
from pathlib import Path

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
PACKAGE_ROOT = DATA_ROOT / "datasets_and_scripts"
REPRODUCED_ROOT = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve()
SECTION_DIR = PACKAGE_ROOT / "3.2_lift_analysis"
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Batch figure generation; no desktop GUI required.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator
from matplotlib.patches import Patch


# ============================
# Configuration
# ============================
BASE_DIR_DEFAULT = PACKAGE_ROOT / "shared" / "data"
INPUT_CSV_NAME   = "system_daily_table.csv"

# OFV cutoffs (same as reference)
GRID_QS   = (0.97, 0.99, 0.995)
CUT_ORDER = [99.5, 99.0, 97.0]
LIFT_CUT_ORDER = [97.0, 99.0, 99.5]

# Palettes (same hues as reference)
HOT_PALETTE  = {99.5: "#d45087", 99.0: "#ff6e54", 97.0: "#ffa600"}
COLD_PALETTE = {99.5: "#1a237e", 99.0: "#5c6bc0", 97.0: "#7986cb"}
DEFAULT_BELOW = "#BDBDBD"

BACKGROUND_COLOR = "#F5F5F5"

# Bootstrap for RiskRatio panels (same as reference)
BOOT_B = 500
BOOT_SEED = 123


# ============================
# Helper functions (copied/adapted from reference script)
# ============================

def pct_lines(series: pd.Series, qs=GRID_QS) -> dict[float, float]:
    """Return quantile thresholds for a series; keys are 97.0/99.0/99.5."""
    s = series.dropna()
    return {int(q * 1000) / 10: float(s.quantile(q)) for q in qs}


def color_by_OV_dual(
    ov: np.ndarray,
    mask_hot: np.ndarray,
    mask_cold: np.ndarray,
    hot_pcts: dict[float, float],
    cold_pcts: dict[float, float],
    hot_palette: dict[float, str],
    cold_palette: dict[float, str],
    below_color: str = DEFAULT_BELOW,
) -> np.ndarray:
    """Color points by OFV bins, with separate palettes for Hot vs Cold."""
    colors = np.full(len(ov), below_color, dtype=object)

    # HOT bins
    q995, q99, q97 = hot_pcts[99.5], hot_pcts[99.0], hot_pcts[97.0]
    colors[(mask_hot) & (ov >= q97) & (ov < q99)] = hot_palette[97.0]
    colors[(mask_hot) & (ov >= q99) & (ov < q995)] = hot_palette[99.0]
    colors[(mask_hot) & (ov >= q995)] = hot_palette[99.5]

    # COLD bins
    q995, q99, q97 = cold_pcts[99.5], cold_pcts[99.0], cold_pcts[97.0]
    colors[(mask_cold) & (ov >= q97) & (ov < q99)] = cold_palette[97.0]
    colors[(mask_cold) & (ov >= q99) & (ov < q995)] = cold_palette[99.0]
    colors[(mask_cold) & (ov >= q995)] = cold_palette[99.5]

    return colors


def thresholds_from_OV(
    df: pd.DataFrame,
    mask: np.ndarray,
    var_col: str,
    ov_col: str,
    ov_thresholds: dict[float, float],
    method: str,
    q: float = 0.10,
) -> dict[float, float]:
    """Threshold values of var among rows with OV >= each cutoff.

    method:
      - "min": min(var)
      - "max": max(var)
      - "q"  : quantile q of var
    """
    out: dict[float, float] = {}
    base = df.loc[mask].copy()
    for lvl, t in ov_thresholds.items():
        subset = base.loc[base[ov_col] >= t, var_col].dropna()
        if subset.empty:
            out[lvl] = float("nan")
        else:
            if method == "min":
                out[lvl] = float(subset.min())
            elif method == "max":
                out[lvl] = float(subset.max())
            else:
                out[lvl] = float(subset.quantile(q))
    return out


def _format_tau_value(value: float) -> str:
    """Format tau values compactly for line labels."""
    if value is None or not np.isfinite(value):
        return "NA"

    abs_v = abs(float(value))
    if abs_v >= 1_000_000:
        txt = f"{value / 1_000_000:.2f}M"
    elif abs_v >= 10_000:
        txt = f"{value / 1_000:.1f}k"
    elif abs_v >= 1_000:
        txt = f"{value / 1_000:.2f}k"
    elif abs_v >= 100:
        txt = f"{value:.0f}"
    elif abs_v >= 10:
        txt = f"{value:.1f}"
    elif abs_v >= 1:
        txt = f"{value:.2f}"
    else:
        txt = f"{value:.3f}"

    return txt.rstrip("0").rstrip(".") if "." in txt and not txt.endswith("M") and not txt.endswith("k") else txt


def add_tau_line_labels(
    ax,
    line_y_values: dict[float, float],
    tau_values: dict[float, float],
    palette: dict[float, str],
    *,
    x: float = 0.985,
    fontsize: int = 18,
    min_gap_points: float = 24.0,
    display_overrides: dict[float, str] | None = None,
) -> None:
    """Label onset-threshold lines as tau values, with automatic non-overlap.

    line_y_values are the actual plotted y-positions.
    tau_values are the threshold values shown in the label.
    """
    items = []
    for lvl in CUT_ORDER:
        y = line_y_values.get(lvl, np.nan)
        tau = tau_values.get(lvl, np.nan)
        if np.isfinite(y) and np.isfinite(tau):
            items.append((lvl, float(y), float(tau)))

    if not items:
        return

    fig = ax.figure
    fig.canvas.draw()

    # Convert data y positions to display-space y positions.
    y_disp = {
        lvl: ax.transData.transform((0.0, y))[1]
        for lvl, y, _tau in items
    }

    # Enforce a minimum vertical gap between labels in display space.
    min_gap_px = min_gap_points * fig.dpi / 72.0
    order = sorted(items, key=lambda t: y_disp[t[0]])
    adjusted_disp = {}

    prev = None
    for lvl, _y, _tau in order:
        cur = y_disp[lvl]
        if prev is not None and cur - prev < min_gap_px:
            cur = prev + min_gap_px
        adjusted_disp[lvl] = cur
        prev = cur

    # Keep labels inside the axes area with a small margin.
    ylim = ax.get_ylim()
    y_min_disp = ax.transData.transform((0.0, min(ylim)))[1]
    y_max_disp = ax.transData.transform((0.0, max(ylim)))[1]
    margin_px = max(4.0, (fontsize * 0.9 + 4.0) * fig.dpi / 72.0)

    top = max(adjusted_disp.values())
    if top > y_max_disp - margin_px:
        shift = top - (y_max_disp - margin_px)
        for lvl in adjusted_disp:
            adjusted_disp[lvl] -= shift

    bottom = min(adjusted_disp.values())
    if bottom < y_min_disp + margin_px:
        shift = (y_min_disp + margin_px) - bottom
        for lvl in adjusted_disp:
            adjusted_disp[lvl] += shift

    # Annotate, using point offsets from the original line positions.
    for lvl, y, tau in items:
        delta_points = (adjusted_disp[lvl] - y_disp[lvl]) * 72.0 / fig.dpi
        ax.annotate(
            rf"$\tau_{{{lvl:g}}}$ = {(display_overrides or {}).get(lvl, _format_tau_value(tau))}",
            xy=(x, y),
            xycoords=ax.get_yaxis_transform(),  # x in axes fraction, y in data units
            xytext=(0, delta_points),
            textcoords="offset points",
            ha="right",
            va="center",
            color=palette[lvl],
            fontsize=fontsize,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=BACKGROUND_COLOR,
                edgecolor="none",
                alpha=0.88,
            ),
            zorder=5,
            clip_on=True,
        )


def tau_mean_lookup(boot_tbl: pd.DataFrame, regime: str, variable: str) -> dict[float, float]:
    """Return {OV_cutoff: Tau_mean} from the bootstrap table for one regime/variable."""
    out: dict[float, float] = {}
    if boot_tbl is None or boot_tbl.empty or "Tau_mean" not in boot_tbl.columns:
        return out

    sub = boot_tbl[(boot_tbl["Regime"] == regime) & (boot_tbl["Variable"] == variable)]
    for _, row in sub.iterrows():
        cutoff = row.get("OV_cutoff", np.nan)
        tau = row.get("Tau_mean", np.nan)
        if np.isfinite(cutoff) and np.isfinite(tau):
            out[float(cutoff)] = float(tau)
    return out


def save_ofv_bin_legend(out_dir: str) -> None:
    """Save the OFV-bin legend as a standalone figure (same as reference)."""
    A = Patch(facecolor=HOT_PALETTE[99.5])
    B = Patch(facecolor=HOT_PALETTE[99.0])
    C = Patch(facecolor=HOT_PALETTE[97.0])
    D = Patch(facecolor=DEFAULT_BELOW)

    E = Patch(facecolor=COLD_PALETTE[99.5])
    F = Patch(facecolor=COLD_PALETTE[99.0])
    G = Patch(facecolor=COLD_PALETTE[97.0])
    H = Patch(facecolor=(0, 0, 0, 0), edgecolor=(0, 0, 0, 0))  # spacer

    labels = [
        "Hot OFV ≥ 99.5%", "Cold OFV ≥ 99.5%",
        "Hot 99–99.5%", "Cold 99–99.5%",
        "Hot 97–99%", "Cold 97–99%",
        "OFV < 97%", " ",
    ]
    handles = [A, E, B, F, C, G, D, H]

    fig_leg = plt.figure(figsize=(11, 2.2))
    fig_leg.patch.set_alpha(0)

    leg = fig_leg.legend(
        handles=handles,
        labels=labels,
        title="Resource Adequacy Stress (OFV) Bin",
        loc="center",
        ncol=4,
        framealpha=1,
        fontsize=22,
        title_fontsize=22,
        columnspacing=1.0,
        handlelength=1.6,
    )
    leg.get_title().set_fontweight("bold")
    plt.axis("off")

    fig_leg.savefig(
        os.path.join(out_dir, "Legend_OFV_Bin.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig_leg)




def plot_signed_degree_days(
    df: pd.DataFrame,
    hot_mask: np.ndarray,
    cold_mask: np.ndarray,
    hot_OV_pcts: dict[float, float],
    cold_OV_pcts: dict[float, float],
    out_dir: str,
    *,
    hot_tau_override: dict[float, float] | None = None,
    cold_tau_override: dict[float, float] | None = None,
) -> None:
    """Hot/Cld signed DD scatter colored by OFV bins with DD-threshold lines."""
    # where OV crosses cutoffs (same as reference)
    threshold_method = "min"
    q_lower = 0.10

    CDD_at_OV = thresholds_from_OV(
        df, hot_mask, "CDD", "ObjectiveValue", hot_OV_pcts,
        method=threshold_method, q=q_lower
    )
    HDD_at_OV = thresholds_from_OV(
        df, cold_mask, "HDD", "ObjectiveValue", cold_OV_pcts,
        method=threshold_method, q=q_lower
    )

    # Use bootstrap-mean tau values for the dashed lines when available.
    if hot_tau_override:
        CDD_at_OV.update({lvl: val for lvl, val in hot_tau_override.items() if np.isfinite(val)})
    if cold_tau_override:
        HDD_at_OV.update({lvl: val for lvl, val in cold_tau_override.items() if np.isfinite(val)})

    # Preferred display/plot value from user: use 39.0 for tau_97 of HDD.
    HDD_at_OV[97.0] = 39.0

    colors = color_by_OV_dual(
        df["ObjectiveValue"].to_numpy(),
        hot_mask, cold_mask,
        hot_OV_pcts, cold_OV_pcts,
        HOT_PALETTE, COLD_PALETTE,
        below_color=DEFAULT_BELOW,
    )

    y_hot = np.where(hot_mask, df["CDD"].to_numpy(), np.nan)
    y_cold = np.where(cold_mask, -df["HDD"].to_numpy(), np.nan)

    fig = plt.figure(figsize=(11, 10))
    ax = plt.gca()
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.scatter(df["Date"], y_cold, color=colors, alpha=0.85, s=20, marker="o", linewidths=0, zorder=2)
    ax.scatter(df["Date"], y_hot, color=colors, alpha=0.85, s=20, marker="o", linewidths=0, zorder=2)

    ax.axhline(0, color="black", linestyle="-", linewidth=1, zorder=1)

    # DD threshold lines (same mapping/colors)
    ax.axhline(y=CDD_at_OV[99.5], color=HOT_PALETTE[99.5], linestyle="--", linewidth=2)
    ax.axhline(y=CDD_at_OV[99.0], color=HOT_PALETTE[99.0], linestyle="--", linewidth=2)
    ax.axhline(y=CDD_at_OV[97.0], color=HOT_PALETTE[97.0], linestyle="--", linewidth=2)

    ax.axhline(y=-HDD_at_OV[99.5], color=COLD_PALETTE[99.5], linestyle="--", linewidth=2)
    ax.axhline(y=-HDD_at_OV[99.0], color=COLD_PALETTE[99.0], linestyle="--", linewidth=2)
    ax.axhline(y=-HDD_at_OV[97.0], color=COLD_PALETTE[97.0], linestyle="--", linewidth=2)

    # y-lims (same; cap hot)
    hot_cap = 25.0
    max_cold_obs = float(np.nanmax(np.where(cold_mask, df["HDD"].to_numpy(), np.nan)))
    ax.set_ylim(-1.05 * (0 if math.isnan(max_cold_obs) else max_cold_obs), hot_cap)

    add_tau_line_labels(
        ax,
        line_y_values=CDD_at_OV,
        tau_values=CDD_at_OV,
        palette=HOT_PALETTE,
    )
    add_tau_line_labels(
        ax,
        line_y_values={lvl: -v for lvl, v in HDD_at_OV.items()},
        tau_values=HDD_at_OV,
        palette=COLD_PALETTE,
        display_overrides={97.0: "39.0"},
    )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):.0f}"))

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel("Month", fontsize=24, fontweight="bold", labelpad=20)
    ax.set_ylabel("Degree-Days", fontsize=24, fontweight="bold", labelpad=60)

    ax.text(-0.09, 0.88, "CDD", transform=ax.transAxes, rotation=90,
            fontsize=24, fontweight="bold", va="center", ha="center")
    ax.text(-0.09, 0.38, "HDD", transform=ax.transAxes, rotation=90,
            fontsize=24, fontweight="bold", va="center", ha="center")

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    plt.subplots_adjust(top=0.94)
    plt.tight_layout()

    # Keep your filename; also save a "noLegend" alias for consistency
    p1 = os.path.join(out_dir, "HotCold_posDD_colorByOV_withWeatherLines.png")
    p2 = os.path.join(out_dir, "HotCold_posDD_colorByOV_withWeatherLines_noLegend.png")
    plt.savefig(p2, dpi=300)
    plt.close(fig)


def plot_signed_coverage(
    df: pd.DataFrame,
    hot_mask: np.ndarray,
    cold_mask: np.ndarray,
    hot_OV_pcts: dict[float, float],
    cold_OV_pcts: dict[float, float],
    out_dir: str,
    *,
    hot_tau_override: dict[float, float] | None = None,
    cold_tau_override: dict[float, float] | None = None,
) -> None:
    """Hot/Cld signed coverage scatter (above/below avg) colored by OFV bins."""
    cov_hot_col = "Above_Avg_Coverage"
    cov_cold_col = "Below_Avg_Coverage"

    def threshold_from_OV(mask: np.ndarray, cov_col: str, ov_thresholds: dict[float, float]):
        out = {}
        base = df.loc[mask]
        for lvl, t in ov_thresholds.items():
            subset = base.loc[base["ObjectiveValue"] >= t, cov_col].dropna()
            out[lvl] = float(subset.min()) if not subset.empty else float("nan")
        return out

    cov_hot_thr = threshold_from_OV(hot_mask, cov_hot_col, hot_OV_pcts)
    cov_cold_thr = threshold_from_OV(cold_mask, cov_cold_col, cold_OV_pcts)

    # Use bootstrap-mean tau values for the dashed lines when available.
    if hot_tau_override:
        cov_hot_thr.update({lvl: val for lvl, val in hot_tau_override.items() if np.isfinite(val)})
    if cold_tau_override:
        cov_cold_thr.update({lvl: val for lvl, val in cold_tau_override.items() if np.isfinite(val)})

    colors = color_by_OV_dual(
        df["ObjectiveValue"].to_numpy(),
        hot_mask, cold_mask,
        hot_OV_pcts, cold_OV_pcts,
        HOT_PALETTE, COLD_PALETTE,
        below_color=DEFAULT_BELOW,
    )

    y_hot = np.where(hot_mask, df[cov_hot_col].to_numpy(), np.nan)
    y_cold = np.where(cold_mask, -df[cov_cold_col].to_numpy(), np.nan)

    fig = plt.figure(figsize=(11, 10))
    ax = plt.gca()
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.scatter(df["Date"], y_cold, color=colors, alpha=0.85, s=20, marker="o", linewidths=0, zorder=2)
    ax.scatter(df["Date"], y_hot, color=colors, alpha=0.85, s=20, marker="o", linewidths=0, zorder=2)

    ax.axhline(0, color="black", linestyle="-", linewidth=1, zorder=1)

    ax.axhline(y=cov_hot_thr[99.5], color=HOT_PALETTE[99.5], linestyle="--", linewidth=2)
    ax.axhline(y=cov_hot_thr[99.0], color=HOT_PALETTE[99.0], linestyle="--", linewidth=2)
    ax.axhline(y=cov_hot_thr[97.0], color=HOT_PALETTE[97.0], linestyle="--", linewidth=2)

    ax.axhline(y=-cov_cold_thr[99.5], color=COLD_PALETTE[99.5], linestyle="--", linewidth=2)
    ax.axhline(y=-cov_cold_thr[99.0], color=COLD_PALETTE[99.0], linestyle="--", linewidth=2)
    ax.axhline(y=-cov_cold_thr[97.0], color=COLD_PALETTE[97.0], linestyle="--", linewidth=2)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel("Month", fontsize=24, fontweight="bold", labelpad=20)
    ax.set_ylabel("Coverage Fraction", fontsize=24, fontweight="bold", labelpad=50)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):.2f}"))

    ax.text(-0.12, 0.78, "Hot", transform=ax.transAxes, rotation=90,
            fontsize=24, fontweight="bold", va="center", ha="center")
    ax.text(-0.12, 0.22, "Cold", transform=ax.transAxes, rotation=90,
            fontsize=24, fontweight="bold", va="center", ha="center")

    max_hot = float(np.nanmax(y_hot)) if np.any(~np.isnan(y_hot)) else 0.0
    max_cold = float(np.nanmax(-y_cold)) if np.any(~np.isnan(y_cold)) else 0.0
    ax.set_ylim(-1.05 * max_cold, 1.05 * max_hot)

    add_tau_line_labels(
        ax,
        line_y_values=cov_hot_thr,
        tau_values=cov_hot_thr,
        palette=HOT_PALETTE,
    )
    add_tau_line_labels(
        ax,
        line_y_values={lvl: -v for lvl, v in cov_cold_thr.items()},
        tau_values=cov_cold_thr,
        palette=COLD_PALETTE,
    )

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Coverage_pos_signed_colorByOV_withLines.png"), dpi=300)
    plt.close(fig)



def _thr_method_from_direction(direction: str | None, default: str = "max") -> str:
    """Map direction ('high'/'low') to thresholds_from_OV method ('min'/'max').

    - direction == 'high' means the onset condition is X >= tau, and tau is the MIN value among OV-tail days.
    - direction == 'low'  means the onset condition is X <= tau, and tau is the MAX value among OV-tail days.

    If direction is None, returns `default`.
    """
    if direction is None:
        return default
    direction = str(direction).lower()
    if direction.startswith("hi"):
        return "min"
    if direction.startswith("lo"):
        return "max"
    return default

def plot_signed_metric_by_OV(
    df: pd.DataFrame,
    metric_col: str,
    hot_mask: np.ndarray,
    cold_mask: np.ndarray,
    hot_OV_pcts: dict[float, float],
    cold_OV_pcts: dict[float, float],
    save_path: str,
    y_label: str,
    *,
    direction_hot: str | None = None,
    direction_cold: str | None = None,
    threshold_method: str = "max",
    q_lower: float = 0.90,
    hot_tau_override: dict[float, float] | None = None,
    cold_tau_override: dict[float, float] | None = None,
) -> None:
    """Signed Hot/Cold scatter for a single metric (reference style).

    IMPORTANT:
      The dashed cutoff lines are *onset thresholds* derived from OV-tail days.
      They are aligned with the Lift/RiskRatio panel direction choice:

        - direction == 'high': event is X >= tau, tau = min(X | OV >= cutoff)
        - direction == 'low' : event is X <= tau, tau = max(X | OV >= cutoff)

      For the Cold half of the signed plot we negate the metric, so the cold cutoff
      line is plotted at -tau.
    """
    method_hot = _thr_method_from_direction(direction_hot, default=threshold_method)
    method_cold = _thr_method_from_direction(direction_cold, default=threshold_method)

    hot_thr = thresholds_from_OV(df, hot_mask, metric_col, "ObjectiveValue", hot_OV_pcts, method=method_hot, q=q_lower)
    cold_thr = thresholds_from_OV(df, cold_mask, metric_col, "ObjectiveValue", cold_OV_pcts, method=method_cold, q=q_lower)

    # Use bootstrap-mean tau values for the dashed lines when available.
    if hot_tau_override:
        hot_thr.update({lvl: val for lvl, val in hot_tau_override.items() if np.isfinite(val)})
    if cold_tau_override:
        cold_thr.update({lvl: val for lvl, val in cold_tau_override.items() if np.isfinite(val)})

    colors = color_by_OV_dual(
        df["ObjectiveValue"].to_numpy(),
        hot_mask, cold_mask,
        hot_OV_pcts, cold_OV_pcts,
        HOT_PALETTE, COLD_PALETTE,
        below_color=DEFAULT_BELOW,
    )

    y_hot = np.where(hot_mask, df[metric_col].to_numpy(), np.nan)
    y_cold = np.where(cold_mask, -df[metric_col].to_numpy(), np.nan)

    # Force visible multiplier by scaling ourselves (same as reference)
    max_abs = 0.0
    if np.any(~np.isnan(y_hot)):
        max_abs = max(max_abs, float(np.nanmax(np.abs(y_hot))))
    if np.any(~np.isnan(y_cold)):
        max_abs = max(max_abs, float(np.nanmax(np.abs(y_cold))))

    scale_exp_overrides = {"Wind": 5}
    exp = scale_exp_overrides.get(metric_col, int(np.floor(np.log10(max_abs))) if max_abs > 0 else 0)
    scale = (10.0 ** exp) if exp != 0 else 1.0

    y_hot_s = y_hot / scale
    y_cold_s = y_cold / scale
    hot_thr_s = {k: (v / scale if (v is not None and not np.isnan(v)) else np.nan) for k, v in hot_thr.items()}
    cold_thr_s = {k: (v / scale if (v is not None and not np.isnan(v)) else np.nan) for k, v in cold_thr.items()}

    fig = plt.figure(figsize=(11, 10))
    ax = plt.gca()
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.scatter(df["Date"], y_cold_s, color=colors, alpha=0.85, s=20, marker="o", linewidths=0, zorder=2)
    ax.scatter(df["Date"], y_hot_s, color=colors, alpha=0.85, s=20, marker="o", linewidths=0, zorder=2)

    ax.axhline(0, color="black", linestyle="-", linewidth=1, zorder=1)

    ax.axhline(y=hot_thr_s[99.5], color=HOT_PALETTE[99.5], linestyle="--", linewidth=2)
    ax.axhline(y=hot_thr_s[99.0], color=HOT_PALETTE[99.0], linestyle="--", linewidth=2)
    ax.axhline(y=hot_thr_s[97.0], color=HOT_PALETTE[97.0], linestyle="--", linewidth=2)

    ax.axhline(y=-cold_thr_s[99.5], color=COLD_PALETTE[99.5], linestyle="--", linewidth=2)
    ax.axhline(y=-cold_thr_s[99.0], color=COLD_PALETTE[99.0], linestyle="--", linewidth=2)
    ax.axhline(y=-cold_thr_s[97.0], color=COLD_PALETTE[97.0], linestyle="--", linewidth=2)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel("Month", fontsize=24, fontweight="bold", labelpad=20)
    ax.set_ylabel(y_label, fontsize=24, fontweight="bold", labelpad=50)

    max_hot = float(np.nanmax(y_hot_s)) if np.any(~np.isnan(y_hot_s)) else 0.0
    max_cold = float(np.nanmax(-y_cold_s)) if np.any(~np.isnan(y_cold_s)) else 0.0
    ax.set_ylim(-1.05 * max_cold, 1.05 * max_hot)

    add_tau_line_labels(
        ax,
        line_y_values=hot_thr_s,
        tau_values=hot_thr_s,
        palette=HOT_PALETTE,
    )
    add_tau_line_labels(
        ax,
        line_y_values={lvl: -v for lvl, v in cold_thr_s.items()},
        tau_values=cold_thr_s,
        palette=COLD_PALETTE,
    )

    tick_step_overrides = {"Wind": 2.0}
    ax.yaxis.set_major_locator(MultipleLocator(tick_step_overrides.get(metric_col, 0.5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):.1f}"))

    ax.text(-0.12, 0.78, "Hot", transform=ax.transAxes,
            rotation=90, fontsize=24, fontweight="bold", va="center", ha="center")
    ax.text(-0.12, 0.22, "Cold", transform=ax.transAxes,
            rotation=90, fontsize=24, fontweight="bold", va="center", ha="center")

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    plt.tight_layout(rect=[0.14, 0.04, 0.98, 0.96])

    if exp != 0:
        ax.text(0.00, 1.02, rf"$\times 10^{{{exp}}}$",
                transform=ax.transAxes, fontsize=20, ha="left", va="bottom")

    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# ============================
# RiskRatio panels (bootstrap)
# ============================

def strict_threshold_from_OV(df: pd.DataFrame, mask: np.ndarray, var_col: str, ov_col: str, ov_thr: float, direction: str) -> float:
    """Strict threshold tau: high -> min(var|OV>=thr); low -> max(var|OV>=thr)."""
    sub = df.loc[mask & (df[ov_col] >= ov_thr), var_col].dropna()
    if sub.empty:
        return float("nan")
    return float(sub.min() if direction == "high" else sub.max())


def conditional_stats(df: pd.DataFrame, mask: np.ndarray, var_col: str, direction: str, tau: float, ov_col: str, ov_thr: float) -> dict:
    """Compute P_tail|cond and RR for one tau and OV cutoff."""
    base = df.loc[mask].copy()
    base = base.dropna(subset=[var_col, ov_col])
    if base.empty or not np.isfinite(tau):
        return {"Threshold": np.nan, "P_tail_given_condition": np.nan, "RiskRatio": np.nan}

    baseline = float((base[ov_col] >= ov_thr).mean())
    if direction == "high":
        cond = base[var_col] >= tau
    else:
        cond = base[var_col] <= tau

    if cond.sum() == 0 or baseline <= 0:
        return {"Threshold": tau, "P_tail_given_condition": np.nan, "RiskRatio": np.nan}

    p_tail_given = float(((base.loc[cond, ov_col] >= ov_thr).mean()))
    rr = p_tail_given / baseline if baseline > 0 else np.nan

    return {"Threshold": tau, "P_tail_given_condition": p_tail_given, "RiskRatio": rr}


def choose_direction_by_rr(df_sub: pd.DataFrame, var_col: str, ov_col: str, cutoff_key: float = 99.5) -> str:
    """Pick direction (high vs low) that yields the larger RR at the given cutoff."""
    ov_pcts = pct_lines(df_sub[ov_col])
    thr = ov_pcts[cutoff_key]

    # High direction
    tau_hi = strict_threshold_from_OV(df_sub, np.ones(len(df_sub), dtype=bool), var_col, ov_col, thr, "high")
    rr_hi = conditional_stats(df_sub, np.ones(len(df_sub), dtype=bool), var_col, "high", tau_hi, ov_col, thr)["RiskRatio"]

    # Low direction
    tau_lo = strict_threshold_from_OV(df_sub, np.ones(len(df_sub), dtype=bool), var_col, ov_col, thr, "low")
    rr_lo = conditional_stats(df_sub, np.ones(len(df_sub), dtype=bool), var_col, "low", tau_lo, ov_col, thr)["RiskRatio"]

    # Prefer finite RR; fall back to high
    if np.isfinite(rr_hi) and np.isfinite(rr_lo):
        return "high" if rr_hi >= rr_lo else "low"
    if np.isfinite(rr_hi):
        return "high"
    if np.isfinite(rr_lo):
        return "low"
    return "high"




def _q_from_cutoff(cutoff: float) -> float:
    """Map 97.0/99.0/99.5 -> 0.97/0.99/0.995."""
    if cutoff == 99.5:
        return 0.995
    if cutoff == 99.0:
        return 0.99
    if cutoff == 97.0:
        return 0.97
    raise ValueError(f"Unexpected cutoff: {cutoff}")


def bootstrap_threshold_lift(
    df: pd.DataFrame,
    regime_name: str,
    mask_regime: np.ndarray,
    specs: list[tuple[str, str]],
    B: int = BOOT_B,
    seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """Fast bootstrap thresholds & RR for a regime.

    Key speed-up vs the earlier implementation:
    - Each bootstrap replicate samples indices ONCE.
    - OV quantile thresholds and tail masks are computed ONCE per replicate.
    - All variables reuse the same replicate's tail masks.

    Output schema matches the plotting code.
    """
    rng = np.random.default_rng(seed)

    df_sub = df.loc[mask_regime].copy()
    if df_sub.empty:
        return pd.DataFrame()

    ov_full = df_sub["ObjectiveValue"].to_numpy(dtype=float)
    n = len(ov_full)

    # Unique vars and their direction
    specs_u = []
    seen = set()
    for var_col, direction in specs:
        key = (var_col, direction)
        if key in seen:
            continue
        if var_col not in df_sub.columns:
            continue
        seen.add(key)
        specs_u.append(key)

    # Build arrays once
    var_arrays = {var_col: df_sub[var_col].to_numpy(dtype=float) for var_col, _ in specs_u}

    # Full-sample OV thresholds and baseline probs
    ov_thr_full = {c: float(np.nanquantile(ov_full, _q_from_cutoff(c))) for c in CUT_ORDER}
    base_p_full = {c: float(np.nanmean(ov_full >= ov_thr_full[c])) for c in CUT_ORDER}

    # Bootstrap storage
    rr_bs = {(var_col, direction, c): np.full(B, np.nan, dtype=float) for var_col, direction in specs_u for c in CUT_ORDER}
    tau_bs = {(var_col, direction, c): np.full(B, np.nan, dtype=float) for var_col, direction in specs_u for c in CUT_ORDER}

    # Precompute full-sample point estimates
    rows = []
    for var_col, direction in specs_u:
        x_full = var_arrays[var_col]
        for c in CUT_ORDER:
            thr = ov_thr_full[c]
            tail = ov_full >= thr
            baseline = base_p_full[c]

            xt = x_full[tail]
            xt = xt[np.isfinite(xt)]
            tau = float(np.nan)
            if xt.size:
                tau = float(np.nanmin(xt) if direction == "high" else np.nanmax(xt))

            if np.isfinite(tau) and baseline > 0:
                cond = (x_full >= tau) if direction == "high" else (x_full <= tau)
                denom = np.nanmean(cond)
                if denom > 0:
                    p_tail_given = float(np.nanmean(tail[cond]))
                    rr = p_tail_given / baseline
                else:
                    p_tail_given = np.nan
                    rr = np.nan
            else:
                p_tail_given = np.nan
                rr = np.nan

            rows.append({
                "Regime": regime_name,
                "Variable": var_col,
                "Direction": direction,
                "OV_cutoff": c,
                "Threshold": tau,  # full-sample tau
                "Tau_mean": np.nan,  # bootstrap-mean tau used for scatterplot lines
                "Tau_lo": np.nan,    # 2.5th percentile of bootstrap tau
                "Tau_hi": np.nan,    # 97.5th percentile of bootstrap tau
                "Tau_n_valid": 0,    # number of finite bootstrap tau values
                "P_tail_given_condition": p_tail_given,
                "RiskRatio": rr,
                "RiskRatio_mean": np.nan,
                "RiskRatio_lo": np.nan,
                "RiskRatio_hi": np.nan,
            })

    # Bootstrap replicates
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ov = ov_full[idx]

        thr = {c: float(np.nanquantile(ov, _q_from_cutoff(c))) for c in CUT_ORDER}
        tail = {c: (ov >= thr[c]) for c in CUT_ORDER}
        baseline = {c: float(np.nanmean(tail[c])) for c in CUT_ORDER}

        for var_col, direction in specs_u:
            x = var_arrays[var_col][idx]

            for c in CUT_ORDER:
                if baseline[c] <= 0:
                    continue

                xt = x[tail[c]]
                xt = xt[np.isfinite(xt)]
                if xt.size == 0:
                    continue

                tau = float(np.nanmin(xt) if direction == "high" else np.nanmax(xt))
                if not np.isfinite(tau):
                    continue

                tau_bs[(var_col, direction, c)][b] = tau

                cond = (x >= tau) if direction == "high" else (x <= tau)
                denom = np.nanmean(cond)
                if denom <= 0:
                    continue

                p_tail_given = float(np.nanmean(tail[c][cond]))
                rr_bs[(var_col, direction, c)][b] = p_tail_given / baseline[c]

    # Fill bootstrap summaries
    for r in rows:
        key = (r["Variable"], r["Direction"], r["OV_cutoff"])

        tau_vals = tau_bs[key]
        tau_finite = tau_vals[np.isfinite(tau_vals)]
        if tau_finite.size:
            r["Tau_mean"] = float(np.nanmean(tau_finite))
            r["Tau_lo"] = float(np.nanpercentile(tau_finite, 2.5))
            r["Tau_hi"] = float(np.nanpercentile(tau_finite, 97.5))
            r["Tau_n_valid"] = int(tau_finite.size)
        else:
            r["Tau_mean"] = np.nan
            r["Tau_lo"] = np.nan
            r["Tau_hi"] = np.nan
            r["Tau_n_valid"] = 0

        vals = rr_bs[key]
        if np.isfinite(vals).any():
            r["RiskRatio_mean"] = float(np.nanmean(vals))
            r["RiskRatio_lo"] = float(np.nanpercentile(vals, 2.5))
            r["RiskRatio_hi"] = float(np.nanpercentile(vals, 97.5))
        else:
            r["RiskRatio_mean"] = np.nan
            r["RiskRatio_lo"] = np.nan
            r["RiskRatio_hi"] = np.nan

    return pd.DataFrame(rows)

def style_clean_two_axes(fig, ax) -> None:
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("white")
    fig.patch.set_linewidth(0)
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="both", which="both", top=False, right=False, direction="out")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def compute_shared_ylim_and_ticks(tbl: pd.DataFrame) -> tuple[tuple[float, float], list[float]]:
    """Compute shared y-limits/ticks across Hot & Cold risk ratio plots."""
    if tbl.empty:
        return (0.0, 2.0), [0, 1, 2]

    top = 65.0
    ticks = [float(t) for t in np.arange(0, 61, 10)]
    return (0.0, top), ticks


def plot_riskratio_panel(tbl: pd.DataFrame, regime: str, outfile: str, legend_title: str, y_lim_shared, y_ticks_shared) -> None:
    sub = tbl[tbl["Regime"] == regime].copy()
    if sub.empty:
        print(f"[skip] no data for {regime}")
        return

    # Rank variables by mean RR at 99.5% within this regime (high->low)
    rank = (
        sub[sub["OV_cutoff"] == 99.5]
        .set_index("Variable")["RiskRatio_mean"]
        .sort_values(ascending=False)
    )
    var_order = [v for v in rank.index.tolist() if (sub["Variable"] == v).any()]

    # Build display labels with threshold direction arrows
    dir_map = (
        sub[sub["OV_cutoff"] == 99.5]
        .drop_duplicates(subset=["Variable"])
        .set_index("Variable")["Direction"]
        .to_dict()
    )
    
    def _arrow(d):
        if str(d).lower().startswith("hi"):
            return "↑"
        if str(d).lower().startswith("lo"):
            return "↓"
        return ""
    
    label_map = {v: f"{v} {_arrow(dir_map.get(v))}".strip() for v in var_order}
    display_labels = [label_map[v] for v in var_order]

    # palettes
    palette = HOT_PALETTE if regime == "Hot" else COLD_PALETTE

    # Only plot onset-threshold bars. The former grey "All Days" bars are removed.
    SERIES_ORDER = LIFT_CUT_ORDER

    FS_AXIS_LABEL = 26
    FS_TICKS = 20
    FS_TITLE_LG = 27
    FS_TITLE_SM = 22
    FS_LEGEND = 28
    FS_LEGEND_TTL = 22

    fig, ax = plt.subplots(figsize=(11.0, 8.5))

    x = np.arange(len(var_order), dtype=float)
    nseries = len(SERIES_ORDER)
    width = min(0.18, 0.75 / nseries)
    center = -(nseries - 1) / 2.0
    offsets = {k: (center + i) * width for i, k in enumerate(SERIES_ORDER)}

    style_clean_two_axes(fig, ax)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, zorder=1)

    # bars
    ymax_hard = 0.0
    for c in SERIES_ORDER:
        subc = sub[sub["OV_cutoff"] == c].set_index("Variable").reindex(var_order)
        vals = subc["RiskRatio_mean"].to_numpy(dtype=float)
        lo = subc["RiskRatio_lo"].to_numpy(dtype=float)
        hi = subc["RiskRatio_hi"].to_numpy(dtype=float)
        errs = np.vstack([vals - lo, hi - vals])

        ax.bar(x + offsets[c], vals, width=width, color=palette[c], edgecolor="none", zorder=2)
        ax.errorbar(x + offsets[c], vals, yerr=errs, fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=3)

        if np.isfinite(hi).any():
            ymax_hard = max(ymax_hard, float(np.nanmax(hi)))

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=FS_TICKS, fontweight="bold")
    # robust rotation (tick labels, not tick marks)
    ax.tick_params(axis="x", pad=12, labelrotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    ax.tick_params(axis="y", labelsize=FS_TICKS)
    ax.set_title(regime, fontsize=FS_TITLE_SM, fontweight="bold", pad=16)

    ax.set_ylim(*y_lim_shared)
    ax.set_yticks(y_ticks_shared)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Legend: variable conditions. Direction is variable-specific:
    #   ↑ variables use X >= tau_alpha; ↓ variables use X <= tau_alpha.
    legend_labels = [
        rf"$X \geq \tau_{{{c:g}}}$ (↑) or $X \leq \tau_{{{c:g}}}$ (↓)"
        for c in CUT_ORDER
    ]
    legend_labels = [rf"$\alpha$ = {c:g}" for c in SERIES_ORDER]
    ax.legend(
        handles=[Patch(facecolor=palette[c]) for c in SERIES_ORDER],
        labels=legend_labels,
        title=legend_title,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.83),  # decrease the second value to move the legend farther down
        frameon=False,
        ncol=1,
        fontsize=FS_LEGEND,
        borderaxespad=0.35,
        handlelength=1.0,
        labelspacing=0.75,  # increase this value for more space between legend lines
    ).set_title(legend_title, prop={"size": FS_LEGEND_TTL, "weight": "bold"})

    # y-axis label outside
    fig.text(0.055, 0.5, "Lift", rotation=90, va="center", ha="center",
             fontsize=FS_AXIS_LABEL, fontweight="bold", color="black")

#    fig.suptitle(
#        "How much each threshold increases grid-stress risk (Risk ratio) — bars show mean with 95% CI",
#        fontsize=FS_TITLE_LG,
#        fontweight="bold",
#        y=0.98,
#    )

    fig.subplots_adjust(bottom=0.32)
    plt.tight_layout(rect=[0.085, 0.05, 0.995, 0.93])

    # --- FORCE x tick label rotation at the very end (after layout) ---
    fig.canvas.draw()  # ensure tick labels exist
    ax.tick_params(axis="x", labelrotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    ax.tick_params(axis="x", labelsize=FS_TICKS)

    plt.savefig(outfile, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[saved] {outfile}")


# ============================
# Main
# ============================

def main() -> None:
    in_csv = BASE_DIR_DEFAULT / INPUT_CSV_NAME
    if not in_csv.is_file():
        raise FileNotFoundError(in_csv)
    out_dir = REPRODUCED_ROOT / "3.2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(in_csv)

    # Harmonize coverage naming for reusing old logic
    df = df.rename(columns={"above_avg": "Above_Avg_Coverage", "below_avg": "Below_Avg_Coverage"})

    # Month-only x-axis anchor (leap year handles Day=366)
    anchor_year = 2000
    df["Date"] = pd.to_datetime(f"{anchor_year}-01-01") + pd.to_timedelta(df["Day"] - 1, unit="D")

    # Hot/cold masks
    hot_mask = df["CDD"].fillna(0).to_numpy() > 0
    cold_mask = df["HDD"].fillna(0).to_numpy() > 0

    # Derived variables
    if ("U10" in df.columns) and ("V10" in df.columns):
        df["WSPD"] = np.sqrt(df["U10"].to_numpy() ** 2 + df["V10"].to_numpy() ** 2)

    # OFV percentiles within regimes (same as reference: by_regime)
    hot_OV_pcts = pct_lines(df.loc[hot_mask, "ObjectiveValue"])
    cold_OV_pcts = pct_lines(df.loc[cold_mask, "ObjectiveValue"])

    # OFV distribution histograms by regime

    # ------------------------------------------------------------------
    # Direction scan (per regime, per variable) for onset thresholds
    #   - 'high' means the stress-associated tail is HIGH values of X
    #   - 'low'  means the stress-associated tail is LOW values of X
    #
    # We compute directions ONCE here and reuse them for:
    #   (1) Lift/RiskRatio panels
    #   (2) The dashed cutoff lines in the signed scatterplots
    # This guarantees the scatter cutoff lines align with the lift-panel direction.
    # ------------------------------------------------------------------

    # Build a unified Coverage variable used for direction-scan and lift:
    #   Hot  -> Above_Avg_Coverage
    #   Cold -> Below_Avg_Coverage
    df["Coverage"] = np.where(
        hot_mask,
        df["Above_Avg_Coverage"].to_numpy(),
        np.where(cold_mask, df["Below_Avg_Coverage"].to_numpy(), np.nan),
    )

    hot_vars = ["CDD", "Coverage", "Solar", "Wind", "Q2", "WSPD", "SWDOWN", "GLW", "WindFrac", "SolarFrac"]
    cold_vars = ["HDD", "Coverage", "Solar", "Wind", "Q2", "WSPD", "SWDOWN", "GLW", "WindFrac", "SolarFrac"]

    fixed_high = {"CDD", "HDD", "Coverage"}                  # severity / coverage: higher means more extreme
    fixed_low  = {"Solar", "Wind", "WindFrac", "SolarFrac"}  # renewables / fractions: lower means more extreme

    dir_hot: dict[str, str] = {}
    dir_cold: dict[str, str] = {}

    df_hot = df.loc[hot_mask].copy()
    df_cold = df.loc[cold_mask].copy()

    for v in hot_vars:
        if v not in df.columns:
            continue
        if v in fixed_high:
            d = "high"
        elif v in fixed_low:
            d = "low"
        else:
            sub2 = df_hot[[v, "ObjectiveValue"]].dropna()
            d = choose_direction_by_rr(sub2, v, "ObjectiveValue", cutoff_key=99.5) if not sub2.empty else "high"
        dir_hot[v] = d

    for v in cold_vars:
        if v not in df.columns:
            continue
        if v in fixed_high:
            d = "high"
        elif v in fixed_low:
            d = "low"
        else:
            sub2 = df_cold[[v, "ObjectiveValue"]].dropna()
            d = choose_direction_by_rr(sub2, v, "ObjectiveValue", cutoff_key=99.5) if not sub2.empty else "high"
        dir_cold[v] = d

    print("[INFO] Direction scan (Hot):", dir_hot)
    print("[INFO] Direction scan (Cold):", dir_cold)

    # --- Bootstrap thresholds and lift summaries ---
    # Reuse the SAME direction scan for both scatter threshold lines and lift.
    hot_specs = [(v, dir_hot[v]) for v in hot_vars if v in dir_hot]
    cold_specs = [(v, dir_cold[v]) for v in cold_vars if v in dir_cold]

    print("[INFO] Hot directions (used for BOTH scatter lines & lift):", {v: d for v, d in hot_specs})
    print("[INFO] Cold directions (used for BOTH scatter lines & lift):", {v: d for v, d in cold_specs})

    hot_tbl = bootstrap_threshold_lift(df, "Hot", hot_mask, hot_specs, B=BOOT_B, seed=BOOT_SEED)
    cold_tbl = bootstrap_threshold_lift(df, "Cold", cold_mask, cold_specs, B=BOOT_B, seed=BOOT_SEED + 1)

    boot_tbl = pd.concat([hot_tbl, cold_tbl], ignore_index=True)
    boot_csv = os.path.join(out_dir, "thresholds_and_tail_probs_by_metric_BOOTSTRAP_allvars.csv")
    boot_tbl.to_csv(boot_csv, index=False)
    print(f"[SAVE] {boot_csv}")

    # Save a compact CSV focused on bootstrap onset-threshold summaries.
    tau_summary_cols = [
        "Regime", "Variable", "Direction", "OV_cutoff",
        "Threshold", "Tau_mean", "Tau_lo", "Tau_hi", "Tau_n_valid",
    ]
    tau_summary = boot_tbl[tau_summary_cols].copy()
    tau_summary = tau_summary.sort_values(["Regime", "Variable", "OV_cutoff"], ascending=[True, True, False])
    tau_csv = os.path.join(out_dir, "bootstrap_onset_thresholds_mean_95CI.csv")
    tau_summary.to_csv(tau_csv, index=False)
    print(f"[SAVE] {tau_csv}")

    # --- plots ---
    plot_signed_degree_days(
        df, hot_mask, cold_mask, hot_OV_pcts, cold_OV_pcts, out_dir,
        hot_tau_override=tau_mean_lookup(boot_tbl, "Hot", "CDD"),
        cold_tau_override=tau_mean_lookup(boot_tbl, "Cold", "HDD"),
    )
    plot_signed_coverage(
        df, hot_mask, cold_mask, hot_OV_pcts, cold_OV_pcts, out_dir,
        hot_tau_override=tau_mean_lookup(boot_tbl, "Hot", "Coverage"),
        cold_tau_override=tau_mean_lookup(boot_tbl, "Cold", "Coverage"),
    )

    save_ofv_bin_legend(out_dir)

    # Solar/Wind (same parameters as reference)
    plot_signed_metric_by_OV(
        df, "Solar", hot_mask, cold_mask,
        hot_OV_pcts, cold_OV_pcts,
        save_path=os.path.join(out_dir, "HotCold_solar_colorByOV_withLines.png"),
        y_label="Solar Power Production (MWh)",
        direction_hot=dir_hot.get("Solar"),
        direction_cold=dir_cold.get("Solar"),
        q_lower=0.90,
        hot_tau_override=tau_mean_lookup(boot_tbl, "Hot", "Solar"),
        cold_tau_override=tau_mean_lookup(boot_tbl, "Cold", "Solar"),
    )

    plot_signed_metric_by_OV(
        df, "Wind", hot_mask, cold_mask,
        hot_OV_pcts, cold_OV_pcts,
        save_path=os.path.join(out_dir, "HotCold_wind_colorByOV_withLines.png"),
        y_label="Wind Power Production (MWh)",
        direction_hot=dir_hot.get("Wind"),
        direction_cold=dir_cold.get("Wind"),
        q_lower=0.90,
        hot_tau_override=tau_mean_lookup(boot_tbl, "Hot", "Wind"),
        cold_tau_override=tau_mean_lookup(boot_tbl, "Cold", "Wind"),
    )

    # --- RiskRatio / Lift panels ---
    # Keep these in boot_tbl so their scatterplots can still use bootstrap-mean tau,
    # but remove them from the Lift panels.
    lift_plot_tbl = boot_tbl[~boot_tbl["Variable"].isin(["GLW", "WindFrac", "SolarFrac"])].copy()

    y_lim_shared, y_ticks_shared = compute_shared_ylim_and_ticks(lift_plot_tbl)

    plot_riskratio_panel(
        lift_plot_tbl,
        regime="Hot",
        outfile=os.path.join(out_dir, "Lift_Hot.png"),
        legend_title="RA Stress Level",
        y_lim_shared=y_lim_shared,
        y_ticks_shared=y_ticks_shared,
    )

    plot_riskratio_panel(
        lift_plot_tbl,
        regime="Cold",
        outfile=os.path.join(out_dir, "Lift_Cold.png"),
        legend_title="RA Stress Level",
        y_lim_shared=y_lim_shared,
        y_ticks_shared=y_ticks_shared,
    )

    print(f"[DONE] Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
