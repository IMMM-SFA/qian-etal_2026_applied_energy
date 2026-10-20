# -*- coding: utf-8 -*-
"""Paper-package script. Output scope is documented in PAPER_OUTPUTS.md at the package root."""

from __future__ import annotations

import os
from pathlib import Path

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
PACKAGE_ROOT = DATA_ROOT / "datasets_and_scripts"
REPRODUCED_ROOT = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve()
SECTION_DIR = PACKAGE_ROOT / "3.4_compound_amplification"
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')  # Batch figure generation; no desktop GUI required.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator


# =============================================================================
# USER CONFIG
# =============================================================================

OUT_DIR = REPRODUCED_ROOT / "3.4" / "figures"
HOT_CSV = SECTION_DIR / "data" / "lift_amplification_gridsearch_bootstrap_HOT.csv"
COLD_CSV = SECTION_DIR / "data" / "lift_amplification_gridsearch_bootstrap_COLD.csv"

SAVE_PNG = True
SAVE_PDF = False
DPI = 300

# Reciprocal limits make multiplicative changes equally intense: 0.5 and 2.0
# match, as do 1/3 and 3.0. Values shown on the colorbar remain raw ratios.
AMPL_VMIN = 1.0 / 3.0
AMPL_CENTER = 1.0
AMPL_VMAX = 3.0

# The original panels excluded the 0th-percentile boundary cells.
DROP_ZERO_PCT = True
SHOW_ALL_TICKS = False

PRODUCTION_SCALE = 1e5
PRODUCTION_SCALE_LABEL = "×10⁵ MWh"
TICK_SIGDIG = 3

TICK_LABEL_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 18
COLORBAR_FONTSIZE = 18

GRIDLINE = True
GRIDLINE_LW = 0.3

MARK_NAN_WITH_X = True
NAN_X_COLOR = "#333333"
NAN_X_SIZE = 20
NAN_X_LW = 0.9
NAN_TRANSPARENT = True


# =============================================================================
# Helpers
# =============================================================================

@dataclass(frozen=True)
class DistributionSummary:
    family: str
    finite_cells: int
    total_cells: int
    minimum: float
    p025: float
    median: float
    p975: float
    maximum: float
    below_vmin: int
    above_vmax: int


def _validate_inputs(df: pd.DataFrame, family: str) -> None:
    required = {
        "Alpha",
        "TempTau",
        "TempCol",
        "WindPctLo",
        "SolarPctLo",
        "WindThr",
        "SolarThr",
        "Ampl_BootMean",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{family} CSV is missing columns: {missing}")

    if "ResourceBasis" in df.columns:
        bases = set(df["ResourceBasis"].dropna().astype(str).unique())
        if bases != {"raw_production"}:
            raise ValueError(
                f"{family} CSV has unexpected ResourceBasis values: {sorted(bases)}"
            )


def _prepare(df: pd.DataFrame, family: str) -> pd.DataFrame:
    _validate_inputs(df, family)
    out = df.copy()
    for column in [
        "Alpha",
        "TempTau",
        "WindPctLo",
        "SolarPctLo",
        "WindThr",
        "SolarThr",
        "Ampl_BootMean",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if DROP_ZERO_PCT:
        out = out[(out["WindPctLo"] != 0) & (out["SolarPctLo"] != 0)].copy()

    out["WindPctLo"] = out["WindPctLo"].astype(int)
    out["SolarPctLo"] = out["SolarPctLo"].astype(int)
    return out


def _ratio_summary(df: pd.DataFrame, family: str) -> DistributionSummary:
    values = df["Ampl_BootMean"].to_numpy(float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{family} has no finite amplification ratios.")
    quantiles = np.quantile(finite, [0.025, 0.5, 0.975])
    return DistributionSummary(
        family=family,
        finite_cells=int(finite.size),
        total_cells=int(values.size),
        minimum=float(np.min(finite)),
        p025=float(quantiles[0]),
        median=float(quantiles[1]),
        p975=float(quantiles[2]),
        maximum=float(np.max(finite)),
        below_vmin=int(np.sum(finite < AMPL_VMIN)),
        above_vmax=int(np.sum(finite > AMPL_VMAX)),
    )


def _print_ratio_summary(summary: DistributionSummary) -> None:
    print(
        f"[{summary.family}] finite={summary.finite_cells}/{summary.total_cells}; "
        f"min={summary.minimum:.3f}; p2.5={summary.p025:.3f}; "
        f"median={summary.median:.3f}; p97.5={summary.p975:.3f}; "
        f"max={summary.maximum:.3f}; below {AMPL_VMIN:g}={summary.below_vmin}; "
        f"above {AMPL_VMAX:g}={summary.above_vmax}"
    )


def _hot_ratio_cmap() -> LinearSegmentedColormap:
    # attenuation -> neutral -> HOT amplification
    return LinearSegmentedColormap.from_list(
        "hot_ratio_green_white_red",
        ["#00441b", "#a1d99b", "#ffffff", "#fcbba1", "#67000d"],
        N=256,
    )


def _cold_ratio_cmap() -> LinearSegmentedColormap:
    # attenuation -> neutral -> COLD amplification
    return LinearSegmentedColormap.from_list(
        "cold_ratio_green_white_blue",
        ["#00441b", "#a1d99b", "#ffffff", "#c6dbef", "#08306b"],
        N=256,
    )


def _format_significant(value: float, sig: int = 3) -> str:
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0." + "0" * (sig - 1)
    digits_left = int(np.floor(np.log10(abs(value)))) + 1
    decimals = max(sig - digits_left, 0)
    return f"{value:.{decimals}f}"


def _prep_matrix(df: pd.DataFrame, pcts: list[int]) -> np.ndarray:
    return (
        df.pivot(
            index="WindPctLo", columns="SolarPctLo", values="Ampl_BootMean"
        )
        .reindex(index=pcts, columns=pcts)
        .to_numpy(float)
    )


def _thresholds_for_panel(
    df: pd.DataFrame, pcts: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    wind = (
        df.drop_duplicates("WindPctLo")
        .set_index("WindPctLo")
        .reindex(pcts)["WindThr"]
        .to_numpy(float)
    )
    solar = (
        df.drop_duplicates("SolarPctLo")
        .set_index("SolarPctLo")
        .reindex(pcts)["SolarThr"]
        .to_numpy(float)
    )
    return wind, solar


def _set_ticks(
    ax: plt.Axes,
    pcts: list[int],
    wind_thresholds: np.ndarray,
    solar_thresholds: np.ndarray,
    show_ylabel: bool,
) -> None:
    n = len(pcts)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    solar_labels = [
        _format_significant(value / PRODUCTION_SCALE, TICK_SIGDIG)
        for value in solar_thresholds
    ]
    wind_labels = [
        _format_significant(value / PRODUCTION_SCALE, TICK_SIGDIG)
        for value in wind_thresholds
    ]

    if SHOW_ALL_TICKS:
        keep = set(range(n))
    else:
        # Display the threshold corresponding to every 10th percentile.
        keep = {i for i, pct in enumerate(pcts) if pct % 10 == 0}

    ax.set_xticklabels(
        [label if i in keep else "" for i, label in enumerate(solar_labels)],
        rotation=90,
        fontsize=TICK_LABEL_FONTSIZE,
    )
    ax.set_yticklabels(
        [label if i in keep else "" for i, label in enumerate(wind_labels)],
        fontsize=TICK_LABEL_FONTSIZE,
    )

    ax.set_xlabel(
        f"≤ Solar availability ({PRODUCTION_SCALE_LABEL})",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight="bold",
        labelpad=8,
    )
    if show_ylabel:
        ax.set_ylabel(
            f"≤ Wind availability ({PRODUCTION_SCALE_LABEL})",
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight="bold",
            labelpad=8,
        )


def _add_grid(ax: plt.Axes, n: int) -> None:
    if not GRIDLINE:
        return
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="k", linewidth=GRIDLINE_LW, alpha=0.15)
    ax.tick_params(which="minor", bottom=False, left=False)


def _overlay_missing_cells(ax: plt.Axes, matrix: np.ndarray) -> None:
    if not MARK_NAN_WITH_X:
        return
    missing = np.argwhere(~np.isfinite(matrix))
    if missing.size == 0:
        return
    ax.scatter(
        missing[:, 1],
        missing[:, 0],
        marker="x",
        s=NAN_X_SIZE,
        c=NAN_X_COLOR,
        linewidths=NAN_X_LW,
        zorder=5,
    )


def _format_temp_threshold(temp_col: str, threshold: float) -> str:
    if temp_col == "HDD" and np.isclose(threshold, 39.0):
        return "39.0"
    return f"{threshold:g}"


def plot_family(
    df: pd.DataFrame,
    family: str,
    cmap: LinearSegmentedColormap,
    out_stem: str,
) -> None:
    pcts = sorted(df["WindPctLo"].unique().astype(int).tolist())
    alphas = sorted(df["Alpha"].unique())
    n = len(pcts)

    if NAN_TRANSPARENT:
        cmap.set_bad((1, 1, 1, 0))
    else:
        cmap.set_bad("white")

    if not np.isclose(AMPL_VMIN * AMPL_VMAX, AMPL_CENTER**2):
        raise ValueError("Amplification color limits must be reciprocal around 1.0.")
    norm = LogNorm(vmin=AMPL_VMIN, vmax=AMPL_VMAX)
    fig, axes = plt.subplots(
        1, len(alphas), figsize=(18.5, 5.8)
    )
    fig.subplots_adjust(
        left=0.10, right=0.90, bottom=0.22, top=0.82, wspace=0.34
    )
    if len(alphas) == 1:
        axes = [axes]

    images = []
    for panel_i, (ax, alpha) in enumerate(zip(axes, alphas)):
        panel = df[df["Alpha"] == alpha].copy()
        matrix = _prep_matrix(panel, pcts)
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            norm=norm,
        )
        images.append(image)
        _overlay_missing_cells(ax, matrix)

        wind_thresholds, solar_thresholds = _thresholds_for_panel(panel, pcts)
        temp_col = str(panel["TempCol"].iloc[0])
        temp_tau = float(panel["TempTau"].iloc[0])
        ax.set_title(
            f"α = {alpha:g} | {temp_col} ≥ "
            f"{_format_temp_threshold(temp_col, temp_tau)}",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            pad=12,
        )
        _set_ticks(
            ax,
            pcts,
            wind_thresholds,
            solar_thresholds,
            show_ylabel=(panel_i == 0),
        )
        _add_grid(ax, n)

    finite_ratios = df["Ampl_BootMean"].to_numpy(float)
    finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
    below = bool(np.any(finite_ratios < AMPL_VMIN))
    above = bool(np.any(finite_ratios > AMPL_VMAX))
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    colorbar = fig.colorbar(
        images[-1], ax=axes, shrink=0.88, pad=0.02, extend=extend
    )
    colorbar.set_label(
        "Amplification ratio",
        fontsize=COLORBAR_FONTSIZE,
        fontweight="bold",
        labelpad=4,
    )
    colorbar.ax.tick_params(labelsize=COLORBAR_FONTSIZE)
    ratio_ticks = [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0]
    colorbar.ax.yaxis.set_major_locator(FixedLocator(ratio_ticks))
    colorbar.ax.yaxis.set_major_formatter(
        FixedFormatter(["0.3", "0.5", "1.0", "2.0", "3.0"])
    )
    colorbar.ax.yaxis.set_minor_locator(NullLocator())

    fig.suptitle(
        family,
        x=0.49,
        y=0.97,
        ha="center",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )

    if SAVE_PNG:
        fig.savefig(
            os.path.join(OUT_DIR, f"{out_stem}.png"),
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.20,
        )
    if SAVE_PDF:
        fig.savefig(
            os.path.join(OUT_DIR, f"{out_stem}.pdf"),
            bbox_inches="tight",
            pad_inches=0.20,
        )
    plt.close(fig)


def main() -> None:
    for path in [HOT_CSV, COLD_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run the raw-production compute script first.")

    os.makedirs(OUT_DIR, exist_ok=True)
    hot = _prepare(pd.read_csv(HOT_CSV), "HOT")
    cold = _prepare(pd.read_csv(COLD_CSV), "COLD")

    _print_ratio_summary(_ratio_summary(hot, "HOT"))
    _print_ratio_summary(_ratio_summary(cold, "COLD"))

    plot_family(
        hot,
        family="HOT",
        cmap=_hot_ratio_cmap(),
        out_stem="HOT_ampl_ratio_raw_production",
    )
    plot_family(
        cold,
        family="COLD",
        cmap=_cold_ratio_cmap(),
        out_stem="COLD_ampl_ratio_raw_production",
    )
    print("Saved direct-ratio plots into:", OUT_DIR)


if __name__ == "__main__":
    main()
