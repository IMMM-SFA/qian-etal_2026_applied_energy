#!/usr/bin/env python3
"""Paper-package script. Output scope is documented in PAPER_OUTPUTS.md at the package root."""

from __future__ import annotations

import argparse
from pathlib import Path

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
PACKAGE_ROOT = DATA_ROOT / "datasets_and_scripts"
REPRODUCED_ROOT = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve()
SECTION_DIR = PACKAGE_ROOT / "3.3_spatial_coverage"

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import numpy as np
from typing import Iterable

GENERAL_ALPHAS = (80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 95.0)
EXTREME_ALPHAS = (97.0, 99.0, 99.5)
ALL_ALPHAS = (*GENERAL_ALPHAS, *EXTREME_ALPHAS)

HOT_DARK = "#8B0000"
HOT_LIGHT = "#FFA07A"
COLD_DARK = "#00008B"
COLD_LIGHT = "#87CEFA"


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    """Raise a useful error if an input table does not contain required fields."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def read_daily_table(path: Path) -> pd.DataFrame:
    """Read the daily stress table and retain fields needed by this analysis."""
    daily = pd.read_csv(path)
    require_columns(daily, ("Date", "ObjectiveValue", "CDD", "HDD"), path)

    daily = daily[["Date", "ObjectiveValue", "CDD", "HDD"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="raise").dt.normalize()
    for column in ("ObjectiveValue", "CDD", "HDD"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")

    if daily["Date"].duplicated().any():
        examples = daily.loc[daily["Date"].duplicated(False), "Date"].head().tolist()
        raise ValueError(f"Daily table has duplicate dates, including: {examples}")

    incomplete = daily[["ObjectiveValue", "CDD", "HDD"]].isna().any(axis=1)
    if incomplete.any():
        print(f"[daily] Dropping {int(incomplete.sum())} rows with missing ObjectiveValue/CDD/HDD.")
        daily = daily.loc[~incomplete].copy()

    daily["Subset"] = np.select(
        [daily["CDD"] > 0, daily["HDD"] > 0],
        ["Hot", "Cold"],
        default="Neither",
    )
    neither = daily["Subset"].eq("Neither")
    if neither.any():
        print(f"[daily] Excluding {int(neither.sum())} days with neither CDD > 0 nor HDD > 0.")
        daily = daily.loc[~neither].copy()
    return daily.reset_index(drop=True)


def read_hourly_coverage(path: Path) -> pd.DataFrame:
    """Convert an hour-of-year x year coverage matrix to a tidy hourly table."""
    wide = pd.read_csv(path)
    require_columns(wide, ("DateTime",), path)
    year_columns = [column for column in wide.columns if str(column).isdigit()]
    if not year_columns:
        raise ValueError(f"{path} has no four-digit year columns.")

    # Parse using a non-leap reference year.  These inputs contain 8,760 rows
    # and intentionally omit February 29 in every year.
    clock = pd.to_datetime(
        "2001-" + wide["DateTime"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M",
        errors="raise",
    )
    if clock.duplicated().any():
        raise ValueError(f"{path} contains duplicate DateTime rows.")
    if len(clock) % 24 != 0:
        raise ValueError(f"{path} has {len(clock)} rows, not a whole number of days.")

    matrix = wide[year_columns].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        count = int(matrix.isna().sum().sum())
        raise ValueError(f"{path} contains {count} missing/non-numeric coverage values.")
    if ((matrix < 0) | (matrix > 1)).any().any():
        raise ValueError(f"{path} contains coverage values outside [0, 1].")

    long = wide[["DateTime", *year_columns]].melt(
        id_vars="DateTime",
        value_vars=year_columns,
        var_name="Year",
        value_name="Coverage",
    )
    repeated_clock = pd.concat(
        [clock.reset_index(drop=True)] * len(year_columns), ignore_index=True
    )
    long["Year"] = pd.to_numeric(long["Year"], errors="raise").astype(int)
    long["Coverage"] = pd.to_numeric(long["Coverage"], errors="raise")
    long["Date"] = pd.to_datetime(
        {
            "year": long["Year"],
            "month": repeated_clock.dt.month,
            "day": repeated_clock.dt.day,
        },
        errors="raise",
    )
    long["Hour"] = repeated_clock.dt.hour.to_numpy()
    return long[["Date", "Hour", "Coverage"]].sort_values(["Date", "Hour"]).reset_index(drop=True)


def align_subset(
    daily: pd.DataFrame,
    coverage: pd.DataFrame,
    subset: str,
) -> pd.DataFrame:
    """Attach all 24 hourly coverage values to each eligible daily row."""
    selected = daily.loc[daily["Subset"].eq(subset)].copy()
    aligned = selected.merge(coverage, on="Date", how="inner", validate="one_to_many")

    expected_dates = set(selected["Date"])
    matched_dates = set(aligned["Date"])
    missing_dates = sorted(expected_dates - matched_dates)
    if missing_dates:
        preview = ", ".join(str(date.date()) for date in missing_dates[:5])
        raise ValueError(f"{subset}: {len(missing_dates)} daily dates lack coverage (e.g., {preview}).")

    counts = aligned.groupby("Date", sort=False).size()
    bad = counts[counts != 24]
    if not bad.empty:
        preview = ", ".join(f"{date.date()}={count}" for date, count in bad.head().items())
        raise ValueError(f"{subset}: expected 24 coverage values per day; found {preview}.")

    aligned["Subset"] = subset
    return aligned[
        ["Date", "Hour", "Subset", "Coverage", "ObjectiveValue", "CDD", "HDD"]
    ].sort_values(["Date", "Hour"]).reset_index(drop=True)


def select_stress_dates(
    daily: pd.DataFrame,
    alphas: Iterable[float] = ALL_ALPHAS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select high-objective days at each within-subset percentile."""
    selected_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for subset in ("Hot", "Cold"):
        eligible = daily.loc[daily["Subset"].eq(subset)].copy()
        values = eligible["ObjectiveValue"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No {subset} days are available for stress selection.")

        for alpha in alphas:
            threshold = float(np.quantile(values, float(alpha) / 100.0))
            chosen = eligible.loc[
                eligible["ObjectiveValue"] >= threshold,
                ["Date", "Subset", "ObjectiveValue", "CDD", "HDD"],
            ].copy()
            chosen["Alpha"] = float(alpha)
            chosen["ObjectiveThreshold"] = threshold
            chosen = chosen[
                [
                    "Date",
                    "Subset",
                    "Alpha",
                    "ObjectiveThreshold",
                    "ObjectiveValue",
                    "CDD",
                    "HDD",
                ]
            ]
            selected_frames.append(chosen)
            summaries.append(
                {
                    "Subset": subset,
                    "Alpha": float(alpha),
                    "ObjectiveThreshold": threshold,
                    "EligibleDays": int(len(eligible)),
                    "SelectedDays": int(len(chosen)),
                    "SelectedHours": int(len(chosen) * 24),
                }
            )

    dates = pd.concat(selected_frames, ignore_index=True).sort_values(
        ["Subset", "Alpha", "Date"]
    )
    summary = pd.DataFrame(summaries).sort_values(["Subset", "Alpha"])
    return dates.reset_index(drop=True), summary.reset_index(drop=True)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted observations and their empirical cumulative probabilities."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return x, np.arange(1, x.size + 1, dtype=float) / x.size


def alpha_date_set(stress_dates: pd.DataFrame, subset: str, alpha: float) -> set[pd.Timestamp]:
    mask = stress_dates["Subset"].eq(subset) & np.isclose(stress_dates["Alpha"], alpha)
    return set(stress_dates.loc[mask, "Date"])


def build_plot_data(
    hourly: pd.DataFrame,
    stress_dates: pd.DataFrame,
    subset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the three ECDF curves and general-stress quantile envelope."""
    curve_frames: list[pd.DataFrame] = []
    for alpha in EXTREME_ALPHAS:
        dates = alpha_date_set(stress_dates, subset, alpha)
        values = hourly.loc[hourly["Date"].isin(dates), "Coverage"].to_numpy()
        x, y = ecdf(values)
        if x.size == 0:
            raise ValueError(f"{subset}: no hourly coverage values for alpha={alpha:g}.")
        curve_frames.append(
            pd.DataFrame({"Subset": subset, "Alpha": alpha, "Coverage": x, "ECDF": y})
        )

    p_grid = np.linspace(0.0, 1.0, 201)
    quantiles: list[np.ndarray] = []
    for alpha in GENERAL_ALPHAS:
        dates = alpha_date_set(stress_dates, subset, alpha)
        values = hourly.loc[hourly["Date"].isin(dates), "Coverage"].to_numpy()
        if values.size == 0:
            raise ValueError(f"{subset}: no hourly coverage values for alpha={alpha:g}.")
        quantiles.append(np.quantile(values, p_grid))

    q_matrix = np.vstack(quantiles)
    band = pd.DataFrame(
        {
            "Subset": subset,
            "ECDF": p_grid,
            "CoverageMin": np.nanmin(q_matrix, axis=0),
            "CoverageMax": np.nanmax(q_matrix, axis=0),
        }
    )
    return pd.concat(curve_frames, ignore_index=True), band


def save_figure(
    subset: str,
    curves: pd.DataFrame,
    band: pd.DataFrame,
    output_path: Path,
    benchmark_values: np.ndarray | pd.Series | None = None,
    benchmark_label: str | None = None,
) -> None:
    """Save one Hot or Cold ECDF figure, optionally with a gray benchmark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if subset == "Hot":
        dark, light = HOT_DARK, HOT_LIGHT
    else:
        dark, light = COLD_DARK, COLD_LIGHT

    styles = {
        97.0: {"ls": ":", "lw": 2.0},
        99.0: {"ls": "--", "lw": 2.0, "dashes": (5, 4)},
        99.5: {"ls": "-", "lw": 2.3},
    }
    rc = {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.8, 5.0), facecolor="white")
        ax.set_facecolor("white")
        ax.fill_betweenx(
            band["ECDF"],
            band["CoverageMin"],
            band["CoverageMax"],
            color=light,
            alpha=0.24,
            lw=0,
            zorder=1,
        )
        benchmark_style = {"color": "#888888", "ls": "-", "lw": 1.9, "alpha": 0.95}
        has_benchmark = benchmark_values is not None
        if has_benchmark:
            benchmark_x, benchmark_y = ecdf(np.asarray(benchmark_values, dtype=float))
            if not benchmark_x.size:
                raise ValueError(f"{subset}: benchmark coverage contains no valid values.")
            ax.plot(
                benchmark_x,
                benchmark_y,
                zorder=2,
                **benchmark_style,
            )
        for alpha in EXTREME_ALPHAS:
            line = curves.loc[np.isclose(curves["Alpha"], alpha)]
            ax.plot(
                line["Coverage"],
                line["ECDF"],
                color=dark,
                alpha=0.95,
                zorder=3,
                **styles[alpha],
            )

        handles: list[object] = [
            Line2D([0], [0], color=dark, **styles[99.5]),
            Line2D([0], [0], color=dark, **styles[99.0]),
            Line2D([0], [0], color=dark, **styles[97.0]),
            Patch(facecolor=light, edgecolor="none", alpha=0.24),
        ]
        labels: list[str] = [
            "α = 99.5",
            "α = 99",
            "α = 97",
            "Band α∈{80, 82, 84, 86, 88, 90, 92, 95}",
        ]
        if has_benchmark:
            handles.append(Line2D([0], [0], **benchmark_style))
            labels.append(benchmark_label or f"All {subset} days")
        ax.legend(
            handles,
            labels,
            loc="best",
            frameon=has_benchmark,
            facecolor="white",
            edgecolor="none",
            framealpha=0.92,
            fontsize=10.5,
            handlelength=3.43,
            handletextpad=0.8,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(subset, pad=9)
        ax.set_xlabel("Coverage", labelpad=5)
        ax.set_ylabel("ECDF", labelpad=5)
        ax.grid(False)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.20)
        fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
        plt.close(fig)




SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DAILY_TABLE = PACKAGE_ROOT / "shared" / "data" / "system_daily_table.csv"
DEFAULT_ABOVE_COVERAGE = PACKAGE_ROOT / "shared" / "upstream_processed" / "TGW_results" / "above_avg_coverage.csv"
DEFAULT_BELOW_COVERAGE = PACKAGE_ROOT / "shared" / "upstream_processed" / "TGW_results" / "below_avg_coverage.csv"
DEFAULT_OUT_DIR = REPRODUCED_ROOT / "3.3"


def hourly_to_daily_mean(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapse 24 hourly coverage values to one arithmetic mean per date."""
    daily = (
        hourly.groupby(["Date", "Subset"], as_index=False, sort=True)
        .agg(
            Coverage=("Coverage", "mean"),
            SourceHours=("Hour", "size"),
            ObjectiveValue=("ObjectiveValue", "first"),
            CDD=("CDD", "first"),
            HDD=("HDD", "first"),
        )
        .sort_values(["Subset", "Date"])
        .reset_index(drop=True)
    )
    bad = daily.loc[daily["SourceHours"] != 24]
    if not bad.empty:
        preview = ", ".join(
            f"{row.Date.date()}={row.SourceHours}" for row in bad.head().itertuples()
        )
        raise ValueError(f"Expected 24 source hours for every daily mean; found {preview}.")
    return daily[
        [
            "Date",
            "Subset",
            "Coverage",
            "SourceHours",
            "ObjectiveValue",
            "CDD",
            "HDD",
        ]
    ]


def main(
    daily_table_path: Path,
    above_coverage_path: Path,
    below_coverage_path: Path,
    output_dir: Path,
) -> None:
    """Create two daily-coverage figures and their reproducibility tables."""
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    daily_inputs = read_daily_table(daily_table_path)
    above_hourly = read_hourly_coverage(above_coverage_path)
    below_hourly = read_hourly_coverage(below_coverage_path)

    hot_hourly = align_subset(daily_inputs, above_hourly, "Hot")
    cold_hourly = align_subset(daily_inputs, below_hourly, "Cold")
    daily_coverage = hourly_to_daily_mean(
        pd.concat([hot_hourly, cold_hourly], ignore_index=True)
    )
    stress_dates, summary = select_stress_dates(daily_inputs)
    summary = summary.rename(columns={"SelectedHours": "SourceHourlyValues"})
    summary.insert(
        summary.columns.get_loc("SourceHourlyValues"),
        "SelectedDailyValues",
        summary["SelectedDays"],
    )

    all_curves: list[pd.DataFrame] = []
    all_bands: list[pd.DataFrame] = []
    for subset in ("Hot", "Cold"):
        subset_daily = daily_coverage.loc[daily_coverage["Subset"].eq(subset)]
        curves, band = build_plot_data(subset_daily, stress_dates, subset)
        all_curves.append(curves)
        all_bands.append(band)
        save_figure(
            subset,
            curves,
            band,
            figure_dir
            / f"fig_ecdf_daily_coverage_{subset.upper()}_ALL_DAYS_BENCHMARK.png",
            benchmark_values=subset_daily["Coverage"].dropna().to_numpy(),
            benchmark_label=f"All {subset} days",
        )

    daily_coverage.to_csv(
        data_dir / "daily_mean_coverage_by_hot_cold_subset.csv", index=False
    )
    stress_dates.to_csv(data_dir / "stress_dates_by_subset.csv", index=False)
    summary.to_csv(data_dir / "stress_selection_summary.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(
        data_dir / "ecdf_extreme_curves.csv", index=False
    )
    pd.concat(all_bands, ignore_index=True).to_csv(
        data_dir / "ecdf_general_band.csv", index=False
    )

    print(
        "Saved two all-days benchmark figures; "
        f"saved five data tables under: {output_dir}"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot daily-mean coverage ECDFs on high-stress Hot/Cold days."
    )
    parser.add_argument("--daily-table", type=Path, default=DEFAULT_DAILY_TABLE)
    parser.add_argument("--above-coverage", type=Path, default=DEFAULT_ABOVE_COVERAGE)
    parser.add_argument("--below-coverage", type=Path, default=DEFAULT_BELOW_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    arguments = parser.parse_args()
    main(
        arguments.daily_table,
        arguments.above_coverage,
        arguments.below_coverage,
        arguments.output_dir,
    )
