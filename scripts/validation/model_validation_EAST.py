# -*- coding: utf-8 -*-
"""Compare 2019 daily simulated load-weighted LMPs with historical day-ahead
LMPs for ISNE, NYIS, PJM, MISO and SWPP. All paths resolve from this script.
Inputs: best_model_settings/sim_LMPs, data_allocation, histr_LMPs,
and the sibling Model/Data/Interconnections bus-to-BA mapping.
Outputs: only the five individual BA figures in best_model_settings/LMPs_comparion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
VALIDATION_DIR = DATA_ROOT / "DCOPF-EAST" / "best_model_settings"
MODEL_RESULTS_ROOT = VALIDATION_DIR
INTERCONNECTION_DIR = VALIDATION_DIR.parent / "Model" / "Data" / "Interconnections"
HIST_ROOT = VALIDATION_DIR / "histr_LMPs"
SIM_LMP_ROOT = VALIDATION_DIR / "sim_LMPs"
OUTPUT_DIR = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve() / "validation" / "LMPs_comparion"
BA_LOAD_ROOT = VALIDATION_DIR / "data_allocation"

YEARS = [2019]
NN_VALUES = [500]
FUEL_TAG = "fp"
BA_WEIGHT_OUTPUT = OUTPUT_DIR / (
    f"EAST_year_specific_BA_load_weights_{min(YEARS)}_{max(YEARS)}.csv"
)

# The renamed simulation folder contains the fixed best-model case.
# Its metadata is explicit because the folder name no longer encodes it.
SIM_YEAR = 2019
SIM_NN = 500
SIM_TRANS_MODE = "pct"
SIM_TRANS_VALUE = 300

# If False, a missing prepared daily historical file skips only that market/year.
STRICT_HISTORICAL_FILES = False
SKIP_INCOMPLETE_CASES = True
DROP_INVALID_DUALS = True
INVALID_DUAL_VALUES = {-999}
SIM_LMP_SIGN = 1.0
MARKET_ORDER = ["ISNE", "NYIS", "PJM", "MISO", "SWPP"]
BA_TO_MARKET = {
    "ISO New England": "ISNE",
    "New York Independent System Operator": "NYIS",
    "PJM Interconnection, LLC": "PJM",
    "Midcontinent Independent System Operator, Inc.": "MISO",
    "Southwest Power Pool": "SWPP",
}
HISTORICAL_FILES = {
    market: HIST_ROOT / (market.lower() + "_daily_da_lmp_{year}.csv")
    for market in MARKET_ORDER
}

# Match 1_Model_Validation (new).py / plots/line_mod_003, scaled from
# 60 inches at 72 dpi to 14.4 inches at 300 dpi (4320 pixels square).
# Keep the style local to plotting; no seaborn dependency is required.
PLOT_FIGSIZE = (14.4, 14.4)
PLOT_LINEWIDTH = 1.68
PLOT_AXIS_FONTSIZE = 43.2
PLOT_TITLE_FONTSIZE = 24
PLOT_METRIC_FONTSIZE = 38.4
PLOT_MONTH_FONTSIZE = 19.2
PLOT_YTICK_FONTSIZE = 31.68
PLOT_TICK_PAD = 2.4
PLOT_AXIS_LABEL_PAD = 2.4
PLOT_DPI = 300
PLOT_STYLE = {
    "axes.facecolor": "#EAEAF2",
    "axes.edgecolor": "white",
    "axes.labelcolor": ".15",
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 0.192,
    "axes.linewidth": 0.24,
    "text.color": ".15",
    "xtick.color": ".15",
    "ytick.color": ".15",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "figure.facecolor": "white",
}

MARKET_PLOT_NAMES = {
    "ISNE": "NE", "NYIS": "NY", "PJM": "PJM", "MISO": "MISO", "SWPP": "SPP",
}


# =============================================================================
# CASE DISCOVERY / PATH HELPERS
# =============================================================================


@dataclass(frozen=True)
class SimCase:
    exp_dir: Path
    exp_name: str
    year: int
    nn: int
    trans_mode: str  # "pct" or "MW"
    trans_value: float

    @property
    def trans_label(self) -> str:
        value = format_number_for_name(self.trans_value)
        return f"MW_{value}" if self.trans_mode == "MW" else f"pct_{value}"

    @property
    def case_id(self) -> str:
        return f"y{self.year}_NN{self.nn}_{self.trans_label}"


def nn_results_dir(nn: int) -> Path:
    return MODEL_RESULTS_ROOT


def data_allocation_dir(nn: int) -> Path:
    return nn_results_dir(nn) / "data_allocation"


def ensure_output_dirs(nn: int, year: int) -> Dict[str, Path]:
    """Create only the figure directory; supporting results stay in memory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {"root": OUTPUT_DIR, "plots": OUTPUT_DIR}



def format_number_for_name(x) -> str:
    try:
        value = float(x)
        if value.is_integer():
            return str(int(value))
        return str(value).replace(".", "p")
    except Exception:
        return str(x)


def discover_cases() -> List[SimCase]:
    """Resolve the packaged best-model results without parsing folder names."""
    if not SIM_LMP_ROOT.is_dir():
        raise FileNotFoundError(f"Missing simulated LMP directory: {SIM_LMP_ROOT}")
    if SIM_YEAR not in YEARS or SIM_NN not in NN_VALUES:
        raise ValueError("Best-model case must be included in YEARS and NN_VALUES")
    return [SimCase(
        exp_dir=SIM_LMP_ROOT,
        exp_name=SIM_LMP_ROOT.name,
        year=SIM_YEAR,
        nn=SIM_NN,
        trans_mode=SIM_TRANS_MODE,
        trans_value=SIM_TRANS_VALUE,
    )]


def first_existing_path(candidates: Iterable[Path], label: str) -> Path:
    checked = []
    for p in candidates:
        checked.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {label}. Checked:\n" + "\n".join(checked))


def find_bus_to_ba_path() -> Path:
    candidates = [
        INTERCONNECTION_DIR / "nodes_to_BA_state_original.csv",
        INTERCONNECTION_DIR / "nodes_to_BA_state.csv",
    ]
    return first_existing_path(candidates, "bus-to-BA mapping file")


# =============================================================================
# READ INPUTS
# =============================================================================


def calculate_year_specific_market_weights() -> pd.DataFrame:
    """Calculate each market's annual-load share within the five-market total."""
    rows: List[Dict[str, object]] = []

    for year in YEARS:
        source = BA_LOAD_ROOT / f"EIA_historical_BA_load_{year}.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing BA-load file for {year}: {source}")

        raw = pd.read_csv(source)
        missing_markets = [market for market in MARKET_ORDER if market not in raw.columns]
        if missing_markets:
            raise ValueError(
                f"{source} is missing required BA columns: {', '.join(missing_markets)}"
            )

        loads = raw[MARKET_ORDER].apply(pd.to_numeric, errors="coerce")
        missing_counts = loads.isna().sum()
        if int(missing_counts.sum()) > 0:
            details = ", ".join(
                f"{market}={int(missing_counts[market])}"
                for market in MARKET_ORDER
                if int(missing_counts[market]) > 0
            )
            raise ValueError(f"Non-numeric or missing BA-load values in {source}: {details}")

        annual_loads = loads.sum(axis=0)
        five_ba_total = float(annual_loads.sum())
        if not np.isfinite(five_ba_total) or five_ba_total <= 0:
            raise ValueError(f"Invalid five-BA annual load total in {source}: {five_ba_total}")

        for market in MARKET_ORDER:
            annual_load = float(annual_loads[market])
            rows.append(
                {
                    "Year": year,
                    "Market": market,
                    "Source_File": str(source),
                    "Hourly_Row_Count": int(len(loads)),
                    "Missing_Value_Count": int(missing_counts[market]),
                    "Nonpositive_Value_Count": int((loads[market] <= 0).sum()),
                    "Annual_Load_MWh": annual_load,
                    "Five_BA_Total_Annual_Load_MWh": five_ba_total,
                    "Year_Specific_Weight": annual_load / five_ba_total,
                }
            )

    weights = pd.DataFrame(rows)
    weight_sums = weights.groupby("Year")["Year_Specific_Weight"].sum()
    invalid_years = weight_sums.loc[~np.isclose(weight_sums, 1.0, atol=1e-10)]
    if not invalid_years.empty:
        raise ValueError(f"Year-specific BA weights do not sum to 1:\n{invalid_years}")
    return weights


def normalize_day_index(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Return a 365-row Day/Daily_DA_LMP table; drop/reindex leap day if needed."""
    required = {"Day", "Daily_DA_LMP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    out = df[["Day", "Daily_DA_LMP"]].copy()
    out["Day"] = pd.to_numeric(out["Day"], errors="coerce")
    out["Daily_DA_LMP"] = pd.to_numeric(out["Daily_DA_LMP"], errors="coerce")
    out = out.dropna(subset=["Day"])
    out["Day"] = out["Day"].astype(int)

    # If a future daily file is accidentally saved as 1..366 including Feb. 29,
    # drop Feb. 29 (day 60) and shift subsequent days down by one.
    unique_days = sorted(out["Day"].dropna().unique())
    if len(unique_days) == 366 and min(unique_days) == 1 and max(unique_days) == 366:
        out = out.loc[out["Day"] != 60].copy()
        out.loc[out["Day"] > 60, "Day"] -= 1

    out = out.loc[(out["Day"] >= 1) & (out["Day"] <= 365)].copy()
    out = out.groupby("Day", as_index=False)["Daily_DA_LMP"].mean()

    expected = pd.DataFrame({"Day": range(1, 366)})
    out = expected.merge(out, on="Day", how="left")
    return out


def read_historical_daily_lmps(year: int) -> Dict[str, pd.DataFrame]:
    hist: Dict[str, pd.DataFrame] = {}
    missing_paths = []
    for market in MARKET_ORDER:
        template = HISTORICAL_FILES[market]
        path = Path(str(template).format(year=year))
        if not path.exists():
            missing_paths.append(f"{market}: {path}")
            continue
        hist[market] = normalize_day_index(pd.read_csv(path), path)

    if missing_paths:
        message = (
            f"Missing prepared historical daily DA LMP file(s) for {year}:\n  "
            + "\n  ".join(missing_paths)
        )
        if STRICT_HISTORICAL_FILES:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}")
    if not hist:
        raise FileNotFoundError(f"No prepared historical daily DA LMP files were found for {year}.")
    return hist


def read_bus_to_ba() -> pd.DataFrame:
    path = find_bus_to_ba_path()
    df = pd.read_csv(path)

    # Accept common alternatives but standardize to Number/NAME.
    if "Number" not in df.columns:
        alternatives = ["bus_i", "Bus", "BusNum", "bus"]
        for col in alternatives:
            if col in df.columns:
                df = df.rename(columns={col: "Number"})
                break
    if "NAME" not in df.columns:
        alternatives = ["Name", "BA", "Balancing Authority", "BA_Name"]
        for col in alternatives:
            if col in df.columns:
                df = df.rename(columns={col: "NAME"})
                break

    required = {"Number", "NAME"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required bus-to-BA columns: {sorted(missing)}")

    out = df[["Number", "NAME"]].copy()
    out["Number"] = pd.to_numeric(out["Number"], errors="coerce")
    out = out.dropna(subset=["Number"])
    out["Number"] = out["Number"].astype(int)
    out["NAME"] = out["NAME"].astype(str)
    out["Market"] = out["NAME"].map(BA_TO_MARKET)
    out = out.dropna(subset=["Market"])
    out = out.drop_duplicates(subset=["Number"])

    if out.empty:
        raise ValueError(
            f"No buses in {path} matched the target BA names. Check BA_TO_MARKET."
        )

    print(f"Using bus-to-BA mapping: {path}")
    print("Selected bus counts by market:")
    print(out["Market"].value_counts().reindex(MARKET_ORDER).fillna(0).astype(int).to_string())
    return out


def read_duals_for_case(case: SimCase) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    log = {
        "year": case.year,
        "exp_folder": case.exp_name,
        "nn": case.nn,
        "trans_mode": case.trans_mode,
        "trans_value": case.trans_value,
        "status": "ok",
        "message": "",
        "n_dual_rows_raw": 0,
        "n_invalid_dual_rows": 0,
        "n_dual_rows_used": 0,
    }

    pieces = []
    missing_files = []
    for win in range(4):
        path = case.exp_dir / f"duals_win{win}.csv"
        if not path.exists():
            missing_files.append(str(path))
            continue
        df = pd.read_csv(path)
        expected = {"Bus", "Time", "Value"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        pieces.append(df[["Bus", "Time", "Value"]].copy())

    if missing_files:
        msg = "Missing dual files: " + "; ".join(missing_files)
        if SKIP_INCOMPLETE_CASES:
            log["status"] = "skipped"
            log["message"] = msg
            return None, log
        raise FileNotFoundError(msg)

    duals = pd.concat(pieces, ignore_index=True)
    log["n_dual_rows_raw"] = len(duals)

    duals["Time"] = pd.to_numeric(duals["Time"], errors="coerce")
    duals["Value"] = pd.to_numeric(duals["Value"], errors="coerce") * SIM_LMP_SIGN
    duals = duals.dropna(subset=["Bus", "Time", "Value"])
    duals["Time"] = duals["Time"].astype(int)
    duals = duals.loc[(duals["Time"] >= 1) & (duals["Time"] <= 8760)].copy()

    invalid_mask = duals["Value"].isin(INVALID_DUAL_VALUES)
    log["n_invalid_dual_rows"] = int(invalid_mask.sum())
    if DROP_INVALID_DUALS:
        duals = duals.loc[~invalid_mask].copy()

    # Duplicates should not normally exist, but average them if they do.
    duals = duals.groupby(["Bus", "Time"], as_index=False)["Value"].mean()
    log["n_dual_rows_used"] = len(duals)
    return duals, log


def read_nodal_load(nn: int, year: int) -> pd.DataFrame:
    path = data_allocation_dir(nn) / f"nodal_load_{nn}_y_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing nodal load file: {path}")

    load = pd.read_csv(path)
    if len(load) != 8760:
        raise ValueError(f"{path} has {len(load)} rows; expected 8760.")

    # Drop accidental index/time columns if present.
    drop_cols = []
    for col in load.columns:
        lower = str(col).lower()
        if lower.startswith("unnamed") or lower in {"time", "hour", "hour_of_year"}:
            drop_cols.append(col)
    if drop_cols:
        load = load.drop(columns=drop_cols)

    load = load.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    load.insert(0, "Time", np.arange(1, len(load) + 1))
    return load


# =============================================================================
# SIMULATED LMP CALCULATION
# =============================================================================


def bus_label_to_number(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    # Handles labels like bus_123, bus_123.0, or 123.
    extracted = text.str.extract(r"(?:bus_)?(-?\d+(?:\.\d+)?)", expand=False)
    nums = pd.to_numeric(extracted, errors="coerce")
    return nums.round().astype("Int64")


def calculate_daily_weighted_sim_lmps(
    duals: pd.DataFrame,
    nodal_load: pd.DataFrame,
    bus_to_ba: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily simulated LMPs by market and hourly QA coverage."""
    load_long = nodal_load.melt(id_vars="Time", var_name="Bus", value_name="Load")
    load_long["Bus_Number"] = bus_label_to_number(load_long["Bus"])
    load_long = load_long.dropna(subset=["Bus_Number"])
    load_long["Bus_Number"] = load_long["Bus_Number"].astype(int)

    duals2 = duals.copy()
    duals2["Bus_Number"] = bus_label_to_number(duals2["Bus"])
    duals2 = duals2.dropna(subset=["Bus_Number"])
    duals2["Bus_Number"] = duals2["Bus_Number"].astype(int)

    merged = duals2.merge(
        load_long[["Time", "Bus_Number", "Load"]],
        on=["Time", "Bus_Number"],
        how="inner",
    )
    merged = merged.merge(
        bus_to_ba[["Number", "Market"]],
        left_on="Bus_Number",
        right_on="Number",
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No dual/load/bus-to-BA rows matched. Check bus labels and the bus-to-BA mapping file."
        )

    merged["Load"] = pd.to_numeric(merged["Load"], errors="coerce").fillna(0.0)
    merged["Load_x_LMP"] = merged["Load"] * merged["Value"]

    grouped = (
        merged.groupby(["Market", "Time"], as_index=False)
        .agg(
            Load_x_LMP=("Load_x_LMP", "sum"),
            Load=("Load", "sum"),
            n_buses_used=("Bus_Number", "nunique"),
        )
    )
    grouped["Simulated_LMP"] = grouped["Load_x_LMP"] / grouped["Load"].replace(0, np.nan)
    grouped["Day"] = ((grouped["Time"] - 1) // 24 + 1).astype(int)

    daily = (
        grouped.groupby(["Market", "Day"], as_index=False)
        .agg(
            Simulated_LMP=("Simulated_LMP", "mean"),
            Sim_Load_MW=("Load", "mean"),
            Hours_Available=("Simulated_LMP", lambda x: int(x.notna().sum())),
            Mean_Buses_Used=("n_buses_used", "mean"),
        )
    )

    expected = pd.MultiIndex.from_product([MARKET_ORDER, range(1, 366)], names=["Market", "Day"]).to_frame(index=False)
    daily = expected.merge(daily, on=["Market", "Day"], how="left")
    return daily, grouped


# =============================================================================
# METRICS / PLOTS
# =============================================================================


def regression_r2(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr * corr)


def r2_one_to_one(observed: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[mask]
    predicted = predicted[mask]
    if len(observed) < 2:
        return np.nan
    denom = np.sum((observed - np.mean(observed)) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((observed - predicted) ** 2) / denom)


def calculate_metrics_for_pair(hist: pd.Series, sim: pd.Series) -> Dict[str, float]:
    h = pd.to_numeric(hist, errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(sim, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(h) & np.isfinite(s)
    if mask.sum() == 0:
        return {
            "n_days": 0,
            "R2_regression": np.nan,
            "R2_1to1": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "Mean_Historical_DA_LMP": np.nan,
            "Mean_Simulated_LMP": np.nan,
        }

    diff = s[mask] - h[mask]
    return {
        "n_days": int(mask.sum()),
        "R2_regression": regression_r2(h, s),
        "R2_1to1": r2_one_to_one(h, s),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "MAE": float(np.mean(np.abs(diff))),
        "Bias": float(np.mean(diff)),
        "Mean_Historical_DA_LMP": float(np.mean(h[mask])),
        "Mean_Simulated_LMP": float(np.mean(s[mask])),
    }


@plt.rc_context(PLOT_STYLE)
def save_market_plot(
    comparison: pd.DataFrame,
    case: SimCase,
    market: str,
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(comparison["Day"], comparison["Historical_DA_LMP"],
            color="black", linewidth=PLOT_LINEWIDTH)
    ax.plot(comparison["Day"], comparison["Simulated_LMP"],
            color="crimson", linewidth=PLOT_LINEWIDTH)
    month_ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=PLOT_MONTH_FONTSIZE)
    ax.margins(x=0.05)
    ax.set_ylabel("LMP ($/MWh)", labelpad=PLOT_AXIS_LABEL_PAD,
                  fontsize=PLOT_AXIS_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Date", labelpad=PLOT_AXIS_LABEL_PAD,
                  fontsize=PLOT_AXIS_FONTSIZE, fontweight="bold")
    mode_label = "+MW" if case.trans_mode == "MW" else "percentage"
    trans_value = format_number_for_name(case.trans_value)
    ax.set_title(
        f"Model: {FUEL_TAG}, {case.nn}, {trans_value} ({mode_label})\n"
        f"{MARKET_PLOT_NAMES.get(market, market)}\n",
        fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold",
    )
    ax.tick_params(axis="both", which="both", pad=PLOT_TICK_PAD)
    ax.tick_params(axis="y", labelsize=PLOT_YTICK_FONTSIZE)
    ax.annotate(f"$R^2$: {metrics['R2_regression']:.3f}",
                xy=(0.7, 0.9), xycoords="axes fraction", fontsize=PLOT_METRIC_FONTSIZE)
    ax.annotate(f"RMSE: {metrics['RMSE']:.1f}",
                xy=(0.7, 0.85), xycoords="axes fraction", fontsize=PLOT_METRIC_FONTSIZE)
    # Match the reference margins; omit legends, including the NE figure.
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.88)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)






def calculate_weighted_case_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if "Year_Specific_Weight" not in metrics_df.columns:
        raise ValueError("metrics_df is missing Year_Specific_Weight")

    rows = []
    group_cols = ["Year", "Exp_Folder", "NN", "Trans_Mode", "Trans_Value"]
    for keys, group in metrics_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        available = group.dropna(subset=["Market"])
        weight_sum = 0.0
        for metric in ["R2_regression", "R2_1to1", "RMSE", "MAE", "Bias"]:
            total = 0.0
            metric_weight_sum = 0.0
            for _, g in available.iterrows():
                value = g[metric]
                w = g["Year_Specific_Weight"]
                if pd.notna(value) and pd.notna(w):
                    total += float(w) * float(value)
                    metric_weight_sum += float(w)
            row[f"Weighted_{metric}"] = total / metric_weight_sum if metric_weight_sum > 0 else np.nan
            weight_sum = max(weight_sum, metric_weight_sum)
        row["Market_Weight_Coverage"] = weight_sum
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if "Weighted_R2_regression" in out.columns:
        out["Rank_R2_regression"] = out["Weighted_R2_regression"].rank(method="min", ascending=False)
    if "Weighted_RMSE" in out.columns:
        out["Rank_RMSE"] = out["Weighted_RMSE"].rank(method="min", ascending=True)
    if {"Rank_R2_regression", "Rank_RMSE"}.issubset(out.columns):
        out["Average_Rank_R2_RMSE"] = (out["Rank_R2_regression"] + out["Rank_RMSE"]) / 2
        out["Rank_Average_R2_RMSE"] = out["Average_Rank_R2_RMSE"].rank(method="min", ascending=True)

    out = out.sort_values(["Year", "Rank_Average_R2_RMSE", "Rank_RMSE", "Rank_R2_regression"])
    return out


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def main() -> None:
    year_specific_weights = calculate_year_specific_market_weights()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Year-specific BA-load weights (not saved):")
    print(
        year_specific_weights.pivot(
            index="Year", columns="Market", values="Year_Specific_Weight"
        ).reindex(columns=MARKET_ORDER).to_string(float_format=lambda value: f"{value:.6f}")
    )

    bus_to_ba = read_bus_to_ba()
    cases = discover_cases()

    # Create the requested NN/year structure up front, even if a later input is
    # unavailable. This also makes future NN additions predictable.
    for nn in NN_VALUES:
        if nn_results_dir(nn).exists():
            for year in YEARS:
                ensure_output_dirs(nn, year)

    print("\nDiscovered simulation cases:")
    for c in cases:
        print(f"  {c.exp_name}")

    hist_by_year = {year: read_historical_daily_lmps(year) for year in YEARS}
    load_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

    metric_rows: List[Dict[str, object]] = []
    run_logs: List[Dict[str, object]] = []

    for case in cases:
        out_dirs = ensure_output_dirs(case.nn, case.year)
        print(f"\n=== Processing {case.exp_name} ===")
        duals, log = read_duals_for_case(case)
        if duals is None:
            run_logs.append(log)
            print(f"Skipped {case.exp_name}: {log['message']}")
            continue

        hist = hist_by_year[case.year]
        missing_hist_markets = [market for market in MARKET_ORDER if market not in hist]
        if missing_hist_markets:
            log["status"] = "ok_partial_historical"
            log["message"] = "Missing prepared historical markets: " + ", ".join(missing_hist_markets)
        run_logs.append(log)

        load_key = (case.nn, case.year)
        if load_key not in load_cache:
            load_cache[load_key] = read_nodal_load(case.nn, case.year)

        daily_sim, hourly_qa = calculate_daily_weighted_sim_lmps(
            duals=duals,
            nodal_load=load_cache[load_key],
            bus_to_ba=bus_to_ba,
        )

        for market in [m for m in MARKET_ORDER if m in hist]:
            hist_df = hist[market].rename(columns={"Daily_DA_LMP": "Historical_DA_LMP"})
            sim_df = daily_sim.loc[daily_sim["Market"] == market, ["Day", "Simulated_LMP", "Sim_Load_MW", "Hours_Available", "Mean_Buses_Used"]]
            comparison = hist_df.merge(sim_df, on="Day", how="left")
            comparison.insert(0, "Market", market)
            comparison.insert(0, "Exp_Folder", case.exp_name)
            comparison.insert(0, "Year", case.year)

            metrics = calculate_metrics_for_pair(comparison["Historical_DA_LMP"], comparison["Simulated_LMP"])
            plot_name = f"daily_lmp_comparison_{case.case_id}_{market}.png"
            combined_plot_name = f"combined_daily_lmp_comparison_{case.case_id}.png"
            metric_row = {
                "Status": "DONE",
                "Model_Results_Base": str(nn_results_dir(case.nn)),
                "Year": case.year,
                "Exp_Folder": case.exp_name,
                "Scenario_Path": str(case.exp_dir),
                "NN": case.nn,
                "Trans_Mode": case.trans_mode,
                "Trans_Value": case.trans_value,
                "Market": market,
                "Plot_File": plot_name,
                "Combined_Plot_File": combined_plot_name,
                **metrics,
            }
            metric_rows.append(metric_row)


            save_market_plot(
                comparison=comparison,
                case=case,
                market=market,
                metrics=metrics,
                out_path=out_dirs["plots"] / plot_name,
            )


        print(
            f"Finished {case.exp_name}: "
            f"raw dual rows={log['n_dual_rows_raw']}, "
            f"invalid dual rows={log['n_invalid_dual_rows']}, "
            f"used dual rows={log['n_dual_rows_used']}"
        )

    metrics_df = pd.DataFrame(metric_rows)
    run_log_df = pd.DataFrame(run_logs)

    if metrics_df.empty:
        raise RuntimeError("No completed cases produced metrics.")

    weight_columns = [
        "Year", "Market", "Annual_Load_MWh", "Five_BA_Total_Annual_Load_MWh",
        "Year_Specific_Weight",
    ]
    metrics_df = metrics_df.merge(
        year_specific_weights[weight_columns],
        on=["Year", "Market"],
        how="left",
        validate="many_to_one",
    )
    if metrics_df["Year_Specific_Weight"].isna().any():
        missing = metrics_df.loc[
            metrics_df["Year_Specific_Weight"].isna(), ["Year", "Market"]
        ].drop_duplicates()
        raise ValueError(
            "Missing year-specific BA weights for metric rows:\n"
            + missing.to_string(index=False)
        )

    metrics_df = metrics_df.sort_values(["Year", "NN", "Trans_Mode", "Trans_Value", "Market"])
    all_weighted_frames = []

    # Calculate summary statistics for console reporting only.
    for nn in NN_VALUES:
        for year in YEARS:
            out_dirs = ensure_output_dirs(nn, year)
            year_metrics = metrics_df.loc[
                (metrics_df["NN"] == nn) & (metrics_df["Year"] == year)
            ].copy()
            year_logs = run_log_df.loc[
                (run_log_df["nn"] == nn) & (run_log_df["year"] == year)
            ].copy() if not run_log_df.empty else run_log_df.copy()

            if year_metrics.empty:
                print(f"[WARN] No metrics to export for NN={nn}, year={year}.")
                continue

            metrics_name = f"PJM_LMP_validation_metrics_{FUEL_TAG}_NN{nn}_{year}.csv"

            weighted_df = calculate_weighted_case_metrics(year_metrics)
            all_weighted_frames.append(weighted_df)


    print("\nValidation complete.")
    print(f"Outputs saved under: {OUTPUT_DIR}")
    print("\nTop cases by average R2/RMSE rank:")
    weighted_df = pd.concat(all_weighted_frames, ignore_index=True)
    weighted_df = weighted_df.sort_values(
        ["Year", "Rank_Average_R2_RMSE", "Rank_RMSE", "Rank_R2_regression"]
    )
    show_cols = [
        "Year", "Exp_Folder", "Weighted_R2_regression", "Weighted_RMSE",
        "Rank_R2_regression", "Rank_RMSE", "Rank_Average_R2_RMSE",
    ]
    print(weighted_df[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

