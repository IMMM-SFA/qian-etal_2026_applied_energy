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
SECTION_DIR = PACKAGE_ROOT / "shared"
import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIG (adjust paths only)
# =============================================================================

# Daily system table exported by:
#   the upstream daily load, wind, solar, OFV aggregation script
DAILY_AGG = "sum"  # "sum" or "mean"
DAILY_EIC_FILE = PACKAGE_ROOT / "shared" / "upstream_processed" / f"daily_EIC_load_wind_solar_objective_{DAILY_AGG}.csv"

# TGW weather directory that contains:
#   results/daily_avg_CDD_cleaned.csv
#   results/daily_avg_HDD_cleaned.csv
#   results/daily_avg_metvars_cleaned.csv
#   results/above_avg_coverage.csv
#   results/below_avg_coverage.csv
WEATHER_DIR = PACKAGE_ROOT / "shared" / "upstream_processed"

DAILY_METVARS_FILE = os.path.join(WEATHER_DIR, "TGW_results", "daily_avg_metvars_cleaned.csv")

# NEW: hourly coverage fraction files
ABOVE_AVG_COVERAGE_HOURLY = os.path.join(WEATHER_DIR, "TGW_results", "above_avg_coverage.csv")
BELOW_AVG_COVERAGE_HOURLY = os.path.join(WEATHER_DIR, "TGW_results", "below_avg_coverage.csv")

# Output folder for Paper 1
OUT_DIR = REPRODUCED_ROOT / "shared"
OUT_FILE = os.path.join(OUT_DIR, "system_daily_table.csv")


# =============================================================================
# Helpers
# =============================================================================

def _ensure_year_day(df_in: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Accept either (Year, Day) or Date; return with Year, Day present.
    Day is day-of-year (1..365), consistent with your system table.
    """
    df2 = df_in.copy()
    if {"Year", "Day"}.issubset(df2.columns):
        df2["Year"] = df2["Year"].astype(int)
        df2["Day"] = df2["Day"].astype(int)
        return df2

    if "Date" in df2.columns:
        dt = pd.to_datetime(df2["Date"], errors="coerce")
        df2["Year"] = dt.dt.year.astype(int)
        df2["Day"] = dt.dt.dayofyear.astype(int)
        return df2

    raise ValueError(f"{label} must contain either (Year, Day) or Date.")


def _read_daily_eic_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing DAILY_EIC_FILE: {path}")

    df = pd.read_csv(path)

    # Normalize date + remove Feb 29 if any
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[~((df["Date"].dt.month == 2) & (df["Date"].dt.day == 29))].copy()
        df["Year"] = df["Date"].dt.year.astype(int)
        df["Day"] = df["Date"].dt.dayofyear.astype(int)
    elif not {"Year", "Day"}.issubset(df.columns):
        raise ValueError("Daily EIC table must have Date or (Year, Day).")

    df["Year"] = df["Year"].astype(int)
    df["Day"] = df["Day"].astype(int)

    req = ["Load", "Wind", "Solar", "ObjectiveValue"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Daily EIC table missing required columns: {missing}")

    # Compute fractions if not present (guard divide-by-zero)
    if "WindFrac" not in df.columns:
        df["WindFrac"] = np.where(df["Load"] > 0, df["Wind"] / df["Load"], np.nan)
    if "SolarFrac" not in df.columns:
        df["SolarFrac"] = np.where(df["Load"] > 0, df["Solar"] / df["Load"], np.nan)

    for c in ["Load", "Wind", "Solar", "ObjectiveValue", "WindFrac", "SolarFrac"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _merge_cdd_hdd(df: pd.DataFrame, weather_dir: str) -> pd.DataFrame:
    hdd_path = os.path.join(weather_dir, "TGW_results", "daily_avg_HDD_cleaned.csv")
    cdd_path = os.path.join(weather_dir, "TGW_results", "daily_avg_CDD_cleaned.csv")

    if not os.path.exists(hdd_path):
        raise FileNotFoundError(f"Missing: {hdd_path}")
    if not os.path.exists(cdd_path):
        raise FileNotFoundError(f"Missing: {cdd_path}")

    daily_hdd = pd.read_csv(hdd_path)
    daily_cdd = pd.read_csv(cdd_path)

    # Weather Day is a no-leap index; electricity Day is calendar day-of-year.
    # Join actual dates so March--December in leap years cannot shift by one day.
    out = _merge_weather_by_date(df, daily_cdd, ["CDD"])
    out = _merge_weather_by_date(out, daily_hdd, ["HDD"])

    out["CDD"] = pd.to_numeric(out["CDD"], errors="coerce")
    out["HDD"] = pd.to_numeric(out["HDD"], errors="coerce")
    return out


def _merge_weather_by_date(df: pd.DataFrame, weather: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    left = df.copy()
    right = weather.copy()
    for frame in (left, right):
        if "Date" not in frame:
            raise ValueError("An actual Date column is required for weather alignment.")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
        if frame["Date"].isna().any() or frame["Date"].duplicated().any():
            raise ValueError("Weather alignment requires unique, nonmissing dates.")
    out = left.merge(right[["Date", *columns]], on="Date", how="left", validate="one_to_one")
    if out[columns].isna().any().any():
        raise ValueError(f"Missing date-matched weather values: {columns}")
    return out


def _merge_metvars(df: pd.DataFrame, metvars_file: str) -> pd.DataFrame:
    if not os.path.exists(metvars_file):
        raise FileNotFoundError(f"Missing DAILY_METVARS_FILE: {metvars_file}")

    met = pd.read_csv(metvars_file)
    met = _ensure_year_day(met, "daily_avg_metvars_cleaned.csv")

    if "Date" in met.columns:
        met["Date"] = pd.to_datetime(met["Date"], errors="coerce")
        met = met[~((met["Date"].dt.month == 2) & (met["Date"].dt.day == 29))].copy()

    keep = ["Year", "Day", "Q2", "T2", "U10", "V10", "SWDOWN", "GLW"]
    missing = [c for c in keep if c not in met.columns]
    if missing:
        raise ValueError(f"Metvars file missing required columns: {missing}")

    for c in ["Q2", "T2", "U10", "V10", "SWDOWN", "GLW"]:
        met[c] = pd.to_numeric(met[c], errors="coerce")

    out = _merge_weather_by_date(df, met, ["Q2", "T2", "U10", "V10", "SWDOWN", "GLW"])
    return out


def _hourly_coverage_to_daily(path: str, out_col: str) -> pd.DataFrame:
    """
    Convert hourly coverage-fraction table to daily mean.

    Expected format (like your attached examples):
      columns: DateTime, 1980, 1981, ..., 2019
      DateTime values like "01-01 00:00" (no year)
      rows: 8760 (Feb 29 already removed)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing coverage file: {path}")

    cov = pd.read_csv(path)
    if "DateTime" not in cov.columns:
        raise ValueError(f"Coverage file missing DateTime column: {path}")

    year_cols = [c for c in cov.columns if c != "DateTime"]
    if not year_cols:
        raise ValueError(f"No year columns found in coverage file: {path}")

    parts = []
    # Loop years to avoid huge melt memory spikes
    for y in year_cols:
        # coerce year label to int
        try:
            year_int = int(y)
        except Exception:
            continue

        tmp = cov[["DateTime", y]].copy()
        tmp.rename(columns={y: out_col}, inplace=True)
        tmp[out_col] = pd.to_numeric(tmp[out_col], errors="coerce")

        # Build datetime by attaching year
        # DateTime string is like "01-01 00:00"
        tmp["DateTimeFull"] = pd.to_datetime(
            tmp["DateTime"].astype(str).apply(lambda s: f"{year_int}-{s}"),
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )

        # Drop Feb 29 defensively (should already be removed)
        tmp = tmp[~((tmp["DateTimeFull"].dt.month == 2) & (tmp["DateTimeFull"].dt.day == 29))].copy()

        tmp["Date"] = tmp["DateTimeFull"].dt.floor("D")
        daily = tmp.groupby("Date")[out_col].mean().reset_index()
        daily["Year"] = daily["Date"].dt.year.astype(int)
        daily["Day"] = daily["Date"].dt.dayofyear.astype(int)

        parts.append(daily[["Year", "Day", out_col]])

    if not parts:
        raise RuntimeError(f"No valid year columns parsed from: {path}")

    out = pd.concat(parts, ignore_index=True)
    # Ensure 365 per year if everything lines up
    return out


def _merge_temp_coverage(df: pd.DataFrame, above_path: str, below_path: str) -> pd.DataFrame:
    daily_above = _hourly_coverage_to_daily(above_path, "above_avg")
    daily_below = _hourly_coverage_to_daily(below_path, "below_avg")

    out = (df.merge(daily_above, on=["Year", "Day"], how="left")
             .merge(daily_below, on=["Year", "Day"], how="left"))

    out["above_avg"] = pd.to_numeric(out["above_avg"], errors="coerce")
    out["below_avg"] = pd.to_numeric(out["below_avg"], errors="coerce")
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("[VERSION] This upstream builder reproduces the date-aligned revision. It differs from the preserved reported-analysis table; see PROVENANCE_AND_ISSUES.md.")
    os.makedirs(OUT_DIR, exist_ok=True)

    df_daily = _read_daily_eic_table(DAILY_EIC_FILE)
    df_daily = _merge_cdd_hdd(df_daily, WEATHER_DIR)
    df_daily = _merge_metvars(df_daily, DAILY_METVARS_FILE)

    # NEW: merge daily coverage fractions
    df_daily = _merge_temp_coverage(df_daily, ABOVE_AVG_COVERAGE_HOURLY, BELOW_AVG_COVERAGE_HOURLY)

    # Final housekeeping: sort and ensure no Feb 29
    if "Date" in df_daily.columns:
        df_daily["Date"] = pd.to_datetime(df_daily["Date"], errors="coerce")
        df_daily = df_daily[~((df_daily["Date"].dt.month == 2) & (df_daily["Date"].dt.day == 29))].copy()
        df_daily = df_daily.sort_values(["Year", "Date"])
    else:
        df_daily = df_daily.sort_values(["Year", "Day"])

    df_daily.to_csv(OUT_FILE, index=False)
    print(f"[SAVE] {OUT_FILE}")
    print(f"[READ]  {DAILY_EIC_FILE}")
    print(f"[MERGE] {os.path.join(WEATHER_DIR, 'TGW_results', 'daily_avg_CDD_cleaned.csv')}")
    print(f"[MERGE] {os.path.join(WEATHER_DIR, 'TGW_results', 'daily_avg_HDD_cleaned.csv')}")
    print(f"[MERGE] {DAILY_METVARS_FILE}")
    print(f"[MERGE] {ABOVE_AVG_COVERAGE_HOURLY}")
    print(f"[MERGE] {BELOW_AVG_COVERAGE_HOURLY}")


if __name__ == "__main__":
    main()
