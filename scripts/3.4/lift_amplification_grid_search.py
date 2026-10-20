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
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIG
# =============================================================================

IN_FILE = PACKAGE_ROOT / "shared" / "data" / "system_daily_table.csv"
OUT_DIR = REPRODUCED_ROOT / "3.4" / "statistics"

HOT_PANELS = [
    {"alpha": 97.0, "temp_tau": 11.7},
    {"alpha": 99.0, "temp_tau": 12.7},
    {"alpha": 99.5, "temp_tau": 13.1},
]
COLD_PANELS = [
    {"alpha": 97.0, "temp_tau": 39.0},
    {"alpha": 99.0, "temp_tau": 43.7},
    {"alpha": 99.5, "temp_tau": 47.6},
]

# Percentile grid of low raw-production cutoffs.
PCTS = [0, *list(range(5, 100, 5)), 100]

N_BOOT = 500
RANDOM_SEED = 20260122
MIN_SUPPORT_DAYS = 10

# Reference pools used to calculate raw Wind/Solar percentile thresholds.
# "regime": all days in the regime
# "min_tau": days at or above the smallest panel temperature threshold
# "custom": days at or above the corresponding custom threshold
HOT_REF_MODE = "regime"
COLD_REF_MODE = "regime"
HOT_CUSTOM_TAU = 11.7
COLD_CUSTOM_TAU = 39.0

WIND_COL = "Wind"
SOLAR_COL = "Solar"
RESOURCE_BASIS = "raw_production"


# =============================================================================
# I/O and reference-pool helpers
# =============================================================================

def _read_system_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "Year", "Day", "ObjectiveValue", "Load", WIND_COL, SOLAR_COL, "CDD", "HDD"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"system_daily_table missing columns: {missing}")

    for column in ["ObjectiveValue", "Load", WIND_COL, SOLAR_COL, "CDD", "HDD"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Day"] = pd.to_numeric(df["Day"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year", "Day", "ObjectiveValue"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Day"] = df["Day"].astype(int)
    return df


def _thresholds_from_percentiles(x: np.ndarray, pcts: List[int]) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(len(pcts), np.nan)

    thresholds = np.empty(len(pcts), dtype=float)
    for i, pct in enumerate(pcts):
        if pct <= 0:
            thresholds[i] = float(np.min(x))
        elif pct >= 100:
            thresholds[i] = float(np.max(x))
        else:
            thresholds[i] = float(np.quantile(x, pct / 100.0))
    return thresholds


def _make_ref_pool(
    df_regime: pd.DataFrame,
    temp_col: str,
    mode: str,
    panels: List[Dict],
    custom_tau: float,
) -> pd.DataFrame:
    mode = (mode or "").lower().strip()
    if mode == "regime":
        return df_regime
    if mode == "min_tau":
        min_tau = float(min(panel["temp_tau"] for panel in panels))
        return df_regime[df_regime[temp_col] >= min_tau]
    if mode == "custom":
        return df_regime[df_regime[temp_col] >= float(custom_tau)]
    raise ValueError(f"Unknown reference-pool mode: {mode}")


# =============================================================================
# Core probability and lift helpers (UNSMOOTHED)
# =============================================================================

def _rate(stress: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    """Return (rate, n); rate is missing with low support or a non-positive value."""
    mask = mask.astype(bool)
    n_days = int(mask.sum())
    if n_days < MIN_SUPPORT_DAYS:
        return np.nan, n_days

    rate = float(stress[mask].mean()) if n_days > 0 else np.nan
    if not np.isfinite(rate) or rate <= 0:
        return np.nan, n_days
    return rate, n_days


def _lift_from_rates(cond_rate: float, base_rate: float) -> float:
    if not np.isfinite(cond_rate) or not np.isfinite(base_rate) or base_rate <= 0:
        return np.nan
    return float(cond_rate / base_rate)


@dataclass
class PanelResult:
    family: str
    alpha: float
    temp_tau: float
    temp_col: str
    ov_thr: float
    p_stress: float
    lift_temp_mean: float
    lift_temp_ci_lo: float
    lift_temp_ci_hi: float
    wind_thr: np.ndarray
    solar_thr: np.ndarray
    lift_comp_mean: np.ndarray
    ampl_mean: np.ndarray
    dlog2lift_mean: np.ndarray
    support_frac: np.ndarray
    n_temp_mean: float
    n_cond_mean: np.ndarray


def _compute_one_panel(
    df_regime: pd.DataFrame,
    family: str,
    temp_col: str,
    alpha: float,
    temp_tau: float,
    wind_thr: np.ndarray,
    solar_thr: np.ndarray,
    rng: np.random.Generator,
) -> PanelResult:
    objective = df_regime["ObjectiveValue"].to_numpy()
    ov_thr = float(np.quantile(objective, alpha / 100.0))
    stress_full = objective >= ov_thr
    p_stress = float(stress_full.mean())

    x_temp = df_regime[temp_col].to_numpy()
    x_wind = df_regime[WIND_COL].to_numpy()
    x_solar = df_regime[SOLAR_COL].to_numpy()
    stress = stress_full.astype(bool)
    n_days = len(df_regime)

    grid_shape = (N_BOOT, len(PCTS), len(PCTS))
    lift_temp_boot = np.full(N_BOOT, np.nan, dtype=float)
    n_temp_boot = np.full(N_BOOT, np.nan, dtype=float)
    lift_comp_boot = np.full(grid_shape, np.nan, dtype=float)
    ampl_boot = np.full(grid_shape, np.nan, dtype=float)
    dlog2_boot = np.full(grid_shape, np.nan, dtype=float)
    n_cond_boot = np.full(grid_shape, np.nan, dtype=float)

    for boot_i in range(N_BOOT):
        idx = rng.integers(0, n_days, size=n_days)
        y = stress[idx]
        temp = x_temp[idx]
        wind = x_wind[idx]
        solar = x_solar[idx]

        base_rate = float(y.mean()) if y.size else np.nan
        if not np.isfinite(base_rate) or base_rate <= 0:
            continue

        temp_mask = temp >= float(temp_tau)
        temp_rate, n_temp = _rate(y, temp_mask)
        n_temp_boot[boot_i] = n_temp
        lift_temp_boot[boot_i] = _lift_from_rates(temp_rate, base_rate)

        if not np.isfinite(temp_rate) or temp_rate <= 0:
            continue

        for wind_i, wind_cutoff in enumerate(wind_thr):
            wind_mask = wind <= wind_cutoff
            for solar_i, solar_cutoff in enumerate(solar_thr):
                compound_mask = temp_mask & wind_mask & (solar <= solar_cutoff)
                comp_rate, n_cond = _rate(y, compound_mask)
                n_cond_boot[boot_i, wind_i, solar_i] = n_cond

                lift_comp_boot[boot_i, wind_i, solar_i] = _lift_from_rates(
                    comp_rate, base_rate
                )
                if np.isfinite(comp_rate) and comp_rate > 0:
                    ampl_boot[boot_i, wind_i, solar_i] = float(comp_rate / temp_rate)
                    dlog2_boot[boot_i, wind_i, solar_i] = float(
                        np.log2(comp_rate) - np.log2(temp_rate)
                    )

    return PanelResult(
        family=family,
        alpha=float(alpha),
        temp_tau=float(temp_tau),
        temp_col=temp_col,
        ov_thr=ov_thr,
        p_stress=p_stress,
        lift_temp_mean=float(np.nanmean(lift_temp_boot)),
        lift_temp_ci_lo=float(np.nanquantile(lift_temp_boot, 0.025)),
        lift_temp_ci_hi=float(np.nanquantile(lift_temp_boot, 0.975)),
        wind_thr=wind_thr,
        solar_thr=solar_thr,
        lift_comp_mean=np.nanmean(lift_comp_boot, axis=0),
        ampl_mean=np.nanmean(ampl_boot, axis=0),
        dlog2lift_mean=np.nanmean(dlog2_boot, axis=0),
        support_frac=np.mean(np.isfinite(dlog2_boot), axis=0),
        n_temp_mean=float(np.nanmean(n_temp_boot)),
        n_cond_mean=np.nanmean(n_cond_boot, axis=0),
    )


def _panel_to_long_df(result: PanelResult) -> pd.DataFrame:
    rows = []
    for wind_i, wind_pct in enumerate(PCTS):
        for solar_i, solar_pct in enumerate(PCTS):
            wind_cutoff = result.wind_thr[wind_i]
            solar_cutoff = result.solar_thr[solar_i]
            rows.append(
                {
                    "Family": result.family,
                    "Alpha": result.alpha,
                    "TempTau": result.temp_tau,
                    "TempCol": result.temp_col,
                    "OV_Threshold": result.ov_thr,
                    "StressRate_Regime": result.p_stress,
                    "ResourceBasis": RESOURCE_BASIS,
                    "WindSourceCol": WIND_COL,
                    "SolarSourceCol": SOLAR_COL,
                    "WindPctLo": int(wind_pct),
                    "SolarPctLo": int(solar_pct),
                    # Retain the threshold names expected by downstream plotting code.
                    "WindThr": float(wind_cutoff) if np.isfinite(wind_cutoff) else np.nan,
                    "SolarThr": float(solar_cutoff) if np.isfinite(solar_cutoff) else np.nan,
                    "Lift_Temp_BootMean": result.lift_temp_mean,
                    "Lift_Temp_ci_lo": result.lift_temp_ci_lo,
                    "Lift_Temp_ci_hi": result.lift_temp_ci_hi,
                    "Lift_Compound_BootMean": (
                        float(result.lift_comp_mean[wind_i, solar_i])
                        if np.isfinite(result.lift_comp_mean[wind_i, solar_i])
                        else np.nan
                    ),
                    "Ampl_BootMean": (
                        float(result.ampl_mean[wind_i, solar_i])
                        if np.isfinite(result.ampl_mean[wind_i, solar_i])
                        else np.nan
                    ),
                    "DeltaLog2Lift_BootMean": (
                        float(result.dlog2lift_mean[wind_i, solar_i])
                        if np.isfinite(result.dlog2lift_mean[wind_i, solar_i])
                        else np.nan
                    ),
                    "SupportFrac": float(result.support_frac[wind_i, solar_i]),
                    "Ntemp_BootMean": result.n_temp_mean,
                    "Ncond_BootMean": (
                        float(result.n_cond_mean[wind_i, solar_i])
                        if np.isfinite(result.n_cond_mean[wind_i, solar_i])
                        else np.nan
                    ),
                    "MinSupportDays": MIN_SUPPORT_DAYS,
                    "N_boot": N_BOOT,
                }
            )
    return pd.DataFrame(rows)


def _run_family(
    df_regime: pd.DataFrame,
    family: str,
    temp_col: str,
    panels: List[Dict],
    wind_thr: np.ndarray,
    solar_thr: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    frames = []
    for panel in panels:
        result = _compute_one_panel(
            df_regime=df_regime,
            family=family,
            temp_col=temp_col,
            alpha=float(panel["alpha"]),
            temp_tau=float(panel["temp_tau"]),
            wind_thr=wind_thr,
            solar_thr=solar_thr,
            rng=rng,
        )
        frames.append(_panel_to_long_df(result))
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = _read_system_daily(IN_FILE)
    resource_cols = [WIND_COL, SOLAR_COL]
    hot = df[df["CDD"].fillna(0) > 0].dropna(subset=resource_cols).copy()
    cold = df[df["HDD"].fillna(0) > 0].dropna(subset=resource_cols).copy()
    if hot.empty or cold.empty:
        raise ValueError(
            "HOT or COLD regime is empty after filtering; check CDD/HDD and Wind/Solar."
        )

    hot_ref = _make_ref_pool(
        hot, "CDD", HOT_REF_MODE, HOT_PANELS, HOT_CUSTOM_TAU
    ).dropna(subset=resource_cols)
    cold_ref = _make_ref_pool(
        cold, "HDD", COLD_REF_MODE, COLD_PANELS, COLD_CUSTOM_TAU
    ).dropna(subset=resource_cols)

    hot_wind_thr = _thresholds_from_percentiles(hot_ref[WIND_COL].to_numpy(float), PCTS)
    hot_solar_thr = _thresholds_from_percentiles(hot_ref[SOLAR_COL].to_numpy(float), PCTS)
    cold_wind_thr = _thresholds_from_percentiles(cold_ref[WIND_COL].to_numpy(float), PCTS)
    cold_solar_thr = _thresholds_from_percentiles(cold_ref[SOLAR_COL].to_numpy(float), PCTS)

    rng = np.random.default_rng(RANDOM_SEED)
    hot_output = _run_family(
        hot, "HOT", "CDD", HOT_PANELS, hot_wind_thr, hot_solar_thr, rng
    )
    cold_output = _run_family(
        cold, "COLD", "HDD", COLD_PANELS, cold_wind_thr, cold_solar_thr, rng
    )

    output_specs = [
        (hot_output, "lift_amplification_gridsearch_bootstrap_HOT.csv"),
        (cold_output, "lift_amplification_gridsearch_bootstrap_COLD.csv"),
    ]
    for output, filename in output_specs:
        output_path = os.path.join(OUT_DIR, filename)
        output.to_csv(output_path, index=False)
        print("[SAVE]", output_path)

    print("Done. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
