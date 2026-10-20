"""Paper-package script. Output scope is documented in PAPER_OUTPUTS.md at the package root."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
PACKAGE_ROOT = DATA_ROOT / "datasets_and_scripts"
REPRODUCED_ROOT = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve()
SECTION_DIR = PACKAGE_ROOT / "3.1_two_sample_comparisons"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = PACKAGE_ROOT
DAILY_FILE = PACKAGE_ROOT / "shared" / "data" / "system_daily_table.csv"
OUT_DIR = REPRODUCED_ROOT / "3.1" / "wind_diagnostics"

HOURLY_FILE = PACKAGE_ROOT / "3.1_two_sample_comparisons/data/system_hourly_wind_event_inputs.csv"

ALPHA = 99.5
WINDOW_BEFORE = 72
WINDOW_AFTER = 95  # event day (0--23) plus the following 72 hours
LAGS = np.arange(-72, 73)
SEED = 20260920


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def seasonal_zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize against the 40-year mean and SD for calendar hour."""
    key = frame.index.strftime("%m-%d %H")
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        x = pd.to_numeric(frame[col], errors="coerce")
        mean = x.groupby(key).transform("mean")
        sd = x.groupby(key).transform("std")
        out[col] = (x - mean) / sd.replace(0, np.nan)
    return out


def event_composite(z: pd.DataFrame, events: pd.DatetimeIndex, boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rel = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1)
    rows = []
    for col in z.columns:
        matrix = np.full((len(events), len(rel)), np.nan)
        for i, day in enumerate(events):
            wanted = day + pd.to_timedelta(rel, unit="h")
            matrix[i, :] = z[col].reindex(wanted).to_numpy(float)
        estimate = np.nanmean(matrix, axis=0)
        draws = np.empty((boot, len(rel)), dtype=float)
        for b in range(boot):
            draws[b] = np.nanmean(matrix[rng.integers(0, len(events), len(events))], axis=0)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        rows.extend(
            {"Variable": col, "RelativeHour": int(h), "MeanZ": m,
             "CI95_low": l, "CI95_high": u, "N_events": int(np.isfinite(matrix[:, j]).sum())}
            for j, (h, m, l, u) in enumerate(zip(rel, estimate, lo, hi))
        )
    return pd.DataFrame(rows)
















def main(boot: int) -> None:
    require(DAILY_FILE)
    require(HOURLY_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(DAILY_FILE, parse_dates=["Date"]).sort_values("Date")
    if "WSPD" not in daily.columns:
        daily["WSPD"] = np.sqrt(pd.to_numeric(daily["U10"], errors="coerce") ** 2 +
                                pd.to_numeric(daily["V10"], errors="coerce") ** 2)
    cold = daily.loc[daily["HDD"] > 0].copy()
    threshold = cold["ObjectiveValue"].quantile(ALPHA / 100.0)
    events = cold.loc[cold["ObjectiveValue"] >= threshold].copy()
    events["StressThreshold"] = threshold
    events.to_csv(OUT_DIR / "cold_stress_days_alpha99p5.csv", index=False)
    print(f"[events] {len(events)} cold-regime stress days; threshold={threshold:.6g}")

    hourly = pd.read_csv(HOURLY_FILE, index_col="Time_EST", parse_dates=["Time_EST"], float_precision="round_trip")
    hourly["NetLoad"] = hourly["Load"] - hourly["Wind"]
    ofv = daily.set_index("Date")["ObjectiveValue"]
    hourly["ObjectiveValue"] = hourly.index.normalize().map(ofv)

    variables = ["Temperature_F", "HDD", "WSPD", "Wind", "Load", "NetLoad", "ObjectiveValue"]
    z = seasonal_zscore(hourly, variables)
    comp = event_composite(z, pd.DatetimeIndex(events["Date"]), boot)
    comp.to_csv(OUT_DIR / "event_window_composite.csv", index=False)
    print(f"[saved] {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Number of event/year-cluster bootstrap replicates (default: 1000).")
    parser.add_argument("--hourly-file", type=Path, default=HOURLY_FILE)
    parser.add_argument("--daily-table", type=Path, default=DAILY_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    DAILY_FILE, OUT_DIR = args.daily_table, args.output_dir
    HOURLY_FILE = args.hourly_file
    main(args.bootstrap)
