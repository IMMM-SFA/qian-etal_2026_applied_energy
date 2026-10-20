"""Paper-package script. Output scope is documented in PAPER_OUTPUTS.md at the package root."""
from pathlib import Path

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
PACKAGE_ROOT = DATA_ROOT / "datasets_and_scripts"
REPRODUCED_ROOT = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve()
SECTION_DIR = PACKAGE_ROOT / "3.4_compound_amplification"

import numpy as np
import pandas as pd

ROOT = SECTION_DIR / "data"
DAILY_FILE = PACKAGE_ROOT / "shared" / "data" / "system_daily_table.csv"
OUT = REPRODUCED_ROOT / "3.4" / "solar_cutoff_diagnostics"
KEYS = ["Family", "Alpha", "TempTau", "TempCol"]
METRICS = ["Load", "Wind", "Solar", "Renewables", "NetLoad", "Temperature"]
FLAGS = ["LowerStressRate", "LowerLoad", "LowerRenewables", "LowerNetLoad",
         "DemandReductionExceedsRenewablesReduction", "MilderTemperature"]


def describe(frame):
    result = {"Days": len(frame), "StressDays": int(frame.Stress.sum()),
              "StressRate": frame.Stress.mean()}
    result.update({"Mean" + m: frame[m].mean() for m in METRICS})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY_FILE)
    required = ["ObjectiveValue", "Load", "Wind", "Solar", "CDD", "HDD"]
    daily[required] = daily[required].apply(pd.to_numeric, errors="coerce")
    grids = pd.concat([pd.read_csv(ROOT / f"lift_amplification_gridsearch_bootstrap_{f}.csv")
                       for f in ["COLD", "HOT"]], ignore_index=True)
    # Match DROP_ZERO_PCT=True in the original plotting script.
    plotted = grids[grids.WindPctLo.gt(0) & grids.SolarPctLo.gt(0)].copy()
    plotted["IsGreen"] = np.isfinite(plotted.Ampl_BootMean) & plotted.Ampl_BootMean.lt(1)
    records = []
    audits = []
    for key, panel in plotted.groupby(KEYS, sort=False):
        family, alpha, tau, temp_col = key
        regime = daily[daily[temp_col].fillna(0).gt(0)].dropna(
            subset=["Year", "Day", "ObjectiveValue", "Wind", "Solar"]).copy()
        threshold = panel.OV_Threshold.iloc[0]
        assert panel.OV_Threshold.nunique() == 1
        assert np.isclose(regime.ObjectiveValue.quantile(alpha / 100), threshold)
        temp = regime[regime[temp_col].ge(tau)].copy()
        if temp[required].isna().any().any():
            raise ValueError(f"Missing analysis values in {key}; do not silently change denominators")
        temp["Temperature"] = temp[temp_col]
        temp["Stress"] = temp.ObjectiveValue.ge(threshold)
        temp["Renewables"] = temp.Wind + temp.Solar
        temp["NetLoad"] = temp.Load - temp.Renewables
        green = panel[panel.IsGreen]
        audits.append(dict(zip(KEYS, key)) | {
            "PlottedCells": len(panel), "FiniteCells": int(np.isfinite(panel.Ampl_BootMean).sum()),
            "GreenCells": len(green), "TemperatureDays": len(temp),
            "TemperatureStressDays": int(temp.Stress.sum())})
        for _, cell in green.iterrows():
            wind = temp[temp.Wind.le(cell.WindThr)]
            compound = wind[wind.Solar.le(cell.SolarThr)]
            selected = describe(compound)
            assert 0 < len(compound) <= len(wind) <= len(temp)
            for comparison, baseline in [("temperature_only", temp), ("same_wind_cutoff", wind)]:
                base = describe(baseline)
                excluded = describe(baseline.loc[~baseline.index.isin(compound.index)])
                row = {k: cell[k] for k in KEYS + ["OV_Threshold", "WindPctLo", "SolarPctLo",
                       "WindThr", "SolarThr", "Ampl_BootMean", "SupportFrac", "MinSupportDays"]}
                row["Comparison"] = comparison
                row.update({"Selected" + k: v for k, v in selected.items()})
                row.update({"Baseline" + k: v for k, v in base.items()})
                row.update({"Excluded" + k: v for k, v in excluded.items()})
                row["ObservedRateRatio"] = selected["StressRate"] / base["StressRate"] if base["StressRate"] > 0 else np.nan
                row["StressRateDifference_pp"] = 100 * (selected["StressRate"] - base["StressRate"])
                for m in METRICS:
                    row["DeltaMean" + m] = selected["Mean" + m] - base["Mean" + m]
                row["LowerStressRate"] = row["StressRateDifference_pp"] < -1e-10
                for m in ["Load", "Renewables", "NetLoad"]:
                    row["Lower" + m] = row["DeltaMean" + m] < -1e-8
                row["MilderTemperature"] = row["DeltaMeanTemperature"] < -1e-10
                row["DemandReductionExceedsRenewablesReduction"] = (
                    row["LowerLoad"] and row["LowerRenewables"] and row["LowerNetLoad"])
                row["SolarRestrictionRemovesDays"] = len(compound) < len(wind)
                row["SelectedBelowOriginalMinimumSupport"] = len(compound) < cell.MinSupportDays
                assert np.isclose(row["DeltaMeanNetLoad"], row["DeltaMeanLoad"] - row["DeltaMeanRenewables"])
                assert selected["Days"] + excluded["Days"] == base["Days"]
                assert selected["StressDays"] + excluded["StressDays"] == base["StressDays"]
                records.append(row)
    detail = pd.DataFrame(records)
    detail.to_csv(OUT / "green_cells_both_comparisons.csv", index=False)
    detail.loc[detail.Comparison.eq("same_wind_cutoff")].to_csv(OUT / "green_cells_vs_same_wind_cutoff.csv", index=False)
    summaries = []
    for keys, group in detail.groupby(KEYS + ["Comparison"], sort=False):
        row = dict(zip(KEYS + ["Comparison"], keys))
        row["GreenCells"] = len(group)
        row["CellsWithNoSolarSelection"] = int((~group.SolarRestrictionRemovesDays).sum())
        row["CellsBelowOriginalMinimumSupport"] = int(group.SelectedBelowOriginalMinimumSupport.sum())
        row["MinSelectedStressDays"] = group.SelectedStressDays.min()
        for flag in FLAGS:
            row[flag + "_Cells"] = int(group[flag].sum())
            row[flag + "_Pct"] = 100 * group[flag].mean()
        for col in ["StressRateDifference_pp"] + ["DeltaMean" + m for m in METRICS]:
            for label, value in [("Min", group[col].min()), ("Median", group[col].median()), ("Max", group[col].max())]:
                row[col + "_" + label] = value
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "panel_summary.csv", index=False)
    assert len(detail) == 2 * sum(a["GreenCells"] for a in audits)
    assert not detail.duplicated(KEYS + ["WindPctLo", "SolarPctLo", "Comparison"]).any()
    print(summary[["Family", "Alpha", "Comparison", "GreenCells", "LowerStressRate_Pct",
                   "LowerLoad_Pct", "LowerNetLoad_Pct", "DemandReductionExceedsRenewablesReduction_Pct"]].to_string(index=False))
    print(f"\nSaved {len(detail)} comparison rows to {OUT}")


if __name__ == "__main__":
    main()
