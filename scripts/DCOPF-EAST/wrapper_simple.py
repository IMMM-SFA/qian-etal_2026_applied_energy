# -*- coding: utf-8 -*-
"""
wrapper_simple.py for the restructured East DCOPF workflow.

Expected layout after reduced_network_data_allocation creates cases:

project folder/
├─ Data/
│  └─ data_allocation/
├─ Exp500_simple_300_2019/
│  ├─ EIC_simple.py
│  ├─ wrapper_simple.py
│  └─ EIC_data.dat
├─ Exp500_simple_MW_25_2019/
│  ├─ EIC_simple.py
│  ├─ wrapper_simple.py
│  └─ EIC_data.dat
└─ ...

This wrapper is copied into each Exp folder and run from there.
It reads EIC_data.dat locally, but reads year/NN-specific CSV inputs from
Data/data_allocation.

Main workflow:
1) Use --win 0, 1, 2, or 3 to solve one of four year windows.
2) Infer NN/year/transmission case from the Exp folder name.
3) Use outage-adjusted SimGenLimit and SimMustrunLimit already written
   into EIC_data.dat by EICDataSetup.py. For Approach 1, those limits should
   come from the 6,937-row ERCOT-style raw-generator metadata and raw available
   capacity files before EIC_data.dat is created.
4) Save window-specific solve outputs, e.g. mwh_win0.csv, duals_win0.csv.
   The wrapper does not re-save HorizonGenLimits/HorizonMustrunLimits because
   those annual base files are already produced by reduced_network_data_allocation.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

from pyomo.opt import SolverFactory
from EIC_simple import model as m1
from pyomo.core import Var
from pyomo.core import Constraint
import pandas as pd
import pyomo.environ as pyo


# =========================================================
# BASIC SETTINGS
# =========================================================
SOLVE_WINDOWS = [
    (1, 100),
    (101, 100),
    (201, 100),
    (301, 65),
]

Solvername = "gurobi"
Timelimit = 14400  # seconds per daily solve
Threadlimit = 4

logging.getLogger("pyomo.core").setLevel(logging.ERROR)


# =========================================================
# PATH / CASE HELPERS
# =========================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve one East DCOPF window from an Exp folder."
    )
    parser.add_argument(
        "--win",
        type=int,
        choices=range(4),
        default=0,
        help="which 0-3 solve window to run; default is 0",
    )

    # parse_known_args is more Spyder-friendly than parse_args because Spyder
    # may pass unrelated kernel arguments.
    args, _unknown = parser.parse_known_args()
    return args


def _ampl_name(x) -> str:
    """Return the same safe AMPL/Pyomo symbol name used by EICDataSetup."""
    s = str(x)
    s = s.replace(" ", "_")
    s = s.replace("&", "_")
    s = s.replace(".", "")
    s = s.replace("-", "_")
    return s


def _format_case_value(value: str) -> str:
    text = str(value)
    try:
        f = float(text)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return text


def _parse_exp_folder(folder_name: str) -> dict:
    """
    Parse folders created by the updated data-allocation script.

    Supported examples:
        Exp500_simple_300_2019
        Exp500_simple_-30_2019
        Exp500_simple_MW_25_2019
        Exp500_simple_MW_-50_2019
    """
    if not folder_name.startswith("Exp"):
        raise ValueError(
            f"wrapper_simple.py should be run from an Exp folder, but current folder is: {folder_name}"
        )

    rest = folder_name[3:]
    m = re.match(r"^(?P<NN>\d+)(?P<tail>_.*)$", rest)
    if m is None:
        raise ValueError(f"Could not parse NN from Exp folder name: {folder_name}")

    nn = m.group("NN")
    parts = m.group("tail").strip("_").split("_")

    if len(parts) < 3:
        raise ValueError(f"Could not parse transmission case/year from: {folder_name}")

    year = parts[-1]
    if not re.fullmatch(r"\d{4}", year):
        raise ValueError(f"Could not parse a four-digit year from: {folder_name}")

    if len(parts) >= 4 and parts[-3].upper() == "MW":
        uc = "_" + "_".join(parts[:-3])
        trans_mode = "MW"
        trans_value = _format_case_value(parts[-2])
        trans_tag = f"MW_{trans_value}"
    else:
        uc = "_" + "_".join(parts[:-2])
        trans_mode = "pct"
        trans_value = _format_case_value(parts[-2])
        trans_tag = f"tp_{trans_value}"

    return {
        "NN": nn,
        "UC": uc,
        "year": year,
        "trans_mode": trans_mode,
        "trans_value": trans_value,
        "trans_tag": trans_tag,
    }


def _find_data_allocation_dir(exp_dir: Path) -> Path:
    """
    Find Data/data_allocation.

    Normal layout:
        project/Exp...
        project/Data/data_allocation
    """
    candidates = [
        exp_dir.parent / "Data" / "data_allocation",
        exp_dir / "Data" / "data_allocation",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError("Could not find Data/data_allocation. Checked:\n" + checked)


def _safe_get_param(param_obj, key, default=0.0):
    """Safely read a Pyomo indexed parameter, returning default when missing."""
    try:
        if key in param_obj:
            return param_obj[key]
    except Exception:
        pass
    return default


def _safe_write(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)



# =========================================================
# CASE SETUP
# =========================================================
args = _parse_args()
win = args.win
win_start, win_len = SOLVE_WINDOWS[win]
win_end = win_start + win_len - 1
out_suffix = f"win{win}"

SCRIPT_DIR = Path(__file__).resolve().parent

# Make relative paths local to the Exp folder even when launched from Spyder.
os.chdir(SCRIPT_DIR)

context = _parse_exp_folder(SCRIPT_DIR.name)
NN = context["NN"]
year = context["year"]

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ALLOCATION_DIR = _find_data_allocation_dir(SCRIPT_DIR)

print(
    "wrapper_simple case: "
    f"NN={NN}, year={year}, mode={context['trans_mode']}, value={context['trans_value']}, "
    f"window={win} ({win_start}-{win_end})"
)
print(f"Exp folder: {SCRIPT_DIR}")
print(f"Reading allocated data from: {DATA_ALLOCATION_DIR}")

DATA_FILE = SCRIPT_DIR / "EIC_data.dat"
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find local EIC_data.dat in {SCRIPT_DIR}. "
        "Run EICDataSetup first to create one data file for each Exp folder."
    )


# =========================================================
# MODEL / SOLVER SETUP
# =========================================================
instance = m1.create_instance(str(DATA_FILE))
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

opt = SolverFactory(Solvername)
if Solvername == "cplex":
    opt.options["timelimit"] = Timelimit
elif Solvername == "gurobi":
    opt.options["TimeLimit"] = Timelimit
    opt.options["DualReductions"] = 0
    # opt.options["MIPFocus"] = 1

opt.options["threads"] = Threadlimit

H = instance.HorizonHours
K = range(1, H + 1)


# =========================================================
# DATA USED ONLY BY WRAPPER
# =========================================================
# EICDataSetup writes outage-adjusted hourly limits into EIC_data.dat:
#     SimGenLimit[j, hour]
#     SimMustrunLimit[bus, hour]
# Therefore the wrapper only transfers those simulation parameters into the
# mutable 24-hour horizon parameters used by EIC_simple.py.

if not (hasattr(instance, "SimGenLimit") and hasattr(instance, "HorizonGenLimit")):
    raise RuntimeError(
        "Model/data mismatch: EIC_simple.py must define SimGenLimit and HorizonGenLimit."
    )

if not (hasattr(instance, "SimMustrunLimit") and hasattr(instance, "HorizonMustrunLimit")):
    raise RuntimeError(
        "Model/data mismatch: EIC_simple.py must define SimMustrunLimit and HorizonMustrunLimit."
    )

print("Using outage-adjusted SimGenLimit and SimMustrunLimit from EIC_data.dat.")
print("Approach 1 wrapper does not read raw metadata directly; metadata consistency is checked upstream in reduced_network_data_allocation.")

# =========================================================
# SPACE TO STORE RESULTS FOR THIS WINDOW
# =========================================================
mwh = []
on = []
switch = []
flow = []
# srsv = []
# nrsv = []
slack = []
vlt_angle = []
duals = []
objective_values = []



# =========================================================
# WINDOW SOLVE LOOP
# =========================================================
for day in range(win_start, win_start + win_len):
    print(f"\n=== Solving day {day} of window {win_start}-{win_end} ===")

    # Load demand time series into the 24-hour horizon.
    for z in instance.buses:
        for i in K:
            hour = (day - 1) * 24 + i
            instance.HorizonDemand[z, i] = instance.SimDemand[z, hour]

    # Load must-run capacity into the 24-hour horizon.
    # These values are already outage-adjusted in EIC_data.dat.
    for z in instance.buses:
        for i in K:
            hour = (day - 1) * 24 + i
            instance.HorizonMustrunLimit[z, i] = _safe_get_param(
                instance.SimMustrunLimit, (z, hour), default=0.0
            )

    # Load daily hydro energy budget.
    if not (hasattr(instance, "HorizonHydroBudget") and hasattr(instance, "SimHydroDaily")):
        raise RuntimeError(
            "Hydro setup mismatch: EIC_data.dat provides SimHydroDaily, "
            "but EIC_simple.py must define HorizonHydroBudget and use it in the hydro budget constraint."
        )
    
    for z in instance.Hydro:
        instance.HorizonHydroBudget[z] = _safe_get_param(
            instance.SimHydroDaily, (z, day), default=0.0
        )

    # Load solar time series.
    for z in instance.Solar:
        for i in K:
            hour = (day - 1) * 24 + i
            instance.HorizonSolar[z, i] = _safe_get_param(instance.SimSolar, (z, hour), default=0.0)

    # Load wind time series.
    for z in instance.Wind:
        for i in K:
            hour = (day - 1) * 24 + i
            instance.HorizonWind[z, i] = _safe_get_param(instance.SimWind, (z, hour), default=0.0)

    # Load daily fuel prices for thermal generators.
    for z in instance.Thermal:
        instance.FuelPrice[z] = instance.SimFuelPrice[z, day]

    # Generator available-capacity limits.
    # These limits are already outage-adjusted and reduced-network aggregated
    # in EIC_data.dat. Do not subtract lost capacity again here.
    for z in instance.Outage:
        base_cap = float(pyo.value(instance.maxcap[z]))
        for i in K:
            hour = (day - 1) * 24 + i
            limit = _safe_get_param(instance.SimGenLimit, (z, hour), default=0.0)
            # Conservative cap protects against tiny numerical/aggregation differences.
            instance.HorizonGenLimit[z, i] = max(0.0, min(base_cap, float(limit)))


    # Solver execution and checking for feasible solutions.
    result = opt.solve(instance, tee=True, symbolic_solver_labels=True, load_solutions=False)

    if result.solver.status == pyo.SolverStatus.ok and (
        result.solver.termination_condition == pyo.TerminationCondition.optimal
        or result.solver.termination_condition == pyo.TerminationCondition.feasible
    ):
        instance.solutions.load_from(result)
        instance.ObjValue = pyo.value(instance.SystemCost)
        print(f"Objective Value for Day {day}: {instance.ObjValue}")
        objective_values.append((day, instance.ObjValue))
    else:
        print(f"The solver did not find a feasible solution for Day {day}.")

    print("LP")

    for c in instance.component_objects(Constraint, active=True):
        cobject = getattr(instance, str(c))
        if str(c) in ["Node_Constraint"]:
            for index in cobject:
                if int(index[1] > 0 and index[1] < 25):
                    try:
                        duals.append((index[0], index[1] + ((day - 1) * 24), instance.dual[cobject[index]]))
                    except KeyError:
                        duals.append((index[0], index[1] + ((day - 1) * 24), -999))

    for v in instance.component_objects(Var, active=True):
        varobject = getattr(instance, str(v))
        a = str(v)

        if a == "Theta":
            for index in varobject:
                if int(index[1] > 0 and index[1] < 25):
                    if index[0] in instance.buses:
                        vlt_angle.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

        if a == "mwh":
            for index in varobject:
                if int(index[1] > 0 and index[1] < 25):
                    if index[0] in instance.Gas:
                        mwh.append((index[0], "Gas", index[1] + ((day - 1) * 24), varobject[index].value))
                    elif index[0] in instance.Coal:
                        mwh.append((index[0], "Coal", index[1] + ((day - 1) * 24), varobject[index].value))
                    elif index[0] in instance.Oil:
                        mwh.append((index[0], "Oil", index[1] + ((day - 1) * 24), varobject[index].value))
                    elif index[0] in instance.Hydro:
                        mwh.append((index[0], "Hydro", index[1] + ((day - 1) * 24), varobject[index].value))
                    elif index[0] in instance.Solar:
                        mwh.append((index[0], "Solar", index[1] + ((day - 1) * 24), varobject[index].value))
                    elif index[0] in instance.Wind:
                        mwh.append((index[0], "Wind", index[1] + ((day - 1) * 24), varobject[index].value))

        if a == "on":
            for index in varobject:
                if int(index[1] > 0 and index[1] < 25):
                    on.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

        if a == "switch":
            for index in varobject:
                if int(index[1] > 0 and index[1] < 25):
                    switch.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

        if a == "S":
            for index in varobject:
                if index[0] in instance.buses:
                    slack.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

        if a == "Flow":
            for index in varobject:
                if int(index[1] > 0 and index[1] < 25):
                    flow.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

        # Carry final-hour dispatch into the next solved day within the window.
        for j in instance.Dispatchable:
            if instance.mwh[j, 24].value <= 0 and instance.mwh[j, 24].value >= -0.0001:
                newval_1 = 0
            else:
                newval_1 = instance.mwh[j, 24].value
            instance.mwh[j, 0] = newval_1
            instance.mwh[j, 0].fixed = True

    print(day)


# =========================================================
# SAVE WINDOW OUTPUTS LOCALLY IN THE EXP FOLDER
# =========================================================
vlt_angle_pd = pd.DataFrame(vlt_angle, columns=("Node", "Time", "Value"))
mwh_pd = pd.DataFrame(mwh, columns=("Generator", "Type", "Time", "Value"))
# on_pd = pd.DataFrame(on, columns=("Generator", "Time", "Value"))
# switch_pd = pd.DataFrame(switch, columns=("Generator", "Time", "Value"))
# srsv_pd = pd.DataFrame(srsv, columns=("Generator", "Time", "Value"))
# nrsv_pd = pd.DataFrame(nrsv, columns=("Generator", "Time", "Value"))
slack_pd = pd.DataFrame(slack, columns=("Node", "Time", "Value"))
flow_pd = pd.DataFrame(flow, columns=("Line", "Time", "Value"))
duals_pd = pd.DataFrame(duals, columns=["Bus", "Time", "Value"])
df_objective = pd.DataFrame(objective_values, columns=["Day", "ObjectiveValue"])


_safe_write(mwh_pd, SCRIPT_DIR / f"mwh_{out_suffix}.csv")
_safe_write(vlt_angle_pd, SCRIPT_DIR / f"vlt_angle_{out_suffix}.csv")
# _safe_write(on_pd, SCRIPT_DIR / f"on_{out_suffix}.csv")
# _safe_write(switch_pd, SCRIPT_DIR / f"switch_{out_suffix}.csv")
# _safe_write(srsv_pd, SCRIPT_DIR / f"srsv_{out_suffix}.csv")
# _safe_write(nrsv_pd, SCRIPT_DIR / f"nrsv_{out_suffix}.csv")
_safe_write(slack_pd, SCRIPT_DIR / f"slack_{out_suffix}.csv")
_safe_write(flow_pd, SCRIPT_DIR / f"flow_{out_suffix}.csv")
_safe_write(duals_pd, SCRIPT_DIR / f"duals_{out_suffix}.csv")
_safe_write(df_objective, SCRIPT_DIR / f"objective_values_{out_suffix}.csv")

print(f"Complete window {win}: {win_start}-{win_end}")
