# -*- coding: utf-8 -*-
"""
EICDataSetup for the restructured East DCOPF data-allocation workflow.

This version is intended to be placed in the same project folder as
the reduced_network_data_allocation script. That folder should contain:
       Data/data_allocation
       Exp500_simple_300_2019
       Exp500_simple_MW_25_2019
       ...

Default behavior:
   Run this script once from any working directory. It uses the folder where
   this script is located as the project root, finds all matching Exp folders
   in that same folder, and writes one EIC_data.dat inside each Exp folder.

It also still supports running from inside a single Exp folder or manually
passing --nn, --year, --trans-mode, and --trans-value.

All generated data inputs are read from:
       <script folder>/Data/data_allocation

The same script supports both transmission-expansion modes:
       +%  folder/file tag:  tp_<percent>, e.g. tp_300
       +MW folder/file tag:  MW_<delta>,   e.g. MW_25
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# =========================================================
# USER-LEVEL CONSTANTS
# =========================================================
SimDays = 365
SimHours = SimDays * 24
HorizonHours = 24
DATA_NAME = "EIC_data"


def _script_dir() -> Path:
    """Return the folder containing this script.

    This is the project/root folder when EICDataSetup.py is placed next to
    reduced_network_data_allocation*.py. Using this instead of Path.cwd()
    avoids Spyder/IDE working-directory issues.
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# =========================================================
# HELPERS
# =========================================================
def _ampl_name(x) -> str:
    """Return a safe AMPL symbol name consistent across all blocks."""
    s = str(x)
    s = s.replace(" ", "_")
    s = s.replace("&", "_")
    s = s.replace(".", "")
    s = s.replace("-", "_")
    return s


def _format_case_value(value: str) -> str:
    """
    Keep folder/file values stable.
    If the value is an integer-like float string, convert 300.0 -> 300.
    Otherwise keep the original text.
    """
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
    Parse experiment folder names created by the updated data-allocation script.

    Supported forms:
        Exp500_simple_300_2019      -> NN=500, mode=pct, value=300, year=2019
        Exp500_simple_MW_25_2019    -> NN=500, mode=MW,  value=25,  year=2019
        Exp500_simple_MW_-50_2019   -> NN=500, mode=MW,  value=-50, year=2019
    """
    if not folder_name.startswith("Exp"):
        raise ValueError(f"Folder does not look like an Exp folder: {folder_name}")

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

    if len(parts) >= 4 and parts[-3] == "MW":
        uc = "_" + "_".join(parts[:-3])
        trans_mode = "MW"
        trans_value = _format_case_value(parts[-2])
        trans_tag = f"MW_{trans_value}"
    else:
        uc = "_" + "_".join(parts[:-2])
        trans_mode = "pct"
        trans_value = _format_case_value(parts[-2])
        trans_tag = f"tp_{trans_value}"

    if uc == "_":
        raise ValueError(f"Could not parse UC treatment from: {folder_name}")

    return {
        "NN": nn,
        "UC": uc,
        "year": year,
        "trans_mode": trans_mode,
        "trans_value": trans_value,
        "trans_tag": trans_tag,
        "exp_folder_name": folder_name,
    }


def _find_data_allocation_dir(user_path: str | None = None) -> Path:
    """Find Data/data_allocation relative to the script folder by default."""
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Provided data-allocation directory does not exist: {p}")
        return p

    # Priority 1: same project folder as this script.
    # This is the expected layout when this file is placed next to the
    # reduced_network_data_allocation script.
    script_root = _script_dir()
    checked = []
    cand = script_root / "Data" / "data_allocation"
    checked.append(str(cand))
    if cand.exists():
        return cand

    # Fallbacks: useful only if the script is copied elsewhere or run from an Exp folder.
    starts = [Path.cwd().resolve(), script_root]
    for start in starts:
        for p in [start] + list(start.parents):
            cand = p / "Data" / "data_allocation"
            checked.append(str(cand))
            if cand.exists():
                return cand

    raise FileNotFoundError(
        "Could not find Data/data_allocation. Checked:\n" + "\n".join(dict.fromkeys(checked))
    )

def _must_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, **kwargs)


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose total is zero, preserving the legacy behavior."""
    empty = []
    for col in df.columns:
        if pd.to_numeric(df[col], errors="coerce").fillna(0).sum() <= 0:
            empty.append(col)
    return df.drop(columns=empty)




def _read_required_hourly_wide(path: Path, expected_rows: int = SimHours) -> pd.DataFrame:
    """
    Read a required hourly wide table with one row per hour.

    Supported formats:
        Hour, col_1, col_2, ...
        unnamed/index column, col_1, col_2, ...
        col_1, col_2, ...       # row position is treated as Hour 1..N

    Returned dataframe is indexed by 1-based Hour and contains numeric values.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required hourly limit file not found: {path}")

    df = pd.read_csv(path, header=0)
    if len(df) != expected_rows:
        raise ValueError(f"{path.name} has {len(df)} rows; expected {expected_rows}.")

    first_col = str(df.columns[0])
    first_lower = first_col.lower()

    if first_lower.startswith("unnamed") or first_lower in ["hour", "time", "hour_of_year"]:
        hour_values = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df = df.drop(columns=[df.columns[0]])

        if hour_values.notna().all():
            hour_values = hour_values.astype(int)
            if hour_values.min() == 0 and hour_values.max() == expected_rows - 1:
                df.index = hour_values + 1
            else:
                df.index = hour_values
        else:
            df.index = range(1, len(df) + 1)
    else:
        df.index = range(1, len(df) + 1)

    df.index.name = "Hour"
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def _build_column_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Map original and AMPL-safe names to dataframe columns."""
    lookup = {}
    for col in df.columns:
        lookup[str(col)] = col
        lookup[_ampl_name(col)] = col
    return lookup


def _hourly_value_from_wide(
    df: pd.DataFrame,
    column_lookup: dict[str, str],
    entity_name: str,
    hour_1based: int,
    default: float = 0.0,
) -> float:
    """Safely read one value from a 1-based hourly wide table."""
    col = column_lookup.get(str(entity_name)) or column_lookup.get(_ampl_name(entity_name))
    if col is None:
        return default

    try:
        value = df.loc[hour_1based, col]
    except Exception:
        return default

    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _write_set(f, set_name: str, values) -> None:
    f.write(f"set {set_name} :=\n")
    for value in values:
        f.write(_ampl_name(value) + " ")
    f.write(";\n\n")


def _write_param_table(f, df: pd.DataFrame, row_name_col: str = "name") -> None:
    """Write generator parameter table in the same AMPL style as the legacy script."""
    f.write("param:" + "\t")
    for c in df.columns:
        if c != row_name_col:
            f.write(c + "\t")
    f.write(":=\n\n")

    for i in range(len(df)):
        f.write(_ampl_name(df.loc[i, row_name_col]) + "\t")
        for c in df.columns:
            if c != row_name_col:
                f.write(str(df.loc[i, c]) + "\t")
        f.write("\n")
    f.write(";\n\n")


def _write_matrix_with_row_name(f, param_name: str, df: pd.DataFrame, row_col: str) -> None:
    f.write(f"param {param_name}:\n")
    f.write("\t")
    for c in df.columns:
        if c != row_col:
            f.write(str(c) + "\t")
    f.write(":=\n")

    for i in range(len(df)):
        for c in df.columns:
            value = df.loc[i, c]
            if c == row_col:
                value = _ampl_name(value)
            f.write(str(value) + "\t")
        f.write("\n")
    f.write(";\n\n")


def _find_exp_case_dirs(search_dir: Path) -> list[Path]:
    """Find immediate child Exp folders that match the supported naming structure."""
    exp_dirs = []
    for p in sorted(search_dir.iterdir()):
        if not p.is_dir() or not p.name.startswith("Exp"):
            continue
        try:
            _parse_exp_folder(p.name)
        except ValueError:
            continue
        exp_dirs.append(p)
    return exp_dirs


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create EIC_data.dat using files from Data/data_allocation."
    )
    parser.add_argument("--nn", "--NN", dest="NN", default=None)
    parser.add_argument("--year", dest="year", default=None)
    parser.add_argument("--trans-mode", dest="trans_mode", choices=["pct", "MW"], default=None)
    parser.add_argument("--trans-value", dest="trans_value", default=None)
    parser.add_argument("--data-allocation-dir", dest="data_allocation_dir", default=None)
    parser.add_argument(
        "--exp-root",
        dest="exp_root",
        default=None,
        help="Folder containing Exp* subfolders. Default: folder where this script is located.",
    )
    # parse_known_args keeps the script more robust when run through Spyder.
    args, _unknown = parser.parse_known_args()
    return args


def _build_single_context_from_args(args) -> dict | None:
    """Return a single explicit context if all case arguments were supplied."""
    supplied = [args.NN, args.year, args.trans_mode, args.trans_value]
    if all(v is None for v in supplied):
        return None
    if not all(v is not None for v in supplied):
        missing = []
        if args.NN is None:
            missing.append("--nn")
        if args.year is None:
            missing.append("--year")
        if args.trans_mode is None:
            missing.append("--trans-mode")
        if args.trans_value is None:
            missing.append("--trans-value")
        raise ValueError(
            "If passing case information manually, please pass all of: " + ", ".join(missing)
        )

    trans_value = _format_case_value(args.trans_value)
    trans_tag = f"tp_{trans_value}" if args.trans_mode == "pct" else f"MW_{trans_value}"
    return {
        "NN": str(args.NN),
        "UC": None,
        "year": str(args.year),
        "trans_mode": args.trans_mode,
        "trans_value": trans_value,
        "trans_tag": trans_tag,
        "exp_folder_name": None,
    }


def _resolve_work_items(args, data_allocation_dir: Path) -> list[tuple[dict, Path]]:
    """
    Decide which EIC_data.dat files to write.

    Priority:
      1. If --nn/--year/--trans-mode/--trans-value are provided, write one file to cwd.
      2. If cwd itself is an Exp folder, write one file to cwd.
      3. Otherwise, find all matching Exp* folders under the script folder or --exp-root
         and write one file in each.
    """
    explicit_context = _build_single_context_from_args(args)
    if explicit_context is not None:
        explicit_context["data_allocation_dir"] = data_allocation_dir
        return [(explicit_context, Path.cwd())]

    # Case 2: run from inside one Exp folder.
    try:
        context = _parse_exp_folder(Path.cwd().name)
        context["data_allocation_dir"] = data_allocation_dir
        return [(context, Path.cwd())]
    except ValueError:
        pass

    # Case 3: run from project/root folder and process all immediate Exp folders.
    exp_root = Path(args.exp_root).expanduser().resolve() if args.exp_root else _script_dir()
    exp_dirs = _find_exp_case_dirs(exp_root)
    if not exp_dirs:
        raise ValueError(
            "Could not infer a case and found no matching Exp folders.\n"
            "Either run from an Exp folder such as Exp500_simple_300_2019 or "
            "Exp500_simple_MW_25_2019, run from the project folder containing those "
            "Exp folders, or pass --nn, --year, --trans-mode, and --trans-value.\n"
            f"Default Exp search folder was: {exp_root}"
        )

    work_items = []
    for exp_dir in exp_dirs:
        context = _parse_exp_folder(exp_dir.name)
        context["data_allocation_dir"] = data_allocation_dir
        work_items.append((context, exp_dir))
    return work_items


# =========================================================
# CORE WRITER
# =========================================================
def write_eic_data_for_case(context: dict, output_dir: Path) -> Path:
    NN = context["NN"]
    year = context["year"]
    trans_tag = context["trans_tag"]
    data_allocation_dir = context["data_allocation_dir"]

    print(
        "EICDataSetup case: "
        f"NN={NN}, year={year}, mode={context['trans_mode']}, "
        f"value={context['trans_value']}, tag={trans_tag}"
    )
    print(f"Reading allocated data from: {data_allocation_dir}")
    print(f"Writing EIC_data.dat to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Read parameters for dispatchable and renewable resources
    # -----------------------------------------------------
    df_gen = _must_read_csv(data_allocation_dir / f"data_genparams_{NN}.csv", header=0)

    thermal_generators_df = df_gen.loc[
        df_gen["typ"].isin(["coal", "ngcc", "ngct", "oil"])
    ].copy()
    thermal_generators_names = list(thermal_generators_df["name"])

    # Read generation and transmission data
    df_bustounitmap = _must_read_csv(data_allocation_dir / f"gen_mat_{NN}.csv", header=0)
    df_linetobusmap = _must_read_csv(data_allocation_dir / f"line_to_bus_{NN}_{trans_tag}.csv", header=0)
    df_line_params = _must_read_csv(data_allocation_dir / f"line_params_{NN}_{trans_tag}.csv", header=0)
    lines = list(df_line_params["line"])

    # Hourly renewable profiles and load
    df_solar = _drop_empty_columns(
        _must_read_csv(data_allocation_dir / f"nodal_solar_{NN}_y_{year}.csv", header=0)
    )
    df_wind = _drop_empty_columns(
        _must_read_csv(data_allocation_dir / f"nodal_wind_{NN}_y_{year}.csv", header=0)
    )
    df_load = _must_read_csv(data_allocation_dir / f"nodal_load_{NN}_y_{year}.csv", header=0)

    # Daily hydro energy budget at each plant
    df_hydro_daily = _drop_empty_columns(
        _must_read_csv(data_allocation_dir / f"nodal_hydro_daily_{NN}.csv", header=0)
    )

    # Reduced-network outage-adjusted capacity limits and fuel prices.
    # These files are created by reduced_network_data_allocation.py from the
    # raw/pre-reduction available-capacity output of GadsOutagesEAST.py.
    gen_limit_path = data_allocation_dir / f"HorizonGenLimits_base_{NN}_y_{year}.csv"
    mustrun_limit_path = data_allocation_dir / f"HorizonMustrunLimits_base_{NN}_y_{year}.csv"

    df_gen_limits = _read_required_hourly_wide(gen_limit_path, expected_rows=SimHours)
    df_mustrun_limits = _read_required_hourly_wide(mustrun_limit_path, expected_rows=SimHours)

    print(f"Using hourly thermal outage limits from: {gen_limit_path.name}")
    print(f"Using hourly must-run limits from: {mustrun_limit_path.name}")

    gen_limit_lookup = _build_column_lookup(df_gen_limits)
    mustrun_limit_lookup = _build_column_lookup(df_mustrun_limits)

    df_fuel = _must_read_csv(data_allocation_dir / f"Fuel_prices_{NN}_y_{year}.csv", header=0)

    all_nodes = list(df_load.columns)
    all_thermals = list(df_fuel.columns)

    missing_gen_limit_cols = [
        z for z in thermal_generators_names
        if str(z) not in gen_limit_lookup and _ampl_name(z) not in gen_limit_lookup
    ]
    if missing_gen_limit_cols:
        raise ValueError(
            f"{gen_limit_path.name} is missing {len(missing_gen_limit_cols)} thermal "
            f"generator columns required by the model. Examples: {missing_gen_limit_cols[:10]}"
        )

    missing_mustrun_cols = [
        z for z in all_nodes
        if str(z) not in mustrun_limit_lookup and _ampl_name(z) not in mustrun_limit_lookup
    ]
    if missing_mustrun_cols:
        raise ValueError(
            f"{mustrun_limit_path.name} is missing {len(missing_mustrun_cols)} bus "
            f"columns required by the model. Examples: {missing_mustrun_cols[:10]}"
        )

    # -----------------------------------------------------
    # Write data file
    # -----------------------------------------------------
    output_file = output_dir / f"{DATA_NAME}.dat"

    with output_file.open("w") as f:
        # Generator sets by type
        _write_set(f, "Coal", df_gen.loc[df_gen["typ"] == "coal", "name"])
        _write_set(f, "Oil", df_gen.loc[df_gen["typ"] == "oil", "name"])
        _write_set(f, "Gas", df_gen.loc[df_gen["typ"].isin(["ngcc", "ngct"]), "name"])
        _write_set(f, "Hydro", df_gen.loc[df_gen["typ"] == "hydro", "name"])
        _write_set(f, "Solar", df_gen.loc[df_gen["typ"] == "solar", "name"])
        _write_set(f, "Wind", df_gen.loc[df_gen["typ"] == "wind", "name"])
        print("Gen sets")

        # Unit outage category sets are intentionally removed.
        # The new outage inputs are model-ready hourly capacity limits:
        #     HorizonGenLimits_base_{NN}_y_{year}.csv
        #     HorizonMustrunLimits_base_{NN}_y_{year}.csv
        # These are written into SimGenLimit and SimMustrunLimit below.

        # Nodes and lines
        _write_set(f, "buses", all_nodes)
        print("nodes")

        _write_set(f, "lines", lines)
        print("lines")

        # Simulation period and horizon
        f.write("param SimHours := %d;" % SimHours)
        f.write("\n")
        f.write("param SimDays:= %d;" % SimDays)
        f.write("\n\n")
        f.write("param HorizonHours := %d;" % HorizonHours)
        f.write("\n\n")

        # Generator parameter matrix
        _write_param_table(f, df_gen, row_name_col="name")

        # Hourly dispatchable thermal capacity after outages.
        # These values are already reduced-network aggregated and outage-adjusted.
        f.write("param:" + "\t" + "SimGenLimit:=" + "\n")
        for z in thermal_generators_names:
            thermal_gen_capacity = float(
                thermal_generators_df.loc[
                    thermal_generators_df["name"] == z, "maxcap"
                ].values[0]
            )

            for h in range(SimHours):
                hour = h + 1
                value = _hourly_value_from_wide(
                    df_gen_limits,
                    gen_limit_lookup,
                    str(z),
                    hour,
                    default=thermal_gen_capacity,
                )
                # Safety bound: the reduced limit should not exceed model maxcap.
                value = max(0.0, min(float(value), thermal_gen_capacity))
                f.write(_ampl_name(z) + "\t" + str(hour) + "\t" + str(value) + "\n")
        f.write(";\n\n")

        # Hourly must-run capacity after nuclear outages.
        f.write("param:" + "\t" + "SimMustrunLimit:=" + "\n")
        for z in all_nodes:
            for h in range(SimHours):
                hour = h + 1
                value = _hourly_value_from_wide(
                    df_mustrun_limits,
                    mustrun_limit_lookup,
                    str(z),
                    hour,
                    default=0.0,
                )
                value = max(0.0, float(value))
                f.write(z + "\t" + str(hour) + "\t" + str(value) + "\n")
        f.write(";\n\n")
        print("Gen params")

        # Transmission paths
        f.write("param:" + "\t" + "FlowLim" + "\t" + "Reactance :=" + "\n")
        for idx, z in enumerate(lines):
            f.write(
                _ampl_name(z) + "\t" +
                str(df_line_params.loc[idx, "limit"]) + "\t" +
                str(df_line_params.loc[idx, "reactance"]) + "\n"
            )
        f.write(";\n\n")
        print("trans paths")

        # Hourly load
        f.write("param:" + "\t" + "SimDemand:=" + "\n")
        for z in all_nodes:
            for h in range(len(df_load)):
                f.write(z + "\t" + str(h + 1) + "\t" + str(df_load.loc[h, z]) + "\n")
        f.write(";\n\n")
        print("load")

        # Hourly solar
        f.write("param:" + "\t" + "SimSolar:=" + "\n")
        for z in df_solar.columns:
            for h in range(len(df_solar)):
                f.write(z + "_SOLAR" + "\t" + str(h + 1) + "\t" + str(df_solar.loc[h, z]) + "\n")
        f.write(";\n\n")
        print("solar")

        # Hourly wind
        f.write("param:" + "\t" + "SimWind:=" + "\n")
        for z in df_wind.columns:
            for h in range(len(df_wind)):
                f.write(z + "_WIND" + "\t" + str(h + 1) + "\t" + str(df_wind.loc[h, z]) + "\n")
        f.write(";\n\n")
        print("wind")

        # Daily hydro energy budget
        f.write("param:" + "\t" + "SimHydroDaily:=" + "\n")
        for z in df_hydro_daily.columns:
            for d in range(len(df_hydro_daily)):
                f.write(z + "_HYDRO" + "\t" + str(d + 1) + "\t" + str(df_hydro_daily.loc[d, z]) + "\n")
        f.write(";\n\n")
        print("hydro daily")

        # Maps
        _write_matrix_with_row_name(f, "BustoUnitMap", df_bustounitmap, row_col="name")
        print("Bus to units")

        _write_matrix_with_row_name(f, "LinetoBusMap", df_linetobusmap, row_col="line")
        print("line to bus")

        # Daily fuel prices
        f.write("param:" + "\t" + "SimFuelPrice:=" + "\n")
        for z in all_thermals:
            for d in range(SimDays):
                f.write(_ampl_name(z) + "\t" + str(d + 1) + "\t" + str(df_fuel.loc[d, z]) + "\n")
        f.write(";\n\n")
        print("fuel prices")

    print("Complete:", output_file)
    return output_file


# =========================================================
# MAIN SCRIPT
# =========================================================
def main() -> None:
    args = _parse_args()
    data_allocation_dir = _find_data_allocation_dir(args.data_allocation_dir)
    work_items = _resolve_work_items(args, data_allocation_dir)

    if len(work_items) > 1:
        print(f"Found {len(work_items)} Exp folders. Writing one {DATA_NAME}.dat file in each folder.")

    outputs = []
    for context, output_dir in work_items:
        outputs.append(write_eic_data_for_case(context, output_dir))

    print(f"Finished {len(outputs)} case(s).")


if __name__ == "__main__":
    main()
