# -*- coding: utf-8 -*-
"""
Restructured East DCOPF data-allocation script.

Main structural changes:
- NN-only work is done once per reduced network size.
- Transmission data are done once per NN and transmission case, supporting both percent expansion and additive-MW expansion.
- Load, wind, solar, fuel prices, and lost-capacity files are handled by year.
- Year-specific run folders contain only model scripts; all generated data stay in Data/data_allocation.
- Raw/pre-reduction generator available-capacity files from GadsOutagesEAST are aggregated into reduced-network hourly HorizonGenLimits and HorizonMustrunLimits.
- HorizonGenLimits_base and HorizonMustrunLimits_base are written once per NN/year and reused by all transmission expansion cases.
"""


import pandas as pd
import math
import numpy as np
import os
import xlrd
from shutil import copy
from pathlib import Path

########################################
# GLOBAL OUTPUT LOCATIONS
########################################
base_dir = 'Data'
data_allocation_dir = os.path.join(base_dir, 'data_allocation')
os.makedirs(data_allocation_dir, exist_ok=True)

NODE_NUMBER = [500]

# UC_TREATMENTS = ['_simple','_coal']
UC_TREATMENTS = ['_simple']

# Transmission expansion cases.
# Percent cases use: limit = base_limit * (1 + pct/100)
# MW cases use:      limit = base_limit + delta_MW
# Use empty lists if you do not want to run one of the two scenario types.
#
# Examples:
#trans_p  = [0, 100, 150, 200, 250, 300]
#trans_MW = [100, 150, 200, 250, 300]
trans_p = [300]
trans_MW = []

# Fuel price processing mode.
#   'annual'   : use year-specific daily fuel-price files when available
#                (fallback to static files if the year-specific file is missing).
#   'constant' : use one constant price per bus/fuel across all 365 days,
#                following the old const-fuel-price script logic.
fuel_price_mode = 'annual'  # options: 'annual', 'constant'

if fuel_price_mode not in {'annual', 'constant'}:
    raise ValueError(
        "fuel_price_mode must be either 'annual' or 'constant'. "
        f"Got: {fuel_price_mode}"
    )


# Helper to save CSVs into the new folder with a name

def _save_csv(df: pd.DataFrame, filename: str, index: bool = False):
    """Save a CSV to Data/data_allocation and return the full path."""
    full = os.path.join(data_allocation_dir, filename)
    df.to_csv(full, index=index)
    return full


def _first_existing_path(base_dir: str, relative_candidates):
    """Return the first existing path from a list of relative paths."""
    checked = []
    for rel in relative_candidates:
        p = os.path.join(base_dir, rel)
        checked.append(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find any of:\n" + "\n".join(checked))


def _read_csv_first_existing(base_dir: str, relative_candidates, **kwargs):
    """Read the first existing CSV from a list of relative paths."""
    return pd.read_csv(_first_existing_path(base_dir, relative_candidates), **kwargs)


def _normalize_bus_id(x):
    """Normalize bus ids so 123, 123.0, and '123' compare consistently."""
    try:
        f = float(x)
        if f.is_integer():
            return int(f)
    except Exception:
        pass
    return str(x)


def _raw_outage_metadata_src(base_dir: str) -> str:
    """
    Find raw/pre-reduction generator metadata created by GadsOutagesEAST.py.

    Expected file from the adjusted GadsOutagesEAST workflow:
        east_rawGens_metadata.csv
    """
    candidates = [
        os.path.join(base_dir, 'Gen', 'east_rawGens_metadata.csv'),
        os.path.join(base_dir, 'data_allocation', 'east_rawGens_metadata.csv'),
        'east_rawGens_metadata.csv',
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not find east_rawGens_metadata.csv. Checked:\n" + "\n".join(candidates)
    )


def _raw_available_cap_src_for_year(base_dir: str, year) -> str:
    """
    Find raw/pre-reduction hourly generator available-capacity file.

    Expected file from the adjusted GadsOutagesEAST workflow:
        east2019_rawGensAvailableCap.csv
    """
    y = str(year)
    candidates = [
        os.path.join(base_dir, 'Gen', f'east{y}_rawGensAvailableCap.csv'),
        os.path.join(base_dir, 'data_allocation', f'east{y}_rawGensAvailableCap.csv'),
        f'east{y}_rawGensAvailableCap.csv',
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Could not find raw generator available-capacity file for year {y}. Checked:\n" +
        "\n".join(candidates)
    )


def _read_hourly_wide_file(path: str):
    """
    Read an 8760-row wide hourly file.

    If the first column is Time/Hour/Hour_of_Year/Unnamed, use it as the hour column;
    otherwise create a 1..8760 hour index.
    """
    df = pd.read_csv(path, header=0)
    if len(df) != 8760:
        raise ValueError(f"{path} has {len(df)} rows; expected 8760.")

    first_col = str(df.columns[0])
    first_lower = first_col.lower()
    if first_lower.startswith('unnamed') or first_lower in {'time', 'hour', 'hour_of_year'}:
        default_hours = pd.Series(np.arange(1, len(df) + 1), index=df.index)
        hours = pd.to_numeric(df[first_col], errors='coerce').fillna(default_hours)
        values = df.drop(columns=[first_col]).copy()
    else:
        hours = pd.Series(np.arange(1, len(df) + 1), index=df.index, name='Hour')
        values = df.copy()

    # Normalize generator column names to strings so they match Raw_Output_Name
    # values from east_rawGens_metadata.csv.
    values.columns = values.columns.astype(str)
    values = values.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return pd.Series(hours, name='Hour'), values


def _validate_raw_metadata_and_available(
    *,
    raw_meta: pd.DataFrame,
    raw_available: pd.DataFrame,
    raw_metadata_path: str,
    raw_available_path: str,
    year: str,
) -> dict:
    """
    Validate that the GADS metadata and raw available-capacity file are aligned.

    After switching to the ERCOT-style row-level GADS allocation, the expected
    relationship is one metadata row per raw thermal generator and one hourly
    available-capacity column per Raw_Output_Name.  Missing source columns should
    be treated as a data error, not as zero capacity/lost capacity.
    """
    meta_names = raw_meta['Raw_Output_Name'].astype(str).tolist()
    available_cols = [str(c) for c in raw_available.columns]

    duplicate_meta_names = sorted(raw_meta.loc[
        raw_meta['Raw_Output_Name'].astype(str).duplicated(),
        'Raw_Output_Name'
    ].astype(str).unique().tolist())
    if duplicate_meta_names:
        raise ValueError(
            f"{raw_metadata_path} has duplicated Raw_Output_Name values. "
            f"Examples: {duplicate_meta_names[:10]}"
        )

    duplicate_available_cols = sorted(pd.Series(available_cols).loc[
        pd.Series(available_cols).duplicated()
    ].astype(str).unique().tolist())
    if duplicate_available_cols:
        raise ValueError(
            f"{raw_available_path} has duplicated generator columns after reading. "
            f"Examples: {duplicate_available_cols[:10]}"
        )

    if 'SourceRowCount' in raw_meta.columns:
        source_row_count = pd.to_numeric(raw_meta['SourceRowCount'], errors='coerce')
        bad_source_rows = raw_meta.loc[source_row_count.ne(1)]
        if len(bad_source_rows) > 0:
            examples = bad_source_rows[[
                'Raw_Output_Name', 'RawBusNum', 'PlantNames', 'Fuel', 'SourceRowCount'
            ]].head(10).to_dict('records')
            raise ValueError(
                f"{raw_metadata_path} appears to contain grouped metadata rows, not "
                f"ERCOT-style raw thermal rows. SourceRowCount != 1 for "
                f"{len(bad_source_rows)} rows. Examples: {examples}"
            )

    meta_set = set(meta_names)
    available_set = set(available_cols)
    missing_available_cols = sorted(meta_set - available_set)
    extra_available_cols = sorted(available_set - meta_set)

    if missing_available_cols or extra_available_cols:
        # Save a small diagnostic before failing, when possible.
        diag_rows = []
        diag_rows.extend({
            'Issue': 'metadata Raw_Output_Name missing from raw available-capacity file',
            'Name': name,
        } for name in missing_available_cols[:5000])
        diag_rows.extend({
            'Issue': 'raw available-capacity column not found in metadata',
            'Name': name,
        } for name in extra_available_cols[:5000])
        if diag_rows:
            diag = pd.DataFrame(diag_rows)
            fn_diag = f'raw_available_metadata_name_mismatch_y_{year}.csv'
            _save_csv(diag, fn_diag, index=False)

        raise ValueError(
            f"Raw metadata and available-capacity file are not aligned for year {year}. "
            f"metadata rows={len(raw_meta)}, available columns={len(available_cols)}, "
            f"missing available columns={len(missing_available_cols)}, "
            f"extra available columns={len(extra_available_cols)}. "
            f"See raw_available_metadata_name_mismatch_y_{year}.csv for examples."
        )

    summary = {
        'Year': year,
        'Metadata_Path': raw_metadata_path,
        'Raw_Available_Path': raw_available_path,
        'Metadata_Rows': int(len(raw_meta)),
        'Unique_Raw_Output_Name': int(len(meta_set)),
        'Raw_Available_Columns': int(len(available_cols)),
        'Missing_Available_Columns': int(len(missing_available_cols)),
        'Extra_Available_Columns': int(len(extra_available_cols)),
        'Row_Level_Metadata': bool('SourceRowCount' not in raw_meta.columns or (pd.to_numeric(raw_meta['SourceRowCount'], errors='coerce') == 1).all()),
    }
    return summary


def _aggregate_raw_available_capacity_to_reduced_limits(
    *,
    NN,
    year: str,
    raw_metadata_path: str,
    raw_available_path: str,
    df_reduced_thermal_map: pd.DataFrame,
    final_genparams_path: str,
    all_buses,
    raw_to_reduced_bus: dict,
):
    """
    Convert raw/pre-reduction generator available capacity to model-ready limits.

    Input from GadsOutagesEAST:
        east_rawGens_metadata.csv
        east{year}_rawGensAvailableCap.csv

    Outputs to Data/data_allocation:
        HorizonGenLimits_base_{NN}_y_{year}.csv
        HorizonMustrunLimits_base_{NN}_y_{year}.csv
        raw_to_reduced_outage_map_{NN}_y_{year}.csv
        raw_to_reduced_outage_unmapped_{NN}_y_{year}.csv

    Notes:
      - Coal and gas keep plant-level reduced generator names.
      - Oil is aggregated to final node-level oil generators, e.g. bus_123_oil.
      - Nuclear is aggregated to bus-level must-run capacity, e.g. bus_123.
    """
    raw_meta = pd.read_csv(raw_metadata_path, header=0)
    hours, raw_available = _read_hourly_wide_file(raw_available_path)
    final_genparams = pd.read_csv(final_genparams_path, header=0)

    required_meta_cols = {'Raw_Output_Name', 'RawBusNum', 'PlantNames', 'Fuel', 'Max_Cap'}
    missing_meta = required_meta_cols - set(raw_meta.columns)
    if missing_meta:
        raise ValueError(
            f"{raw_metadata_path} is missing required columns: {sorted(missing_meta)}"
        )

    consistency_summary = _validate_raw_metadata_and_available(
        raw_meta=raw_meta,
        raw_available=raw_available,
        raw_metadata_path=raw_metadata_path,
        raw_available_path=raw_available_path,
        year=str(year),
    )
    full_consistency = _save_csv(
        pd.DataFrame([consistency_summary]),
        f'raw_available_metadata_consistency_{NN}_y_{year}.csv',
        index=False,
    )

    # Final model outage generators are the dispatchable thermal generators in EIC_simple.py.
    model_outage_gens = final_genparams.loc[
        final_genparams['typ'].isin(['coal', 'ngcc', 'ngct', 'oil']),
        'name'
    ].astype(str).tolist()

    # Model buses for must-run limits. Use all reduced-network buses, including non-nuclear buses.
    bus_cols = [f'bus_{_normalize_bus_id(b)}' for b in all_buses]

    gen_limits = pd.DataFrame(0.0, index=raw_available.index, columns=model_outage_gens)
    mustrun_limits = pd.DataFrame(0.0, index=raw_available.index, columns=bus_cols)

    # Lookup from reduced pre-oil thermal groups to their reduced generator name.
    # df_reduced_thermal_map is created in the same order as thermal_gens_{NN}.csv.
    reduced_lookup = {}
    for _, row in df_reduced_thermal_map.iterrows():
        key = (
            _normalize_bus_id(row['Bus']),
            str(row['BasePlantName']),
            str(row['Fuel']),
        )
        reduced_lookup[key] = str(row['Name'])

    model_gen_set = set(model_outage_gens)
    mustrun_bus_set = set(bus_cols)
    raw_available_cols = set(str(c) for c in raw_available.columns)

    map_records = []
    unmapped_records = []

    for _, row in raw_meta.iterrows():
        raw_name = str(row['Raw_Output_Name'])
        raw_bus = _normalize_bus_id(row['RawBusNum'])
        new_bus = raw_to_reduced_bus.get(raw_bus, raw_bus)
        plant_name = str(row['PlantNames'])
        fuel = str(row['Fuel'])
        max_cap = float(row['Max_Cap'])

        if raw_name not in raw_available_cols:
            unmapped_records.append({
                'Raw_Output_Name': raw_name,
                'RawBusNum': raw_bus,
                'NewBusNum': new_bus,
                'PlantNames': plant_name,
                'Fuel': fuel,
                'Reason': 'raw available-capacity column not found',
            })
            continue

        target_type = None
        target_name = None
        reason = 'mapped'

        if fuel == 'NUC (Nuclear)':
            target_type = 'mustrun_bus'
            target_name = f'bus_{new_bus}'
            if target_name not in mustrun_bus_set:
                reason = 'reduced nuclear bus not in model bus set'
                target_name = None
        elif fuel == 'DFO (Distillate Fuel Oil)':
            target_type = 'oil_node'
            target_name = f'bus_{new_bus}_oil'
            if target_name not in model_gen_set:
                reason = 'final node-level oil generator not found'
                target_name = None
        elif fuel in {'NG (Natural Gas)', 'BIT (Bituminous Coal)'}:
            target_type = 'thermal_generator'
            target_name = reduced_lookup.get((new_bus, plant_name, fuel))
            if target_name is None:
                reason = 'reduced thermal generator group not found'
            elif target_name not in model_gen_set:
                reason = 'reduced thermal generator not in final model outage set'
                target_name = None
        else:
            reason = 'fuel not represented in outage/mustrun model'

        if target_name is None:
            unmapped_records.append({
                'Raw_Output_Name': raw_name,
                'RawBusNum': raw_bus,
                'NewBusNum': new_bus,
                'PlantNames': plant_name,
                'Fuel': fuel,
                'Reason': reason,
            })
            continue

        if target_type == 'mustrun_bus':
            mustrun_limits[target_name] += raw_available[raw_name]
        else:
            gen_limits[target_name] += raw_available[raw_name]

        map_records.append({
            'Raw_Output_Name': raw_name,
            'RawBusNum': raw_bus,
            'NewBusNum': new_bus,
            'PlantNames': plant_name,
            'Fuel': fuel,
            'Raw_Max_Cap': max_cap,
            'SourceRowCount': row.get('SourceRowCount', 1),
            'RawRowID': row.get('RawRowID', row.get('SourceRawRowIDs', '')),
            'Target_Type': target_type,
            'Target_Name': target_name,
        })

    # Clip generator limits to final model maxcap to protect against any duplicate aggregation.
    final_cap = final_genparams.set_index('name')['maxcap'].apply(pd.to_numeric, errors='coerce')
    final_cap = final_cap.reindex(model_outage_gens).fillna(np.inf)
    gen_limits = gen_limits.clip(lower=0.0, upper=final_cap, axis=1)

    # The raw available-capacity series is already capped by raw generator MWMax. Keep only nonnegative values.
    mustrun_limits = mustrun_limits.clip(lower=0.0)

    # Write model-ready hourly limit files.
    gen_limits_out = gen_limits.copy()
    gen_limits_out.insert(0, 'Hour', hours.values)
    mustrun_limits_out = mustrun_limits.copy()
    mustrun_limits_out.insert(0, 'Hour', hours.values)

    fn_gen_limits = f'HorizonGenLimits_base_{NN}_y_{year}.csv'
    fn_mustrun_limits = f'HorizonMustrunLimits_base_{NN}_y_{year}.csv'
    full_gen_limits = _save_csv(gen_limits_out, fn_gen_limits, index=False)
    full_mustrun_limits = _save_csv(mustrun_limits_out, fn_mustrun_limits, index=False)

    # Also save standardized copies of raw inputs for traceability.
    raw_meta_copy = os.path.join(data_allocation_dir, 'east_rawGens_metadata.csv')
    raw_avail_copy = os.path.join(data_allocation_dir, f'east{year}_rawGensAvailableCap.csv')
    if os.path.abspath(raw_metadata_path) != os.path.abspath(raw_meta_copy):
        copy(raw_metadata_path, raw_meta_copy)
    if os.path.abspath(raw_available_path) != os.path.abspath(raw_avail_copy):
        copy(raw_available_path, raw_avail_copy)

    map_df = pd.DataFrame(map_records)
    unmapped_df = pd.DataFrame(unmapped_records)

    # Strict model-coverage check: every final outage generator in EIC_simple.py
    # should receive at least one raw generator group. Otherwise its hourly limit
    # would remain zero for the full year, which would silently remove capacity.
    mapped_model_targets = set()
    if not map_df.empty and 'Target_Name' in map_df.columns:
        mapped_model_targets = set(
            map_df.loc[map_df['Target_Type'].isin(['thermal_generator', 'oil_node']), 'Target_Name']
            .astype(str)
            .tolist()
        )
    missing_model_gens = sorted(set(model_outage_gens) - mapped_model_targets)
    missing_model_gens_df = pd.DataFrame({'missing_model_outage_generator': missing_model_gens})

    fn_map = f'raw_to_reduced_outage_map_{NN}_y_{year}.csv'
    fn_unmapped = f'raw_to_reduced_outage_unmapped_{NN}_y_{year}.csv'
    fn_missing_model = f'raw_to_reduced_outage_missing_model_gens_{NN}_y_{year}.csv'
    full_map = _save_csv(map_df, fn_map, index=False)
    full_unmapped = _save_csv(unmapped_df, fn_unmapped, index=False)
    full_missing_model = _save_csv(missing_model_gens_df, fn_missing_model, index=False)

    print('\n========== Raw-to-reduced outage aggregation diagnostic ==========' )
    print(f'Year: {year}')
    print(f"Raw metadata rows: {consistency_summary['Metadata_Rows']}")
    print(f"Unique Raw_Output_Name values: {consistency_summary['Unique_Raw_Output_Name']}")
    print(f'Raw available-capacity columns: {len(raw_available.columns)}')
    print(f'Saved metadata/available consistency check: {full_consistency}')
    print(f'Mapped raw generator rows: {len(map_df)}')
    print(f'Unmapped raw generator rows: {len(unmapped_df)}')
    print(f'Model outage generators: {len(model_outage_gens)}')
    print(f'Model buses in must-run limit file: {len(bus_cols)}')
    print(f'Saved: {full_gen_limits}')
    print(f'Saved: {full_mustrun_limits}')
    print(f'Saved mapping: {full_map}')
    if len(unmapped_df) > 0:
        print(f'Saved unmapped records: {full_unmapped}')
        print('Examples of unmapped records:')
        print(unmapped_df.head(10).to_string(index=False))

    if missing_model_gens:
        print(f'Saved missing model-generator records: {full_missing_model}')
        print('Examples of missing model outage generators:')
        print(missing_model_gens_df.head(10).to_string(index=False))
        raise ValueError(
            f"{len(missing_model_gens)} final model outage generators did not receive "
            f"any raw available-capacity mapping. See {full_missing_model}."
        )

    return {
        'gen_limits_path': full_gen_limits,
        'mustrun_limits_path': full_mustrun_limits,
        'mapping_path': full_map,
        'unmapped_path': full_unmapped,
        'missing_model_gens_path': full_missing_model,
    }

# Static references
#df_load = pd.read_csv('BA_load_corrected.csv',header=0, index_col=0)
df_BAs = pd.read_csv(os.path.join(base_dir, 'Interconnections/BAs_full.csv'), header=0)
BAs = list(df_BAs['Name'])

df_full = pd.read_csv(os.path.join(base_dir, 'Interconnections/nodes_to_BA_state.csv'), header=0,index_col=0)
df_full = df_full.reset_index(drop=True)
full_available = list(df_full['Number'])


#df_wind = pd.read_csv('BA_wind.csv',header=0,index_col=0)
#df_solar = pd.read_csv('BA_solar_corrected.csv',header=0,index_col=0)
df_hydro = pd.read_csv(os.path.join(base_dir, 'Gen/BA_hydro_corrected.csv'), header=0,index_col=0)


def _build_transmission_cases(trans_p, trans_MW):
    """Create explicit transmission cases for percent and additive-MW runs."""
    cases = []

    for pct in trans_p:
        cases.append({
            'mode': 'pct',
            'value': pct,
            'tag': f'tp_{pct}',
            'folder_token': str(pct),
            'print_label': f'Tp={pct}%',
        })

    for delta_mw in trans_MW:
        cases.append({
            'mode': 'MW',
            'value': delta_mw,
            'tag': f'MW_{delta_mw}',
            'folder_token': f'MW_{delta_mw}',
            'print_label': f'delta_MW={delta_mw}',
        })

    if not cases:
        raise ValueError(
            "No transmission cases specified. Add values to trans_p and/or trans_MW."
        )

    return cases


# Use the years for which generator-level lost-capacity files exist.
#years = range(2019, 2023)
years = range(1980, 2020)

for NN in NODE_NUMBER:
    FN = 'reduced_network/Results_' + str(NN) + '.xlsx'

    # Pre-read NN-specific sheets used many times
    df_selected = pd.read_excel(os.path.join(base_dir, FN), sheet_name='Bus', header=0)
    buses = list(df_selected['bus_i'])

    # Map selected nodes to BAs (NN-specific)
    selected_BAs = []
    for b in buses:
        BA = df_full.loc[df_full['Number']==b,'NAME']
        BA = BA.reset_index(drop=True)
        selected_BAs.append(BA[0])
    df_selected['BA'] = selected_BAs

    # BA totals & load weights depend only on NN; compute once
    BA_totals = []
    for b in BAs:
        sample = list(df_selected.loc[df_selected['BA']==b,'Pd'])
        corrected = [0 if x<0 else x for x in sample]
        BA_totals.append(sum(corrected))
    BA_totals = np.column_stack((BAs,BA_totals))
    df_BA_totals_load = pd.DataFrame(BA_totals, columns=['Name','Total'])

    weights = []
    for i in range(0,len(df_selected)):
        area = df_selected.loc[i,'BA']
        if df_selected.loc[i,'Pd'] <0:
            weights.append(0)
        else:
            X = float(df_BA_totals_load.loc[df_BA_totals_load['Name']==area,'Total'].iloc[0])
            W = (df_selected.loc[i,'Pd']/X)
            weights.append(W)
    df_selected['BA Load Weight'] = weights

    # Parse reduction summary once (NN-specific) for merged nodes
    df_summary = pd.read_excel(os.path.join(base_dir, FN), sheet_name='Summary', header=6)
    N = []
    merged = {}
    for i in range(0,len(df_summary)):
        test = df_summary.iloc[i,0]
        res = [int(i) for i in test.split() if i.isdigit()]
        if res[1] not in N:
            N.append(res[1])
    for n in N:
        k = []
        for i in range(0,len(df_summary)):
            test = df_summary.iloc[i,0]
            res = [int(i) for i in test.split() if i.isdigit()]
            if res[1] == n:
                k.append(res[0])
        merged[n] = k

    # ==========================
    # THERMAL GENS (NN-only)
    # ==========================
    import re
    df_gens = pd.read_csv(os.path.join(base_dir, 'Gen/Generators_EIA.csv'), header=0)
    df_gens = df_gens.replace('', np.nan, regex=True)
    df_gens_heat_rate = pd.read_csv(os.path.join(base_dir, 'Gen/Heat_rates_EIA.csv'), header=0)

    old_bus_num, new_bus_num, NB = [], [], []
    old_bus_num_hr, new_bus_num_hr, NB_hr = [], [], []
    for n in N:
        k = merged[n]
        for s in k:
            old_bus_num.append(s)
            new_bus_num.append(n)
    for i in range(0,len(df_gens)):
        OB = df_gens.loc[i,'BusNum']
        NB.append(new_bus_num[old_bus_num.index(OB)] if OB in old_bus_num else OB)
    df_gens['NewBusNum'] = NB

    # Raw BusNum -> reduced NewBusNum map used later to aggregate raw outage-adjusted
    # available capacity into the reduced-network model generators.
    raw_to_reduced_bus = {
        _normalize_bus_id(row.BusNum): _normalize_bus_id(row.NewBusNum)
        for row in df_gens[['BusNum', 'NewBusNum']].drop_duplicates().itertuples(index=False)
    }

    for i in range(0,len(df_gens_heat_rate)):
        OB = df_gens_heat_rate.loc[i,'BusNum']
        NB_hr.append(new_bus_num[old_bus_num.index(OB)] if OB in old_bus_num else OB)
    df_gens_heat_rate['NewBusNum'] = NB_hr

    names = list(df_gens['BusName'])
    fts = list(df_gens['FuelType'])
    names_hr = list(df_gens_heat_rate['BusName'])
    fts_hr = list(df_gens_heat_rate['BusName'])
    bus_area = list(df_gens['BusAreaName'])
    bus_area_hr = list(df_gens_heat_rate['AreaName'])

    # sanitize names
    # IMPORTANT: use enumerate rather than names.index(n_). BusName can repeat;
    # list.index() would always return the first duplicate and corrupt PlantNames
    # for later duplicate rows, which then breaks outage/available-capacity mapping.
    for i, n_ in enumerate(names):
        corrected = re.sub(r'[^A-Z]', r'', str(n_))
        f = fts[i]
        bn = str(bus_area[i]).replace(" ", "_")
        if f == 'NUC (Nuclear)': f = 'Nuc'
        elif f == 'NG (Natural Gas)': f = 'NG'
        elif f == 'BIT (Bituminous Coal)': f = 'C'
        elif f == 'SUN (Solar)': f = 'S'
        elif f == 'WAT (Water)': f = 'H'
        elif f == 'WND (Wind)': f = 'W'
        elif f == 'DFO (Distillate Fuel Oil)': f = 'O'
        names[i] = corrected + '_' + f + '_' + bn
    for i, n_ in enumerate(names_hr):
        corrected = re.sub(r'[^A-Z]', r'', str(n_))
        f = fts_hr[i]
        bn = str(bus_area_hr[i]).replace(" ", "_")
        if f == 'NUC (Nuclear)': f = 'Nuc'
        elif f == 'NG (Natural Gas)': f = 'NG'
        elif f == 'BIT (Bituminous Coal)': f = 'C'
        elif f == 'SUN (Solar)': f = 'S'
        elif f == 'WAT (Water)': f = 'H'
        elif f == 'WND (Wind)': f = 'W'
        elif f == 'DFO (Distillate Fuel Oil)': f = 'O'
        names_hr[i] = corrected + '_' + f + '_' + bn

    df_gens['PlantNames'] = names
    df_gens_heat_rate['PlantNames'] = names_hr

    NB_unique = df_gens['NewBusNum'].unique()
    plants, plant_bases, caps, mw_min, nbs, heat_rate, f = [], [], [], [], [], [], []
    count = 2
    thermal = ['NG (Natural Gas)','NUC (Nuclear)','BIT (Bituminous Coal)','DFO (Distillate Fuel Oil)']

    for n_ in NB_unique:
        sample = df_gens.loc[df_gens['NewBusNum'] == n_]
        sublist = sample['PlantNames'].unique()
        for s in sublist:
            fuel = list(sample.loc[sample['PlantNames']==s,'FuelType'])
            if fuel[0] in thermal:
                c = sum(sample.loc[sample['PlantNames']==s,'MWMax'].values)
                hr = np.nanmean(sample.loc[sample['PlantNames']==s,'Heat Rate MBTU/MWh'].values)
                if hr == np.nan or hr == 0 or hr == 'nan' or hr == '':
                    hr = np.nanmean(df_gens.loc[df_gens['FuelType']==fuel[0],'Heat Rate MBTU/MWh'].values)
                mn = sum(sample.loc[sample['PlantNames']==s,'MWMin'].values)
                plant_bases.append(s)
                mw_min.append(mn)
                caps.append(c)
                nbs.append(n_)
                heat_rate.append(hr)
                f.append(fuel[0])
                if s in plants:
                    plants.append(s + '_' + str(count)); count += 1
                else:
                    plants.append(s)

    C=np.column_stack((plants,nbs,f,caps,mw_min,heat_rate))
    df_C = pd.DataFrame(C, columns=['Name','Bus','Fuel','Max_Cap','Min_Cap','Heat_Rate'])
    fn_thermal = f'thermal_gens_{NN}.csv'
    full_thermal_path = _save_csv(df_C, fn_thermal, index=False)

    # Keep a reduced thermal mapping table with the unsuffixed plant name.
    # This table is used to aggregate raw/pre-reduction outage-adjusted capacity
    # to the reduced-network generator names without relying on one-to-one matches.
    df_reduced_thermal_map = df_C.copy()
    df_reduced_thermal_map['BasePlantName'] = plant_bases
    fn_thermal_map = f'thermal_gens_mapping_{NN}.csv'
    full_thermal_map_path = _save_csv(df_reduced_thermal_map, fn_thermal_map, index=False)


# =============================================================================
# 
#     # ==========================
#     # FUEL PRICES (NN-only)
#     # ==========================
#     # Build bus-level price tables
#     NG_price = pd.read_csv(os.path.join(base_dir, 'NG_price/Average_NG_prices_BAs.csv'), header=0)
#     Fuel_buses = ['bus_' + str(b) for b in buses]
#     # NG by BA per bus
#     NG_prices_all = None
#     for bus in buses:
#         selected_node_BA = df_full.loc[df_full['Number']==bus,'NAME'].values[0]
#         specific_node_NG_price = NG_price.loc[:,selected_node_BA].copy()
#         NG_prices_all = specific_node_NG_price.copy() if NG_prices_all is None else pd.concat([NG_prices_all, specific_node_NG_price], axis=1)
#     if NG_prices_all is None:
#         NG_prices_all = pd.DataFrame(columns=Fuel_buses)
#     NG_prices_all.columns = Fuel_buses
#     # Coal by state per bus
#     Coal_price = pd.read_csv(os.path.join(base_dir, 'Coal_price/coal_prices_state.csv'), header=0)
#     Coal_prices_all = None
#     for bus in buses:
#         selected_node_state = df_full.loc[df_full['Number']==bus,'STATE'].values[0]
#         if selected_node_state == 'NB': selected_node_state = 'ME'
#         elif selected_node_state == 'CO': selected_node_state = 'KS'
#         elif selected_node_state == 'OR': selected_node_state = 'AR'
#         elif selected_node_state == 'TX': selected_node_state = 'OK'
#         elif selected_node_state == 'NM': selected_node_state = 'OK'
#         elif selected_node_state == 'MT': selected_node_state = 'SD'
#         specific_node_coal_price = Coal_price.loc[:,selected_node_state].copy()
#         Coal_prices_all = specific_node_coal_price.copy() if Coal_prices_all is None else pd.concat([Coal_prices_all, specific_node_coal_price], axis=1)
#     if Coal_prices_all is None:
#         Coal_prices_all = pd.DataFrame(columns=Fuel_buses)
#     Coal_prices_all.columns = Fuel_buses
#     Oil_prices_all = pd.DataFrame({'all': np.reshape(np.ones((365,1))*20,(365,))})
# 
#     # generator-based fuel prices
#     # NOTE: genparams (used below) are year-dependent because RES capacity checks use nodal RES profiles.
#     #       Fuel prices themselves do not vary by year in this script, so we compute once per NN using buses.
#     # We'll create the mapping columns now but fill columns later when gen list is ready per year.
# 
# 
# =============================================================================


    # ==========================
    # HYDRO (NN-only; not year-specific)
    # ==========================
    # Build MWMax/FuelType mapping once
    df_gen_all = pd.read_csv(os.path.join(base_dir, 'Gen/Generators_EIA.csv'), header=0)
    MWMax = []
    fuel_type = []
    nums = list(df_gen_all['BusNum'])
    for i in range(0,len(df_full)):
        bus = df_full.loc[i,'Number']
        if bus in nums:
            MWMax.append(df_gen_all.loc[df_gen_all['BusNum']==bus,'MWMax'].values[0])
            fuel_type.append(df_gen_all.loc[df_gen_all['BusNum']==bus,'FuelType'].values[0])
        else:
            MWMax.append(0)
            fuel_type.append('none')
    df_full['MWMax'] = MWMax
    df_full['FuelType'] = fuel_type

    # BA totals (hydro) and weights once per NN
    BA_totals_h = []
    for b in BAs:
        sample = list(df_full.loc[(df_full['NAME']==b) & (df_full['FuelType'] == 'WAT (Water)'),'MWMax'])
        BA_totals_h.append(sum(sample))
    df_BA_totals_h = pd.DataFrame(np.column_stack((BAs,BA_totals_h)), columns=['Name','Total'])

    weights_h = []
    for i in range(0,len(df_full)):
        area = df_full.loc[i,'NAME']
        if str(area) in BAs and str(df_full.loc[i,'FuelType']) == 'WAT (Water)':
            X = float(df_BA_totals_h.loc[df_BA_totals_h['Name']==area,'Total'].iloc[0])
            W = (df_full.loc[i,'MWMax']/X) if X != 0 else 0
            weights_h.append(W)
        else:
            weights_h.append(0)
    df_full['BA Hydro Weight'] = weights_h

    # Nodal hydro profile (8760 x buses) using BA hydro timeseries
    buses_local = list(df_selected['bus_i'])
    T = np.zeros((8760,len(buses_local)))
    idx = 0
    for b in buses_local:
        sample = df_full.loc[df_full['Number'] == b].reset_index(drop=True)
        name = sample['NAME'][0]
        if str(name) in BAs and float(df_BA_totals_h.loc[df_BA_totals_h['Name']==str(name),'Total'].values[0]) >= 1:
            abbr = df_BAs.loc[df_BAs['Name']==name,'Abbreviation'].values[0]
            weight = sample['BA Hydro Weight'].values[0]
            T[:,idx] += np.reshape(df_hydro[abbr].values*weight,(8760,))
        try:
            m_nodes = merged[b]
            for m in m_nodes:
                sample = df_full.loc[df_full['Number'] == m].reset_index(drop=True)
                name = sample['NAME'][0]
                if str(name) in BAs and float(df_BA_totals_h.loc[df_BA_totals_h['Name']==str(name),'Total'].values[0]) >= 1:
                    abbr = df_BAs.loc[df_BAs['Name']==name,'Abbreviation'].values[0]
                    weight = sample['BA Hydro Weight']
                    T[:,idx] += np.reshape(df_hydro[abbr].values*weight.values[0],(8760,))
        except KeyError:
            pass
        idx += 1
    h_buses = ['bus_' + str(b) for b in buses_local]

    # Hourly nodal hydro profile. This is NN-specific but not year-specific.
    # Shape: 8760 x reduced buses. Units follow BA_hydro_corrected.csv, normally MW.
    df_hydro_hourly = pd.DataFrame(T, columns=h_buses)
    fn_hydro_hourly = f'nodal_hydro_hourly_{NN}.csv'
    full_hydro_hourly_path = _save_csv(df_hydro_hourly, fn_hydro_hourly)

    # Daily nodal hydro data passed directly to wrapper_simple.
    # Shape: 365 x reduced buses. Values are daily sums of the hourly profile.
    df_hydro_daily = df_hydro_hourly.groupby(np.arange(len(df_hydro_hourly)) // 24).sum()
    fn_hydro_daily = f'nodal_hydro_daily_{NN}.csv'
    full_hydro_daily_path = _save_csv(df_hydro_daily, fn_hydro_daily)

    # precompute which reduced buses actually have wind/solar capacity based on the generator database and merging

    # After HYDRO block, before the transmission loop
    
    buses_reduced = list(df_selected['bus_i'])  # reduced-network buses
    
    # Use the generator database you already loaded as df_gen_all
    has_wind  = {b: False for b in buses_reduced}
    has_solar = {b: False for b in buses_reduced}
    
    for b in buses_reduced:
        # Original buses merged into this reduced bus
        orig_nodes = [b] + merged.get(b, [])
        gens_here = df_gen_all[df_gen_all['BusNum'].isin(orig_nodes)]
        if gens_here.empty:
            continue
        fuels_here = gens_here['FuelType'].unique()
        if 'WND (Wind)' in fuels_here:
            has_wind[b] = True
        if 'SUN (Solar)' in fuels_here:
            has_solar[b] = True

    # Outage-adjusted capacity limits are NN/year-specific and do not depend on
    # transmission expansion cases. Track completed NN/year pairs so we do not
    # repeatedly regenerate identical files inside the transmission-case loop.
    #
    # Key idea:
    #   - line_params_* and line_to_bus_* depend on trans_p / trans_MW.
    #   - HorizonGenLimits_base_* and HorizonMustrunLimits_base_* do NOT.
    #
    # Store the output paths in a dict so later transmission cases can explicitly
    # reuse the same base files.
    outage_limits_by_year = {}

    # ==========================
    # TRANSMISSION (NN, case)
    # ==========================
    for trans_case in _build_transmission_cases(trans_p, trans_MW):
        # Transmission matrices do not depend on year.
        # Each case is either percent expansion or additive-MW expansion.
        trans_mode = trans_case['mode']
        trans_value = trans_case['value']
        trans_tag = trans_case['tag']
        trans_folder_token = trans_case['folder_token']
        trans_print_label = trans_case['print_label']

        df_branch = pd.read_excel(os.path.join(base_dir, FN), sheet_name='Branch', header=0)

        # Eliminate repeated directed branches using the original East-script behavior.
        df = df_branch.copy()
        lines = []
        repeats = []
        index = []
        for i in range(0, len(df)):
            t = tuple((df.loc[i, 'fbus'], df.loc[i, 'tbus']))

            if t in lines:
                df = df.drop([i])
                repeats.append(t)
                r = lines.index(t)
                ii = index[r]
                df.loc[ii, 'rateA'] += df.loc[ii, 'rateA']
            else:
                lines.append(t)
                index.append(i)

        df = df.reset_index(drop=True)

        sources = df.loc[:,'fbus']
        sinks = df.loc[:,'tbus']
        combined = np.append(sources, sinks)
        df_combined = pd.DataFrame(combined,columns=['node'])
        unique_nodes = df_combined['node'].unique()
        unique_nodes.sort()

        A = np.zeros((len(df),len(unique_nodes)))
        df_line_to_bus = pd.DataFrame(A)
        df_line_to_bus.columns = unique_nodes

        negative = []
        positive = []
        lines_lbl = []
        ref_node = 0
        reactance = []
        limit = []

        for i in range(0,len(df)):
            s = df.loc[i,'fbus']
            k = df.loc[i,'tbus']
            line = str(s) + '_' + str(k)
            if s == df.loc[0,'fbus']:
                lines_lbl.append(line)
                positive.append(s)
                negative.append(k)
                df_line_to_bus.loc[ref_node,s] = 1
                df_line_to_bus.loc[ref_node,k] = -1
                reactance.append(df.loc[i,'x'])
                base_MW = (1/df.loc[i,'x'])*100
                if trans_mode == 'pct':
                    MW = base_MW * (1 + trans_value/100)
                elif trans_mode == 'MW':
                    MW = base_MW + trans_value
                else:
                    raise ValueError(f"Unknown transmission mode: {trans_mode}")
                limit.append(MW)
                ref_node += 1
            elif k == df.loc[0,'fbus']:
                lines_lbl.append(line)
                positive.append(k)
                negative.append(s)
                df_line_to_bus.loc[ref_node,k] = 1
                df_line_to_bus.loc[ref_node,s] = -1
                reactance.append(df.loc[i,'x'])
                base_MW = (1/df.loc[i,'x'])*100
                if trans_mode == 'pct':
                    MW = base_MW * (1 + trans_value/100)
                elif trans_mode == 'MW':
                    MW = base_MW + trans_value
                else:
                    raise ValueError(f"Unknown transmission mode: {trans_mode}")
                limit.append(MW)
                ref_node += 1

        for i in range(0,len(df)):
            s = df.loc[i,'fbus']
            k = df.loc[i,'tbus']
            line = str(s) + '_' + str(k)
            if s != df.loc[0,'fbus'] and k != df.loc[0,'fbus']:
                lines_lbl.append(line)
                if s in positive and k in negative:
                    df_line_to_bus.loc[ref_node,s] = 1
                    df_line_to_bus.loc[ref_node,k] = -1
                elif k in positive and s in negative:
                    df_line_to_bus.loc[ref_node,k] = 1
                    df_line_to_bus.loc[ref_node,s] = -1
                elif s in positive and k in positive:
                    df_line_to_bus.loc[ref_node,s] = 1
                    df_line_to_bus.loc[ref_node,k] = -1
                elif s in negative and k in negative:
                    df_line_to_bus.loc[ref_node,s] = 1
                    df_line_to_bus.loc[ref_node,k] = -1
                elif s in positive:
                    df_line_to_bus.loc[ref_node,s] = 1
                    df_line_to_bus.loc[ref_node,k] = -1
                    negative.append(k)
                elif s in negative:
                    df_line_to_bus.loc[ref_node,k] = 1
                    df_line_to_bus.loc[ref_node,s] = -1
                    positive.append(k)
                elif k in positive:
                    df_line_to_bus.loc[ref_node,k] = 1
                    df_line_to_bus.loc[ref_node,s] = -1
                    negative.append(s)
                elif k in negative:
                    df_line_to_bus.loc[ref_node,s] = 1
                    df_line_to_bus.loc[ref_node,k] = -1
                    positive.append(s)
                else:
                    positive.append(s)
                    negative.append(k)
                    df_line_to_bus.loc[ref_node,s] = 1
                    df_line_to_bus.loc[ref_node,k] = -1
                reactance.append(df.loc[i,'x'])
                base_MW = (1/df.loc[i,'x'])*100
                if trans_mode == 'pct':
                    MW = base_MW * (1 + trans_value/100)
                elif trans_mode == 'MW':
                    MW = base_MW + trans_value
                else:
                    raise ValueError(f"Unknown transmission mode: {trans_mode}")
                limit.append(MW)
                ref_node += 1

        unique_nodes = list(unique_nodes)
        unique_nodes_lbl = ['bus_' + str(v) for v in unique_nodes]
        df_line_to_bus.columns = unique_nodes_lbl
        for i in range(0,len(lines_lbl)):
            lines_lbl[i] = 'line_' + lines_lbl[i]
        df_line_to_bus['line'] = lines_lbl
        df_line_to_bus.set_index('line', inplace=True)
        
        # FIX: keep 'line' as an explicit first column in the CSV
        df_line_to_bus_out = df_line_to_bus.reset_index()  # 'line' becomes first column
        
        fn_line_to_bus = f'line_to_bus_{NN}_{trans_tag}.csv'
        full_line_to_bus_path = _save_csv(df_line_to_bus_out, fn_line_to_bus, index=False)

        df_line_params = pd.DataFrame()
        df_line_params['line'] = lines_lbl
        df_line_params['reactance'] = reactance
        df_line_params['limit'] = limit

        min_limit = min(limit)
        if min_limit <= 0:
            bad_idx = [i for i, v in enumerate(limit) if v <= 0]
            bad_lines = [lines_lbl[i] for i in bad_idx[:10]]
            raise ValueError(
                f"Non-positive retained line limits detected for {trans_print_label}. "
                f"Minimum limit = {min_limit:.3f} MW. "
                f"Example offending lines: {bad_lines}. "
                f"Use a less negative MW addition or a larger percent expansion."
            )

        fn_line_params = f'line_params_{NN}_{trans_tag}.csv'
        full_line_params_path = _save_csv(df_line_params, fn_line_params, index=False)

        # ==========================
        # UC treatment loop (behavior unchanged)
        # ==========================
        for UC in UC_TREATMENTS:
            for y in years:
                # Pre-load annual BA profiles once per y
                y_string = str(y)
                fn_load = 'Load/TELL_BA_load_corrected_' + y_string + '.csv'
#                fn_load = 'Load/EIA_BA_load_corrected_' + y_string + '.csv'
                df_load = pd.read_csv(os.path.join(base_dir, fn_load), header=0, index_col=0)
                df_wind = _read_csv_first_existing(
                    base_dir,
                    [
                        'Gen/reV_BA_wind_' + y_string + '.csv',
#                        'Gen/EIA_BA_wind_corrected_' + y_string + '.csv',
                    ],
                    header=0,
                    index_col=0
                )
                fn_solar = 'Gen/reV_BA_solar_corrected_' + y_string + '.csv'
#                fn_solar = 'Gen/EIA_BA_solar_corrected_' + y_string + '.csv'
                df_solar = pd.read_csv(os.path.join(base_dir, fn_solar), header=0,index_col=0)

                # clean NaNs for generators (same as before)
                for _df in (df_wind, df_solar, df_hydro):
                    a = _df.values
                    m=np.where(np.isnan(a))
                    r,c=np.shape(m)
                    for i in range(0,c):
                        _df.iloc[m[0][i],m[1][i]] = 0

                # Run folders are created later once per NN/UC/transmission-case/year.
                # The data products below are written once to Data/data_allocation.

                # -----------------
                # NODAL LOAD (NN, year)
                # -----------------
                T = np.zeros((8760,len(buses)))
                for i in range(0,len(df_selected)):
                    name = df_selected.loc[i,'BA']
                    if float(df_BA_totals_load.loc[df_BA_totals_load['Name']==str(name),'Total'].values[0]) < 1:
                        pass
                    else:
                        abbr = df_BAs.loc[df_BAs['Name']==name,'Abbreviation'].values[0]
                        weight = df_selected.loc[i,'BA Load Weight']
                        if max(df_load[abbr]) < 1:
                            T[:,i] += np.reshape(df_load[abbr].values,(8760,))
                        else:
                            T[:,i] += np.reshape(df_load[abbr].values*weight,(8760,))
                buses_lbl = ['bus_' + str(b) for b in buses]
                df_C = pd.DataFrame(T, columns=buses_lbl)
                fn_load_out = f'nodal_load_{NN}_y_{y}.csv'
                full_load_path = _save_csv(df_C, fn_load_out)
                #copy(full_load_path, path)

                # ============= WIND (per year) =============
                df_gen = pd.read_csv(os.path.join(base_dir, 'Gen/Generators_EIA.csv'), header=0)
                MWMax = []
                fuel_type = []
                nums = list(df_gen['BusNum'])
                for i in range(0,len(df_full)):
                    bus = df_full.loc[i,'Number']
                    if bus in nums:
                        MWMax.append(df_gen.loc[df_gen['BusNum']==bus,'MWMax'].values[0])
                        fuel_type.append(df_gen.loc[df_gen['BusNum']==bus,'FuelType'].values[0])
                    else:
                        MWMax.append(0)
                        fuel_type.append('none')
                df_full['MWMax'] = MWMax
                df_full['FuelType'] = fuel_type

                BA_totals = []
                for b in BAs:
                    sample = list(df_full.loc[(df_full['NAME']==b) & (df_full['FuelType'] == 'WND (Wind)'),'MWMax'])
                    BA_totals.append(sum(sample))
                df_BA_totals = pd.DataFrame(np.column_stack((BAs,BA_totals)), columns=['Name','Total'])

                weights = []
                for i in range(0,len(df_full)):
                    area = df_full.loc[i,'NAME']
                    if str(area) in BAs and str(df_full.loc[i,'FuelType']) == 'WND (Wind)':
                        X = float(df_BA_totals.loc[df_BA_totals['Name']==area,'Total'].iloc[0])
                        W = df_full.loc[i,'MWMax']/X
                        weights.append(W)
                    else:
                        weights.append(0)
                df_full['BA Wind Weight'] = weights

                buses = list(df_selected['bus_i'])
                T = np.zeros((8760,len(buses)))
                BA_sums = np.zeros((len(BAs),1))
                BA_test = np.zeros((len(BAs),1))

                idx = 0
                for b in buses:
                    sample = df_full.loc[df_full['Number'] == b].reset_index(drop=True)
                    name = sample['NAME'][0]
                    if str(name) in BAs and float(df_BA_totals.loc[df_BA_totals['Name']==str(name),'Total'].values[0]) >= 1:
                        abbr = df_BAs.loc[df_BAs['Name']==name,'Abbreviation'].values[0]
                        weight = sample['BA Wind Weight'].values[0]
                        T[:,idx] += np.reshape(df_wind[abbr].values*weight,(8760,))
                        dx = BAs.index(name)
                        BA_sums[dx] += weight
                        BA_test[dx] += sum(df_wind[abbr].values*weight)
                    try:
                        m_nodes = merged[b]
                        for m in m_nodes:
                            sample = df_full.loc[df_full['Number'] == m].reset_index(drop=True)
                            name = sample['NAME'][0]
                            if str(name) in BAs and float(df_BA_totals.loc[df_BA_totals['Name']==str(name),'Total'].values[0]) >= 1:
                                abbr = df_BAs.loc[df_BAs['Name']==name,'Abbreviation'].values[0]
                                weight = sample['BA Wind Weight']
                                dx = BAs.index(name)
                                BA_sums[dx] += weight
                                BA_test[dx] += sum(df_wind[abbr].values*weight.values[0])
                                T[:,idx] += np.reshape(df_wind[abbr].values*weight.values[0],(8760,))
                    except KeyError:
                        pass
                    idx += 1
                w_buses = ['bus_' + str(b) for b in buses]
                df_C = pd.DataFrame(T, columns=w_buses)
                fn_wind_out = f'nodal_wind_{NN}_y_{y}.csv'
                full_wind_path = _save_csv(df_C, fn_wind_out)
                #copy(full_wind_path, path)

                # ============= SOLAR (per year) =============
                BA_totals = []
                for b in BAs:
                    sample = list(df_full.loc[(df_full['NAME']==b) & (df_full['FuelType'] == 'SUN (Solar)'),'MWMax'])
                    BA_totals.append(sum(sample))
                df_BA_totals = pd.DataFrame(np.column_stack((BAs,BA_totals)), columns=['Name','Total'])

                weights = []
                for i in range(0,len(df_full)):
                    area = df_full.loc[i,'NAME']
                    if str(area) in BAs and str(df_full.loc[i,'FuelType']) == 'SUN (Solar)':
                        X = float(df_BA_totals.loc[df_BA_totals['Name']==area,'Total'].iloc[0])
                        W = df_full.loc[i,'MWMax']/X
                        weights.append(W)
                    else:
                        weights.append(0)
                df_full['BA Solar Weight'] = weights

                T = np.zeros((8760,len(buses)))
                BA_sums = np.zeros((len(BAs),1))

                idx = 0
                for b in buses:
                    sample = df_full.loc[df_full['Number'] == b].reset_index(drop=True)
                    name = sample['NAME'][0]
                    if str(name) in BAs and float(df_BA_totals.loc[df_BA_totals['Name']==str(name),'Total'].values[0]) >= 1:
                        abbr = df_BAs.loc[df_BAs['Name']==name,'Abbreviation'].values[0]
                        weight = sample['BA Solar Weight'].values[0]
                        T[:,idx] += np.reshape(df_solar[abbr].values*weight,(8760,))
                        dx = BAs.index(name)
                        BA_sums[dx] += weight
                    try:
                        m_nodes = merged[b]
                        for m in m_nodes:
                            sample = df_full.loc[df_full['Number'] == m].reset_index(drop=True)
                            name = sample['NAME'][0]
                            if str(name) in BAs and float(df_BA_totals.loc[df_BA_totals['Name']==str(name),'Total'].values[0]) >= 1:
                                abbr = df_BAs.loc[df_BAs['Name']==name,'Abbreviation'].values[0]
                                weight = sample['BA Solar Weight']
                                dx = BAs.index(name)
                                BA_sums[dx] += weight
                                T[:,idx] += np.reshape(df_solar[abbr].values*weight.values[0],(8760,))
                    except KeyError:
                        pass
                    idx += 1
                s_buses = ['bus_' + str(b) for b in buses]
                df_C = pd.DataFrame(T, columns=s_buses)
                fn_solar_out = f'nodal_solar_{NN}_y_{y}.csv'
                full_solar_path = _save_csv(df_C, fn_solar_out)
                #copy(full_solar_path, path)
            #########################################
            # GENERATOR PARAMS (full + aggregated oil)
                # Note: these remain written per NN (filename), same as v1; behavior preserved (overwrites across years).
                df_G = pd.read_csv(full_thermal_path,header=0)

                names = []
                typs = []
                nodes = []
                maxcaps = []
                mincaps = []
                heat_rates = []
                var_oms = []
                no_loads = []
                st_costs = []
                ramps = []
                minups = []
                mindns = []

                must_nodes = []
                must_caps = []

                for i in range(0,len(df_G)):
                    name = df_G.loc[i,'Name']
                    t = df_G.loc[i,'Fuel']
                    if t == 'NG (Natural Gas)':
                        typ = 'ngcc'
                    elif t == 'BIT (Bituminous Coal)':
                        typ = 'coal'
                    elif t == 'DFO (Distillate Fuel Oil)':
                        typ = 'oil'
                    else:
                        typ = 'nuclear'
                    node = 'bus_' + str(df_G.loc[i,'Bus'])
                    maxcap = df_G.loc[i,'Max_Cap']
                    mincap = df_G.loc[i,'Min_Cap']
                    hr_2 = df_G.loc[i,'Heat_Rate']

                    if typ == 'ngcc':
                        var_om = 3; minup = 4; mindn = 4; ramp = maxcap
                    elif typ == 'oil':
                        var_om = 8; minup = 1; mindn = 1; ramp = maxcap
                    else:
                        var_om = 4; minup = 12; mindn = 12; ramp = 0.33*maxcap
                    st_cost = 70*maxcap; no_load = 3*maxcap

                    if typ != 'nuclear':
                        names.append(name); typs.append(typ); nodes.append(node)
                        maxcaps.append(maxcap); mincaps.append(mincap)
                        var_oms.append(var_om); no_loads.append(no_load); st_costs.append(st_cost)
                        ramps.append(ramp); minups.append(minup); mindns.append(mindn); heat_rates.append(hr_2)
                    else:
                        must_nodes.append(node); must_caps.append(maxcap)

                # wind
                df_W = pd.read_csv(os.path.join(data_allocation_dir, f'nodal_wind_{NN}_y_{y}.csv'), header=0)
                for b in buses_reduced:
                    node_lbl = f'bus_{b}'
                    # We still require the column to exist in df_W, but presence is decided by has_wind
                    if node_lbl in df_W.columns and has_wind.get(b, False):
                        name = node_lbl + '_WIND'
                        maxcap = 100000
                        names.append(name); typs.append('wind'); nodes.append(node_lbl)
                        maxcaps.append(maxcap); mincaps.append(0)
                        var_oms.append(0); no_loads.append(0); st_costs.append(0)
                        ramps.append(0); minups.append(0); mindns.append(0); heat_rates.append(0)

                # solar
                df_S = pd.read_csv(os.path.join(data_allocation_dir, f'nodal_solar_{NN}_y_{y}.csv'), header=0)
                for b in buses_reduced:
                    node_lbl = f'bus_{b}'
                    if node_lbl in df_S.columns and has_solar.get(b, False):
                        name = node_lbl + '_SOLAR'
                        maxcap = 100000
                        names.append(name); typs.append('solar'); nodes.append(node_lbl)
                        maxcaps.append(maxcap); mincaps.append(0)
                        var_oms.append(0); no_loads.append(0); st_costs.append(0)
                        ramps.append(0); minups.append(0); mindns.append(0); heat_rates.append(0)

                # hydro
                df_H = pd.read_csv(full_hydro_hourly_path, header=0)
                for ncol in df_H.columns:
                    if sum(df_H[ncol]) > 0:
                        name = ncol + '_HYDRO'; maxcap = max(df_H[ncol])
                        names.append(name); typs.append('hydro'); nodes.append(ncol)
                        maxcaps.append(maxcap); mincaps.append(0)
                        var_oms.append(1); no_loads.append(1); st_costs.append(1)
                        ramps.append(maxcap); minups.append(0); mindns.append(0); heat_rates.append(0)

                df_genparams = pd.DataFrame()
                df_genparams['name'] = names
                df_genparams['typ'] = typs
                df_genparams['node'] = nodes
                df_genparams['maxcap'] = maxcaps
                df_genparams['heat_rate'] = heat_rates
                df_genparams['mincap'] = mincaps
                df_genparams['var_om'] = var_oms
                df_genparams['no_load'] = no_loads
                df_genparams['st_cost'] = st_costs
                df_genparams['ramp'] = ramps
                df_genparams['minup'] = minups
                df_genparams['mindn'] = mindns

                fn_genparams_full = f'data_genparams_full_{NN}.csv'
                full_genparams_full_path = _save_csv(df_genparams, fn_genparams_full, index=False)
                #copy(full_genparams_full_path, path)

                df_must = pd.DataFrame()
                for i in range(0,len(must_nodes)):
                    ncol = must_nodes[i]
                    df_must[ncol] = [must_caps[i]]
                fn_must = f'must_run_{NN}.csv'
                full_must_path = _save_csv(df_must, fn_must, index=False)
                #copy(full_must_path, path)

                # Gen-to-bus matrix (full)
                df = pd.read_csv(full_genparams_full_path,header=0)
                gens = list(df.loc[:,'name'])
                df_nodes = pd.read_excel(os.path.join(base_dir, FN), sheet_name = 'Bus', header=0)
                all_nodes = list(df_nodes['bus_i'])
                all_nodes = ['bus_' + str(v) for v in all_nodes]
                A = np.zeros((len(gens),len(all_nodes)))
                df_A = pd.DataFrame(A, columns=all_nodes)
                df_A['name'] = gens
                df_A.set_index('name',inplace=True)
                for i in range(0,len(gens)):
                    node = df.loc[i,'node']
                    g = gens[i]
                    df_A.loc[g,node] = 1
                fn_gen_mat_full = f'gen_mat_full_{NN}.csv'
                full_gen_mat_full_path = _save_csv(df_A, fn_gen_mat_full, index=True)
                #copy(full_gen_mat_full_path, path)

                # Aggregate oil on nodes and rewrite params (same logic)
                df = pd.read_csv(full_genparams_full_path,header=0)
                df_oil = df.loc[df['typ'] == 'oil'].reset_index(drop=True)
                gens = list(df.loc[:,'name'])
                gens_oil = list(df_oil.loc[:,'name'])
                gens_cap = list(df.loc[:,'maxcap'])
                gens_hr = list(df.loc[:,'heat_rate'])
                gens_min = list(df.loc[:,'mincap'])
                gens_var_om = list(df.loc[:,'var_om'])
                gens_no_load = list(df.loc[:,'no_load'])
                gens_st_cost = list(df.loc[:,'st_cost'])
                gens_ramp = list(df.loc[:,'ramp'])

                df_nodes = pd.read_excel(os.path.join(base_dir, FN), sheet_name = 'Bus', header=0)
                all_nodes = list(df_nodes['bus_i'])
                df_gen_mat = pd.read_csv(full_gen_mat_full_path,header=0)

                all_nodes_lbl = ['bus_' + str(v) for v in all_nodes]
                df_nodes['bus_i'] = all_nodes_lbl
                A = np.zeros((len(gens),len(all_nodes_lbl)))
                df_A = pd.DataFrame(A, columns=all_nodes_lbl)
                df_A['name'] = gens
                df_A.set_index('name',inplace=True)
                for i in range(0,len(gens_oil)):
                    node = df_oil.loc[i,'node']
                    g = gens_oil[i]
                    df_A.loc[g,node] = 1

                tot_cap = np.zeros(len(all_nodes_lbl))
                oil_cap = np.zeros(len(all_nodes_lbl))
                oil_max = np.zeros(len(all_nodes_lbl))
                oil_hr = np.zeros(len(all_nodes_lbl))
                oil_min = np.zeros(len(all_nodes_lbl))
                oil_var_om = np.zeros(len(all_nodes_lbl))
                oil_no_load = np.zeros(len(all_nodes_lbl))
                oil_st_cost = np.zeros(len(all_nodes_lbl))
                oil_ramp = np.zeros(len(all_nodes_lbl))

                for i in range(0,len(all_nodes_lbl)):
                    tot_cap[i] = sum(gens_cap*df_gen_mat.iloc[:,i+1])
                    oil_cap[i] = sum(gens_cap*df_A.iloc[:,i])
                    oil_max[i] = sum(gens_cap*df_A.iloc[:,i])
                    oil_hr[i] = sum(gens_hr*(gens_cap*df_A.iloc[:,i])/sum(gens_cap*df_A.iloc[:,i]))
                    oil_min[i] = sum(gens_min*(gens_cap*df_A.iloc[:,i])/sum(gens_cap*df_A.iloc[:,i]))
                    oil_var_om[i] = sum(gens_var_om*(gens_cap*df_A.iloc[:,i])/sum(gens_cap*df_A.iloc[:,i]))
                    oil_no_load[i] = sum(gens_no_load*(gens_cap*df_A.iloc[:,i])/sum(gens_cap*df_A.iloc[:,i]))
                    oil_st_cost[i] = sum(gens_st_cost*(gens_cap*df_A.iloc[:,i])/sum(gens_cap*df_A.iloc[:,i]))
                    oil_ramp[i] = sum(gens_ramp*df_A.iloc[:,i])

                df_oil = pd.DataFrame()
                df_oil.loc[:,'node'] = list(df_nodes.loc[:,'bus_i'])
                df_oil.loc[:,'maxcap'] = oil_max
                df_oil.loc[:,'heat_rate'] = oil_hr
                df_oil.loc[:,'mincap'] = oil_min
                df_oil.loc[:,'var_om'] = oil_var_om
                df_oil.loc[:,'no_load'] = oil_no_load
                df_oil.loc[:,'st_cost'] = oil_st_cost
                df_oil.loc[:,'ramp'] = oil_ramp
                df_oil = df_oil.dropna().reset_index(drop=True)

                # If aggregation produced no valid rows (e.g., no oil gens anywhere), skip gracefully
                if df_oil.empty:
                    df = df.loc[df['typ'] != 'oil']
                else:
                    df_oil = df_oil.copy()
                    df_oil['minup'] = 1
                    df_oil['mindn'] = 1
                    df_oil['typ'] = 'oil'
                    oil_name = []
                    for i in range(0,len(df_oil.index)):
                        oil_name.append(df_oil.loc[i,'node'] + '_oil')
                    df_oil['name'] = oil_name
                    df_oil = df_oil[["name", "typ", "node", "maxcap", "heat_rate", "mincap", "var_om", "no_load", "st_cost", "ramp", "minup", "mindn"]]
                    df = df.loc[df['typ'] != 'oil']
                    df = pd.concat([df, df_oil], ignore_index=True)
                fn_genparams = f'data_genparams_{NN}.csv'
                full_genparams_path = _save_csv(df, fn_genparams, index=False)
                #copy(full_genparams_path, path)

                # Recreate gen-to-bus matrix
                df = pd.read_csv(full_genparams_path,header=0)
                gens = list(df.loc[:,'name'])
                df_nodes = pd.read_excel(os.path.join(base_dir, FN), sheet_name = 'Bus', header=0)
                all_nodes = ['bus_' + str(v) for v in list(df_nodes['bus_i'])]
                A = np.zeros((len(gens),len(all_nodes)))
                df_A = pd.DataFrame(A, columns=all_nodes)
                df_A['name'] = gens
                df_A.set_index('name',inplace=True)
                for i in range(0,len(gens)):
                    node = df.loc[i,'node']
                    g = gens[i]
                    df_A.loc[g,node] = 1
                fn_gen_mat = f'gen_mat_{NN}.csv'
                full_gen_mat_path = _save_csv(df_A, fn_gen_mat, index=True)
                #copy(full_gen_mat_path, path)


                # -----------------
                # YEAR-SPECIFIC FUEL PRICES
                # -----------------

                # Natural gas price by BA.
                # In annual mode, prefer year-specific files.
                # In constant mode, follow the old const-fuel-price script:
                # read the static file first, then replace each bus column
                # by its time-average value repeated across 365 days.
                if fuel_price_mode == 'annual':
                    ng_price_candidates = [
                        f'NG_price/Average_NG_prices_BAs_{y_string}.csv',
                    ]
                else:  # fuel_price_mode == 'constant'
                    ng_price_candidates = [
                        f'NG_price/Average_NG_prices_BAs_{y_string}.csv',
                    ]

                NG_price = _read_csv_first_existing(
                    base_dir,
                    ng_price_candidates,
                    header=0
                )

                Fuel_buses = ['bus_' + str(b) for b in buses]

                NG_prices_all = None
                for bus in buses:
                    selected_node_BA = df_full.loc[df_full['Number'] == bus, 'NAME'].values[0]
                    specific_node_NG_price = NG_price.loc[:, selected_node_BA].copy()

                    NG_prices_all = (
                        specific_node_NG_price.copy()
                        if NG_prices_all is None
                        else pd.concat([NG_prices_all, specific_node_NG_price], axis=1)
                    )

                if NG_prices_all is None:
                    NG_prices_all = pd.DataFrame(columns=Fuel_buses)

                NG_prices_all.columns = Fuel_buses

                if fuel_price_mode == 'constant':
                    # Column-wise bus averages, repeated for every simulation day.
                    # This mirrors the constant-fuel-price script behavior.
                    NG_prices_bus_avg = NG_prices_all.mean(axis=0)
                    NG_prices_all = pd.DataFrame(
                        [NG_prices_bus_avg] * 365,
                        columns=NG_prices_all.columns
                    )


                # Coal price by state.
                # Same annual/constant switch as natural gas.
                if fuel_price_mode == 'annual':
                    coal_price_candidates = [
                        f'Coal_price/coal_prices_state_{y_string}.csv',
                    ]
                else:  # fuel_price_mode == 'constant'
                    coal_price_candidates = [
                        f'Coal_price/coal_prices_state_{y_string}.csv',
                    ]

                Coal_price = _read_csv_first_existing(
                    base_dir,
                    coal_price_candidates,
                    header=0
                )

                Coal_prices_all = None
                for bus in buses:
                    selected_node_state = df_full.loc[df_full['Number'] == bus, 'STATE'].values[0]

                    # Existing state substitutions
                    if selected_node_state == 'NB':
                        selected_node_state = 'ME'
                    elif selected_node_state == 'CO':
                        selected_node_state = 'KS'
                    elif selected_node_state == 'OR':
                        selected_node_state = 'AR'
                    elif selected_node_state == 'TX':
                        selected_node_state = 'OK'
                    elif selected_node_state == 'NM':
                        selected_node_state = 'OK'
                    elif selected_node_state == 'MT':
                        selected_node_state = 'SD'

                    specific_node_coal_price = Coal_price.loc[:, selected_node_state].copy()

                    Coal_prices_all = (
                        specific_node_coal_price.copy()
                        if Coal_prices_all is None
                        else pd.concat([Coal_prices_all, specific_node_coal_price], axis=1)
                    )

                if Coal_prices_all is None:
                    Coal_prices_all = pd.DataFrame(columns=Fuel_buses)

                Coal_prices_all.columns = Fuel_buses

                if fuel_price_mode == 'constant':
                    # Column-wise bus averages, repeated for every simulation day.
                    # This mirrors the constant-fuel-price script behavior.
                    Coal_prices_bus_avg = Coal_prices_all.mean(axis=0)
                    Coal_prices_all = pd.DataFrame(
                        [Coal_prices_bus_avg] * 365,
                        columns=Coal_prices_all.columns
                    )


                # Oil still fixed unless you also want oil to vary by year
                Oil_prices_all = pd.DataFrame({
                    'all': np.reshape(np.ones((365, 1)) * 20, (365,))
                })


                # ============= Fuel price mapping to gens (NN-only structures, but gen list is year-specific)
                thermal_gens_info = df.loc[(df['typ']=='ngcc') | (df['typ']=='coal') | (df['typ']=='oil')].copy()
                thermal_gens_names = [*thermal_gens_info['name']]
                Fuel_prices_all = None
                for _, row in thermal_gens_info.iterrows():
                    if row['typ'] == 'ngcc':
                        gen_fuel_price = NG_prices_all.loc[:, row['node']].copy()
                    elif row['typ'] == 'coal':
                        gen_fuel_price = Coal_prices_all.loc[:, row['node']].copy()
                    elif row['typ'] == 'oil':
                        gen_fuel_price = Oil_prices_all.loc[:,'all'].copy()
                    Fuel_prices_all = gen_fuel_price.copy() if Fuel_prices_all is None else pd.concat([Fuel_prices_all, gen_fuel_price], axis=1)
                if Fuel_prices_all is None:
                    Fuel_prices_all = pd.DataFrame()
                Fuel_prices_all.columns = thermal_gens_names
                
                #fn_fuel_prices = f'Fuel_prices_{NN}.csv'
                #full_fuel_prices_path = _save_csv(Fuel_prices_all, fn_fuel_prices, index=False)
                #copy(full_fuel_prices_path, path)

                fn_fuel_prices = f'Fuel_prices_{NN}_y_{y}.csv'
                full_fuel_prices_path = _save_csv(Fuel_prices_all, fn_fuel_prices, index=False)
                print(
                    f"Saved fuel prices ({fuel_price_mode} mode): {full_fuel_prices_path}"
                )

                # Also save a mode-specific copy for diagnostics, while keeping
                # Fuel_prices_{NN}_y_{y}.csv as the standard file read by EICDataSetup.
#                fn_fuel_prices_mode = f'Fuel_prices_{NN}_y_{y}_{fuel_price_mode}.csv'
#                _save_csv(Fuel_prices_all, fn_fuel_prices_mode, index=False)

                # -----------------
                # YEAR-SPECIFIC RAW-TO-REDUCED OUTAGE-ADJUSTED CAPACITY
                # -----------------
                # GadsOutagesEAST now produces raw/pre-reduction available capacity:
                #     east_rawGens_metadata.csv
                #     east{year}_rawGensAvailableCap.csv
                # Here we aggregate those raw generator limits into the final reduced-network
                # model structure used by EIC_simple.py. This replaces the old one-to-one
                # generator-name match check against east{year}_gensGadsLostcap.csv.

                outage_key = (str(NN), y_string)
                if outage_key not in outage_limits_by_year:
                    raw_metadata_src = _raw_outage_metadata_src(base_dir)
                    raw_available_src = _raw_available_cap_src_for_year(base_dir, y_string)

                    outage_limit_outputs = _aggregate_raw_available_capacity_to_reduced_limits(
                        NN=NN,
                        year=y_string,
                        raw_metadata_path=raw_metadata_src,
                        raw_available_path=raw_available_src,
                        df_reduced_thermal_map=df_reduced_thermal_map,
                        final_genparams_path=full_genparams_path,
                        all_buses=buses_reduced,
                        raw_to_reduced_bus=raw_to_reduced_bus,
                    )
                    outage_limits_by_year[outage_key] = outage_limit_outputs
                else:
                    outage_limit_outputs = outage_limits_by_year[outage_key]
                    print(
                        f"Reusing base outage-adjusted capacity limits for NN={NN}, year={y_string}; "
                        f"these limits are independent of transmission case {trans_tag}."
                    )
                    print(f"  HorizonGenLimits:    {outage_limit_outputs['gen_limits_path']}")
                    print(f"  HorizonMustrunLimits: {outage_limit_outputs['mustrun_limits_path']}")

                # -----------------
                # YEAR-SPECIFIC RUN FOLDER
                # -----------------
                # The Exp folder is intentionally lightweight. All generated data
                # remain in Data/data_allocation and should be read from there by
                # wrapper_simple.py / EIC_simple.py.
                path = str(
                    Path.cwd()
                    / f'Exp{NN}{UC}_{trans_folder_token}_{y_string}'
                )
                os.makedirs(path, exist_ok=True)

                # Copy only the model scripts. wrapper_simple.py handles the four
                # solve windows internally.
                w = 'wrapper' + UC + '.py'
                milp = 'EIC_MILP' + UC + '.py'
                lp = 'EIC_LP' + UC + '.py'

                copy(w, path)

                if UC == '_simple':
                    copy('EIC' + UC + '.py', path)
                else:
                    copy(milp, path)
                    copy(lp, path)

                print(
                    f"Prepared lightweight run folder: NN={NN}, UC={UC}, {trans_print_label}, "
                    f"year={y_string} -> {path}"
                )
