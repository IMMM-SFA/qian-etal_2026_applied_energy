# DCOPF EAST

Model source is in `github/scripts/DCOPF-EAST/`; archived inputs are in `MSD-LIVE/DCOPF-EAST/Data/`. Prepared validation inputs and outputs are in `MSD-LIVE/DCOPF-EAST/best_model_settings/`.

## Run-folder layout

The model uses relative paths and expects its five Python files alongside a `Data/` folder. Create a fresh working folder, for example `paper1/reproduced/DCOPF-EAST/`, and copy the five `.py` files from this directory and the archived `Data/` folder into it:

```text
reproduced/DCOPF-EAST/
  reduced_network_data_allocation.py
  mapper_updated.py
  EICDataSetup.py
  EIC_simple.py
  wrapper_simple.py
  Data/
```

Run allocation and data setup from this working folder. Generated inputs go into `Data/data_allocation/`; allocation creates year-specific `Exp...` directories containing the solver scripts. Install the model dependencies listed in the repository's `requirements-model.txt`; solving requires Gurobi and a valid license.

## Configurations and execution

Both cases use NN500 and transmission parameter 300%. In the working copy of `reduced_network_data_allocation.py`, set the following explicitly:

| Case | Years | Fuel prices | Load, wind and solar profiles |
|---|---|---|---|
| Validation | `range(2019, 2020)` | `fuel_price_mode = 'annual'` | EIA historical profiles |
| Experiment for paper | `range(1980, 2020)` | `fuel_price_mode = 'constant'` | TELL/reV profiles |

Select the matching load, wind and solar input assignments in the allocation script as well as the year and fuel settings. The packaged raw outage-capacity input covers 2019. Archived 1980–2019 objective values can be used directly for the paper analyses without rerunning the optimization.

Run `python reduced_network_data_allocation.py`, then `python EICDataSetup.py`. Inside each generated experiment directory, run `python wrapper_simple.py --win 0`, followed by windows `--win 1`, `--win 2` and `--win 3`.

## LMP comparison

To reproduce the five 2019 LMP comparison plots (PJM, SWPP, MISO, ISNE and NYIS) from the archived validation results, run `python scripts/validation/model_validation_EAST.py` from the GitHub repository root. It reads `../MSD-LIVE/DCOPF-EAST/best_model_settings/` and the bus-to-BA mapping in `../MSD-LIVE/DCOPF-EAST/Data/Interconnections/`, and writes plots under `../reproduced/validation/LMPs_comparion/`.
