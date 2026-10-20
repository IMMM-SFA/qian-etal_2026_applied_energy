# qian-etal_2026_applied_energy

**Statistical and Machine Learning Analysis of Compound Weather Events and Resource Adequacy Stress in Interconnected Power Grids**

Jingwei Qian¹, Jordan Kern²*, Casey Burleyson³, Cameron Bracken³, Kerem Ziya Akdemir³, and Henry Ssembatya¹.

¹ North Carolina State University, Department of Civil, Construction and Environmental Engineering, Raleigh, NC 27606.

² North Carolina State University, Department of Industrial and Systems Engineering, Raleigh, NC 27606.

³ Pacific Northwest National Laboratory, Richland WA 99354.

*Corresponding author: jkern@ncsu.edu

## Abstract

While most resource adequacy planning for bulk power systems occurs at the control area level, operators commonly assume some ability to import electricity from neighboring systems during periods of scarcity. In large, interconnected networks, however, the ability to transfer power under stress depends not only on infrastructure limits but also on the spatial pattern and compound structure of weather extremes. This can make it difficult to perform post hoc analysis of failures, let alone anticipate future resource adequacy stress. In this paper, we use 40 years of weather-driven direct current optimal power flow simulations of the U.S. Eastern Interconnection and develop a statistical analysis and machine learning framework tailored to examine rare stress events from multiple perspectives. We use this framework to characterize how weather and renewable conditions influence resource adequacy stress. We find that severe stress events often involve geographically widespread heat or cold conditions across the interconnection. Wind-related conditions show evidence of amplifying stress either alone or under some temperature extremes. Solar availability exhibits a different pattern: low solar output can coincide with reductions in net load that partially offset the potential stress associated with reduced solar generation. At the most acute stress levels, however, our ability to learn the nature of weather–stress relationships for use in prediction is limited by imbalanced data, leading to sparse effective data support. This points to the need for larger, synthetic ensembles to better characterize rare weather-driven resource adequacy stress.

## Journal reference

Qian, J., Kern, J., Burleyson, C., Bracken, C., Akdemir, K. Z., & Ssembatya, H. (2026). Statistical and machine learning analysis of compound weather events and resource adequacy stress in interconnected power grids. Manuscript submitted to Applied Energy.

## Data and code reference

Qian, Jingwei, Jordan Kern, Casey Burleyson, Cameron Bracken, Kerem Ziya Akdemir, and Henry Ssembatya. “Statistical and Machine Learning Analysis of Compound Weather Events and Rare Resource Adequacy Stress in the U.S. Eastern Interconnection”. MSD-LIVE Data Repository, October 8, 2026. DOI: 10.57931/17290006

The MSD-LIVE Data Repository includes network/generator/fuel/outage inputs for the Direct Current Optimal Power Flow (DCOPF) model. This archive also shared analysis inputs, statistical/ML outputs, and the historical experiment objective values.

### Other existing source datasets

| Dataset | Reference |
|---|---|
| TGW county meteorology | https://doi.org/10.57931/1960548 |
| TGW BA meteorology | https://doi.org/10.57931/1960530 |
| TELL demand and reV wind/solar profiles | https://doi.org/10.57931/17290006 |
| EIA historical forecast demand and historical wind/solar profiles | https://doi.org/10.57931/17290006 |

## Contributing modeling software

| Component | Version | Role | Model Repository Link | DOI of Specific Version
|---|---|---|---|---|
| TELL | v1.1.0 | demand profiles | https://github.com/IMMM-SFA/tell | https://doi.org/10.5281/zenodo.8264217 |
| reV | v0.7.0 | wind/solar profiles | https://github.com/NREL/reV | https://doi.org/10.5281/zenodo.7301491 |
| TGW | v1.0.0 | upstream weather profiles | https://github.com/IMMM-SFA/jones-etal_2023_scidata | https://doi.org/10.57931/1885756 |

## Download and environment

Download the archive data via the MSD-LIVE Data Repository.

Default local layout:

```text
paper/
  github/                 # this repository
    scripts/DCOPF-EAST/    # model source
  MSD-LIVE/               # extracted data archive
    DCOPF-EAST/
      Data/               # model inputs
      best_model_settings/ # prepared validation data
    datasets/
    experiment_1980_2019/objective_values/
  reproduced/             # newly generated outputs
```

Install the analysis dependencies with `python -m pip install -r requirements.txt`. The prepared scripts require Python 3. Analysis and figure reproduction do not require a DCOPF solver. Running the DCOPF model additionally needs Pyomo, Gurobi and a valid Gurobi license; see `requirements-model.txt`.

## Reproduce my experiment analyses

The 1980–2019 experiment uses NN500, transmission parameter 300% and constant fuel prices. The prepared Results 3.1–3.5 analyses use daily objective values as the DCOPF stress signal, alongside independently prepared weather and renewable availability predictors. The data package includes all 160 original objective files (40 years × four solve windows) and the processed tables consumed directly by the analyses.

```powershell
python "scripts/3.1/two_sample_comparison.py"
python "scripts/3.1/wind_speed_diagnostics.py"
python "scripts/3.2/scatter_cut_lift.py"
python "scripts/3.3/ecdf_daily_coverage_by_stress.py"
python "scripts/3.4/lift_amplification_grid.py"
python "scripts/3.4/grid_search_panels.py"
python "scripts/3.4/compare_all_green_cells.py"
python "scripts/3.5/nestedcv_metrics_viz.py"
python "scripts/3.5/topk_coverage_viz.py"
```

Optional shared-table reconstruction:

```powershell
python "scripts/shared/system_daily_table.py"
```

Optional ML input check and retraining, preserving the prepared trainer:

```powershell
python "scripts/3.5/nested_stratified_cv.py" --check-inputs
python "scripts/3.5/nested_stratified_cv.py"
```

New outputs are written under `reproduced/` and do not overwrite the reference datasets.

## Reproduce my figures

| Figure(s) | Code under this repository | Figure files |
|---|---|---|
| 3, S3 | `scripts/3.1/linear_tick.py` | `linear_ticks_rates_alpha99p5_hot_cold_system.png`; S3 uses alpha97p0 and alpha99p0 |
| 4, 5, 6, S4, S5 | `scripts/3.2/scatter_cut_lift.py` | Temperature, coverage, Lift_Hot/Cold, wind and solar panels |
| 7 | `scripts/3.3/ecdf_daily_coverage_by_stress.py` | HOT/COLD ALL_DAYS_BENCHMARK panels |
| 8 | `scripts/3.4/grid_search_panels.py` | HOT/COLD amplification panels |
| 9 | `scripts/3.5/topk_coverage_viz.py` | Hot/cold k_coverage panels |
| 10 | `scripts/3.5/nestedcv_metrics_viz.py` | Hot/cold PR_AUC panels |
| S1 | `scripts/validation/model_validation_EAST.py` | Five 2019 LMP panels: manuscript order PJM, SWPP, MISO, ISNE, NYIS |

Prepared images are in `figures/3.1/` through `figures/3.5/` and `figures/validation/`.

Table S1 is the prepared two-sample risk-ratio workbook; Table S2 is the threshold/lift workbook; Table S3 comprises the cell and reconstructed comparison CSVs. They are in the data archive's section `tables/` folders. The solar-cutoff diagnostic script generates supporting diagnostics.

## Reproduce validation

Figure S1 can be reproduced directly from archived 2019 dual/LMP outputs, historical prices, loads and the bus-to-BA mapping:

```powershell
python "scripts/validation/model_validation_EAST.py"
```

For a new optimization run, use the model source in [`scripts/DCOPF-EAST/`](scripts/DCOPF-EAST/) and the downloaded `../MSD-LIVE/DCOPF-EAST/Data/` inputs. Follow the [DCOPF instructions](scripts/DCOPF-EAST/README.md) to assemble a separate run folder with the scripts alongside `Data/`, as required by the model's relative paths. The packaged outage-capacity input covers 2019; rerunning other years requires the corresponding year-specific outage inputs.
