# Solar PV Capacity, Waste, Material and Economic Valuation Dashboard

This project provides an integrated modelling and visualization framework for:

1. solar PV capacity forecasting;
2. annual PV capacity additions;
3. end-of-life PV waste forecasting;
4. material-wise waste estimation;
5. commodity-price pathway analysis;
6. quality-adjusted gross material-value estimation;
7. present-value analysis;
8. economic sensitivity analysis.

## Dashboard

The Streamlit dashboard allows users to select:

- region;
- capacity-growth scenario;
- module-lifetime scenario;
- commodity-price scenario;
- real discount rate;
- material;
- economic assessment year.

## Project structure

```text
pv__waste_application/
├── data/
│   └── pv_eol_config.json
├── outputs/
│   ├── pv_eol_model_outputs/
│   ├── pv_material_waste_outputs/
│   └── pv_eol_economic_outputs/
├── app.py
├── capacity_model.py
├── waste_model.py
├── material_model.py
├── pv_eol_economic_pipeline.py
├── requirements.txt
└── README.md