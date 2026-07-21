from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# FILE LOCATIONS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

CAPACITY_FILE = (
    OUTPUT_DIR
    / "pv_forecast_all_scenarios_2027_2050_gw.csv"
)

ADDITIONS_FILE = (
    OUTPUT_DIR
    / "pv_annual_additions_all_scenarios_2027_2050_gw.csv"
)

WASTE_FILE = (
    OUTPUT_DIR
    / "pv_eol_model_outputs"
    / "pv_eol_waste_annual_tonnes_long.csv"
)

MATERIAL_FILE = (
    OUTPUT_DIR
    / "pv_material_waste_outputs"
    / "tables"
    / "pv_material_waste_all_years_long.csv"
)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="PV Waste Forecasting Dashboard",
    page_icon="☀️",
    layout="wide",
)

st.title("Solar PV Capacity, Waste and Material Forecasting Dashboard")

st.write(
    "This dashboard presents projected solar PV capacity, "
    "end-of-life waste and material quantities for India "
    "and selected states."
)


# ============================================================
# CHECK OUTPUT FILES
# ============================================================

required_files = {
    "Capacity forecast": CAPACITY_FILE,
    "Annual capacity additions": ADDITIONS_FILE,
    "PV waste forecast": WASTE_FILE,
    "Material waste inventory": MATERIAL_FILE,
}

missing_files = []

for file_name, file_path in required_files.items():
    if file_path.exists():
        st.success(f"{file_name} file found.")
    else:
        st.error(f"{file_name} file not found:")
        st.code(str(file_path))
        missing_files.append(file_path)

if missing_files:
    st.stop()


# ============================================================
# LOAD MODEL OUTPUTS
# ============================================================

capacity_df = pd.read_csv(CAPACITY_FILE)
additions_df = pd.read_csv(ADDITIONS_FILE)
waste_df = pd.read_csv(WASTE_FILE)
material_df = pd.read_csv(MATERIAL_FILE)

st.info("All model output files loaded successfully.")


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Dashboard Controls")

if "state" not in capacity_df.columns:
    st.error(
        "The capacity forecast file does not contain a column named 'state'."
    )
    st.stop()

available_regions = sorted(
    capacity_df["state"].dropna().astype(str).unique()
)

selected_region = st.sidebar.selectbox(
    "Select region",
    available_regions,
)


# ============================================================
# CAPACITY FORECAST
# ============================================================

st.divider()
st.header("1. Solar PV Capacity Forecast")

selected_capacity_df = capacity_df[
    capacity_df["state"].astype(str) == selected_region
].copy()

selected_capacity_df = selected_capacity_df.sort_values("year")

scenario_columns = [
    column
    for column in ["Conservative", "Medium", "High"]
    if column in selected_capacity_df.columns
]

if not scenario_columns:
    st.error(
        "Conservative, Medium and High scenario columns "
        "were not found in the capacity file."
    )
    st.stop()

st.subheader(
    f"Cumulative installed PV capacity: {selected_region}"
)

capacity_chart_df = (
    selected_capacity_df[
        ["year"] + scenario_columns
    ]
    .set_index("year")
)

st.line_chart(capacity_chart_df)

st.subheader("Capacity forecast data")

st.dataframe(
    selected_capacity_df[
        ["year"] + scenario_columns
    ],
    use_container_width=True,
    hide_index=True,
)
capacity_download_df = selected_capacity_df[
    ["year"] + scenario_columns
].copy()

capacity_csv = capacity_download_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download capacity forecast CSV",
    data=capacity_csv,
    file_name=f"{selected_region}_capacity_forecast.csv",
    mime="text/csv",
)
# ============================================================
# PV END-OF-LIFE WASTE FORECAST
# ============================================================
# ============================================================
# ANNUAL CAPACITY ADDITIONS
# ============================================================

st.subheader(
    f"Annual PV capacity additions: {selected_region}"
)

if "state" not in additions_df.columns:
    st.error(
        "The annual-additions file does not contain a column named 'state'."
    )

else:
    selected_additions_df = additions_df[
        additions_df["state"].astype(str) == selected_region
    ].copy()

    selected_additions_df = selected_additions_df.sort_values("year")

    additions_scenario_columns = [
        column
        for column in ["Conservative", "Medium", "High"]
        if column in selected_additions_df.columns
    ]

    if selected_additions_df.empty:
        st.warning(
            f"No annual-capacity-addition results were found for "
            f"{selected_region}."
        )

    elif not additions_scenario_columns:
        st.error(
            "Conservative, Medium and High scenario columns "
            "were not found in the annual-additions file."
        )

    else:
        additions_chart_df = (
            selected_additions_df[
                ["year"] + additions_scenario_columns
            ]
            .set_index("year")
        )

        st.line_chart(additions_chart_df)

        st.caption(
            "Annual additions are shown in GW per year."
        )

        st.dataframe(
            selected_additions_df[
                ["year"] + additions_scenario_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

        additions_csv = selected_additions_df[
            ["year"] + additions_scenario_columns
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download annual capacity additions CSV",
            data=additions_csv,
            file_name=(
                f"{selected_region}_annual_capacity_additions.csv"
            ),
            mime="text/csv",
        )


# ============================================================
# CURRENT DATA PREVIEWS
# ============================================================

# ============================================================
# PV END-OF-LIFE WASTE FORECAST
# ============================================================

st.divider()
st.header("2. PV End-of-Life Waste Forecast")

region_waste_df = waste_df[
    waste_df["region"].astype(str) == selected_region
].copy()

if region_waste_df.empty:
    st.warning(
        f"No waste-model results were found for {selected_region}."
    )

else:
    capacity_scenarios = sorted(
        region_waste_df["capacity_scenario"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_capacity_scenario = st.sidebar.selectbox(
        "Select capacity scenario",
        capacity_scenarios,
    )

    capacity_waste_df = region_waste_df[
        region_waste_df["capacity_scenario"].astype(str)
        == selected_capacity_scenario
    ].copy()

    lifetime_scenarios = sorted(
        capacity_waste_df["lifetime_scenario"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_lifetime_scenario = st.sidebar.selectbox(
        "Select lifetime scenario",
        lifetime_scenarios,
    )

    selected_waste_df = capacity_waste_df[
        capacity_waste_df["lifetime_scenario"].astype(str)
        == selected_lifetime_scenario
    ].copy()

    selected_waste_df = selected_waste_df.sort_values("year")

    st.subheader(
        f"{selected_region}: {selected_capacity_scenario} capacity, "
        f"{selected_lifetime_scenario} lifetime"
    )

    if not selected_waste_df.empty:
        peak_row = selected_waste_df.loc[
            selected_waste_df["annual_waste_tonnes"].idxmax()
        ]

        final_row = selected_waste_df.iloc[-1]

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "Peak annual waste",
            f"{peak_row['annual_waste_tonnes']:,.0f} tonnes",
        )

        metric_2.metric(
            "Peak-waste year",
            f"{int(peak_row['year'])}",
        )

        metric_3.metric(
            f"Cumulative waste by {int(final_row['year'])}",
            f"{final_row['cumulative_waste_tonnes']:,.0f} tonnes",
        )

        annual_column, cumulative_column = st.columns(2)

        with annual_column:
            st.subheader("Annual waste generation")

            annual_chart = (
                selected_waste_df[
                    ["year", "annual_waste_tonnes"]
                ]
                .set_index("year")
            )

            st.line_chart(annual_chart)

        with cumulative_column:
            st.subheader("Cumulative waste generation")

            cumulative_chart = (
                selected_waste_df[
                    ["year", "cumulative_waste_tonnes"]
                ]
                .set_index("year")
            )

            st.line_chart(cumulative_chart)

        st.subheader("Waste forecast data")

        st.dataframe(
            selected_waste_df[
                [
                    "year",
                    "annual_waste_tonnes",
                    "cumulative_waste_tonnes",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        waste_download_df = selected_waste_df[
            [
                "year",
                "annual_waste_tonnes",
                "cumulative_waste_tonnes",
            ]
        ].copy()

        waste_csv = waste_download_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download waste forecast CSV",
            data=waste_csv,
            file_name=(
                f"{selected_region}_"
                f"{selected_capacity_scenario}_"
                f"{selected_lifetime_scenario}_waste.csv"
            ),
            mime="text/csv",
        )

# ============================================================
# MATERIAL-WISE WASTE INVENTORY
# ============================================================

st.divider()
st.header("3. Material-Wise Waste Inventory")

if region_waste_df.empty:
    st.warning(
        "Material results cannot be filtered because no matching "
        "waste-model results were found."
    )

else:
    selected_material_df = material_df[
        (
            material_df["region"].astype(str)
            == selected_region
        )
        & (
            material_df["capacity_scenario"].astype(str)
            == selected_capacity_scenario
        )
        & (
            material_df["lifetime_scenario"].astype(str)
            == selected_lifetime_scenario
        )
    ].copy()

    if selected_material_df.empty:
        st.warning(
            f"No material results were found for {selected_region}, "
            f"{selected_capacity_scenario}, "
            f"{selected_lifetime_scenario}."
        )

    else:
        selected_material_df["year"] = pd.to_numeric(
            selected_material_df["year"],
            errors="coerce",
        )

        selected_material_df = selected_material_df.dropna(
            subset=["year"]
        )

        selected_material_df["year"] = (
            selected_material_df["year"].astype(int)
        )

        available_material_years = sorted(
            selected_material_df["year"].unique()
        )

        if 2045 in available_material_years:
            default_year_index = available_material_years.index(2045)
        else:
            default_year_index = len(available_material_years) - 1

        selected_material_year = st.sidebar.selectbox(
            "Select material-analysis year",
            available_material_years,
            index=default_year_index,
        )

        material_year_df = selected_material_df[
            selected_material_df["year"]
            == selected_material_year
        ].copy()

        material_year_df = material_year_df.sort_values(
            "annual_material_tonnes",
            ascending=False,
        )

        st.subheader(
            f"Material quantities in {selected_material_year}"
        )

        total_annual_material = (
            material_year_df["annual_material_tonnes"].sum()
        )

        total_cumulative_material = (
            material_year_df["cumulative_material_tonnes"].sum()
        )

        largest_material_row = material_year_df.loc[
            material_year_df[
                "annual_material_tonnes"
            ].idxmax()
        ]

        material_metric_1, material_metric_2, material_metric_3 = (
            st.columns(3)
        )

        material_metric_1.metric(
            f"Total material waste in {selected_material_year}",
            f"{total_annual_material:,.0f} tonnes",
        )

        material_metric_2.metric(
            f"Cumulative material waste by {selected_material_year}",
            f"{total_cumulative_material:,.0f} tonnes",
        )

        material_metric_3.metric(
            "Largest material stream",
            str(largest_material_row["material"]),
            f"{largest_material_row['annual_material_tonnes']:,.0f} tonnes",
        )

        annual_material_column, cumulative_material_column = (
            st.columns(2)
        )

        with annual_material_column:
            st.subheader("Annual material waste")

            annual_material_chart = (
                material_year_df[
                    ["material", "annual_material_tonnes"]
                ]
                .set_index("material")
            )

            st.bar_chart(annual_material_chart)

        with cumulative_material_column:
            st.subheader("Cumulative material waste")

            cumulative_material_chart = (
                material_year_df[
                    ["material", "cumulative_material_tonnes"]
                ]
                .set_index("material")
            )

            st.bar_chart(cumulative_material_chart)

        st.subheader("Material inventory data")

        material_table = material_year_df[
            [
                "material",
                "source_range",
                "raw_midpoint_weight_percent",
                "normalized_weight_percent",
                "annual_material_tonnes",
                "cumulative_material_tonnes",
            ]
        ].copy()

        material_table = material_table.rename(
            columns={
                "material": "Material",
                "source_range": "Source composition range",
                "raw_midpoint_weight_percent": "Raw midpoint (%)",
                "normalized_weight_percent": "Normalized composition (%)",
                "annual_material_tonnes": "Annual material waste (tonnes)",
                "cumulative_material_tonnes": (
                    "Cumulative material waste (tonnes)"
                ),
            }
        )

        st.dataframe(
            material_table,
            use_container_width=True,
            hide_index=True,
        )
        material_csv = material_table.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download material inventory CSV",
            data=material_csv,
            file_name=(
                f"{selected_region}_"
                f"{selected_capacity_scenario}_"
                f"{selected_lifetime_scenario}_"
                f"{selected_material_year}_materials.csv"
            ),
            mime="text/csv",
        )