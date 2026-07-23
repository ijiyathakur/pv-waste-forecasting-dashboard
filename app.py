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
ECONOMIC_TABLES_DIR = (
    OUTPUT_DIR
    / "pv_eol_economic_outputs"
    / "tables"
)

VALUATION_FILE = (
    ECONOMIC_TABLES_DIR
    / "valuation_annual_long.csv"
)

PRICE_PATHWAYS_FILE = (
    ECONOMIC_TABLES_DIR
    / "commodity_price_pathways_2026_2075.csv"
)

SENSITIVITY_FILE = (
    ECONOMIC_TABLES_DIR
    / "sensitivity_global_one_at_a_time.csv"
)
MATERIAL_SENSITIVITY_FILE = (
    ECONOMIC_TABLES_DIR
    / "sensitivity_quality_discount_by_material.csv"
)
ASSUMPTIONS_FILE = (
    ECONOMIC_TABLES_DIR
    / "model_assumptions.csv"
)

VALIDATION_FILE = (
    ECONOMIC_TABLES_DIR
    / "model_validation_checks.csv"
)

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="PV Waste Forecasting Dashboard",
    page_icon="☀️",
    layout="wide",
)

st.title(
    "Solar PV Capacity, Waste, Material and Economic Valuation Dashboard"
)

st.write(
    "This dashboard presents projected solar PV capacity, "
    "end-of-life waste, material quantities, commodity-price pathways, "
    "economic valuation and sensitivity analysis for India "
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
    "Economic valuation": VALUATION_FILE,
    "Commodity price pathways": PRICE_PATHWAYS_FILE,
    "Economic sensitivity analysis": SENSITIVITY_FILE,
    "Economic model assumptions": ASSUMPTIONS_FILE,
    "Economic model validation": VALIDATION_FILE,
    "Material quality-discount sensitivity": MATERIAL_SENSITIVITY_FILE,
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

valuation_df = pd.read_csv(VALUATION_FILE)
price_pathways_df = pd.read_csv(PRICE_PATHWAYS_FILE)
sensitivity_df = pd.read_csv(SENSITIVITY_FILE)
material_sensitivity_df = pd.read_csv(MATERIAL_SENSITIVITY_FILE)
assumptions_df = pd.read_csv(ASSUMPTIONS_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

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
# ============================================================
# ECONOMIC VALUATION
# ============================================================

st.divider()
st.header("4. Economic Valuation of PV Materials")

st.caption(
    "This section reports quality-adjusted gross material-value "
    "potential in constant 2026 USD. It does not represent recycling "
    "profit because recovery losses, collection, transport, processing, "
    "refining, CAPEX and OPEX are not deducted."
)

if region_waste_df.empty:
    st.warning(
        "Economic valuation cannot be displayed because no matching "
        "regional waste results were found."
    )

else:
    region_valuation_df = valuation_df[
        (
            valuation_df["region"].astype(str)
            == selected_region
        )
        & (
            valuation_df["capacity_scenario"].astype(str)
            == selected_capacity_scenario
        )
        & (
            valuation_df["lifetime_scenario"].astype(str)
            == selected_lifetime_scenario
        )
    ].copy()

    if region_valuation_df.empty:
        st.warning(
            f"No economic-valuation results were found for "
            f"{selected_region}, {selected_capacity_scenario}, "
            f"{selected_lifetime_scenario}."
        )

    else:
        # ----------------------------------------------------
        # ECONOMIC CONTROLS
        # ----------------------------------------------------

        price_scenario_order = [
            scenario
            for scenario in [
                "Downside",
                "Reference",
                "Upside",
            ]
            if scenario
            in region_valuation_df[
                "price_scenario"
            ].astype(str).unique()
        ]

        if not price_scenario_order:
            price_scenario_order = sorted(
                region_valuation_df["price_scenario"]
                .dropna()
                .astype(str)
                .unique()
            )

        if "Reference" in price_scenario_order:
            default_price_index = (
                price_scenario_order.index("Reference")
            )
        else:
            default_price_index = 0

        selected_price_scenario = st.sidebar.selectbox(
            "Select commodity-price scenario",
            price_scenario_order,
            index=default_price_index,
        )

        discount_rate_columns = {
            "3%": "present_value_3pct_real_2026_usd",
            "5%": "present_value_5pct_real_2026_usd",
            "7%": "present_value_7pct_real_2026_usd",
        }

        available_discount_rates = [
            label
            for label, column
            in discount_rate_columns.items()
            if column in region_valuation_df.columns
        ]

        if not available_discount_rates:
            st.error(
                "No present-value columns were found in the "
                "economic-valuation output."
            )

        else:
            default_discount_index = (
                available_discount_rates.index("5%")
                if "5%" in available_discount_rates
                else 0
            )

            selected_discount_rate = st.sidebar.selectbox(
                "Select real discount rate",
                available_discount_rates,
                index=default_discount_index,
            )

            selected_pv_column = discount_rate_columns[
                selected_discount_rate
            ]

            selected_valuation_df = region_valuation_df[
                region_valuation_df[
                    "price_scenario"
                ].astype(str)
                == selected_price_scenario
            ].copy()

            selected_valuation_df["year"] = pd.to_numeric(
                selected_valuation_df["year"],
                errors="coerce",
            )

            selected_valuation_df = (
                selected_valuation_df.dropna(
                    subset=["year"]
                )
            )

            selected_valuation_df["year"] = (
                selected_valuation_df["year"].astype(int)
            )

            selected_valuation_df = (
                selected_valuation_df.sort_values(
                    ["year", "material"]
                )
            )

            # ------------------------------------------------
            # ANNUAL ECONOMIC TOTALS
            # ------------------------------------------------

            annual_valuation_df = (
                selected_valuation_df.groupby(
                    "year",
                    as_index=False,
                )[
                    [
                        "annual_material_tonnes",
                        "gross_benchmark_value_real_2026_usd",
                        "quality_adjusted_gross_value_real_2026_usd",
                        selected_pv_column,
                    ]
                ]
                .sum()
                .sort_values("year")
            )

            annual_valuation_df[
                "cumulative_present_value_real_2026_usd"
            ] = (
                annual_valuation_df[
                    selected_pv_column
                ].cumsum()
            )

            annual_valuation_df[
                "annual_quality_adjusted_value_billion_usd"
            ] = (
                annual_valuation_df[
                    "quality_adjusted_gross_value_real_2026_usd"
                ]
                / 1_000_000_000
            )

            annual_valuation_df[
                "annual_present_value_billion_usd"
            ] = (
                annual_valuation_df[selected_pv_column]
                / 1_000_000_000
            )

            annual_valuation_df[
                "cumulative_present_value_billion_usd"
            ] = (
                annual_valuation_df[
                    "cumulative_present_value_real_2026_usd"
                ]
                / 1_000_000_000
            )

            # ------------------------------------------------
            # ECONOMIC KPI CARDS
            # ------------------------------------------------

            total_quality_adjusted_value = (
                annual_valuation_df[
                    "quality_adjusted_gross_value_real_2026_usd"
                ].sum()
            )

            total_present_value = (
                annual_valuation_df[
                    selected_pv_column
                ].sum()
            )

            peak_value_row = annual_valuation_df.loc[
                annual_valuation_df[
                    "quality_adjusted_gross_value_real_2026_usd"
                ].idxmax()
            ]

            economic_metric_1, economic_metric_2, economic_metric_3 = (
                st.columns(3)
            )

            economic_metric_1.metric(
                "Full-period quality-adjusted value",
                f"USD {total_quality_adjusted_value / 1e9:,.2f} billion",
            )

            economic_metric_2.metric(
                f"Full-period present value ({selected_discount_rate})",
                f"USD {total_present_value / 1e9:,.2f} billion",
            )

            economic_metric_3.metric(
                "Peak annual-value year",
                f"{int(peak_value_row['year'])}",
                (
                    f"USD "
                    f"{peak_value_row['quality_adjusted_gross_value_real_2026_usd'] / 1e9:,.2f} "
                    f"billion"
                ),
            )

            st.subheader(
                f"{selected_region}: {selected_price_scenario} "
                f"commodity-price pathway"
            )
            
                        # ------------------------------------------------
            # COMMODITY-PRICE SCENARIO COMPARISON
            # ------------------------------------------------

            st.subheader(
                "Commodity-Price Scenario Comparison"
            )

            scenario_comparison_df = (
                region_valuation_df.groupby(
                    "price_scenario",
                    as_index=False,
                )[
                    [
                        "quality_adjusted_gross_value_real_2026_usd",
                        selected_pv_column,
                    ]
                ]
                .sum()
            )

            scenario_comparison_df[
                "quality_adjusted_value_billion_usd"
            ] = (
                scenario_comparison_df[
                    "quality_adjusted_gross_value_real_2026_usd"
                ]
                / 1_000_000_000
            )

            scenario_comparison_df[
                "present_value_billion_usd"
            ] = (
                scenario_comparison_df[
                    selected_pv_column
                ]
                / 1_000_000_000
            )

            scenario_order = [
                scenario
                for scenario in [
                    "Downside",
                    "Reference",
                    "Upside",
                ]
                if scenario
                in scenario_comparison_df[
                    "price_scenario"
                ].astype(str).unique()
            ]

            scenario_comparison_df[
                "price_scenario"
            ] = pd.Categorical(
                scenario_comparison_df[
                    "price_scenario"
                ],
                categories=scenario_order,
                ordered=True,
            )

            scenario_comparison_df = (
                scenario_comparison_df.sort_values(
                    "price_scenario"
                )
            )

            scenario_chart_df = (
                scenario_comparison_df[
                    [
                        "price_scenario",
                        "present_value_billion_usd",
                    ]
                ]
                .rename(
                    columns={
                        "price_scenario":
                        "Commodity-price scenario",
                        "present_value_billion_usd":
                        f"Present value at {selected_discount_rate}",
                    }
                )
                .set_index(
                    "Commodity-price scenario"
                )
            )

            st.bar_chart(
                scenario_chart_df
            )

            st.caption(
                "Values represent full-period present value in "
                "billion constant 2026 USD."
            )

            scenario_table = (
                scenario_comparison_df[
                    [
                        "price_scenario",
                        "quality_adjusted_value_billion_usd",
                        "present_value_billion_usd",
                    ]
                ]
                .rename(
                    columns={
                        "price_scenario":
                        "Commodity-price scenario",
                        "quality_adjusted_value_billion_usd":
                        "Quality-adjusted value (billion USD)",
                        "present_value_billion_usd":
                        f"Present value at {selected_discount_rate} "
                        "(billion USD)",
                    }
                )
            )

            st.dataframe(
                scenario_table,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # ANNUAL VALUE CHARTS
            # ------------------------------------------------
            # ------------------------------------------------
            # COMMODITY PRICE PATHWAYS
            # ------------------------------------------------

            available_price_materials = sorted(
                price_pathways_df["material"]
                .dropna()
                .astype(str)
                .unique()
            )

            if "Silver" in available_price_materials:
                default_material_index = (
                    available_price_materials.index("Silver")
                )
            else:
                default_material_index = 0

            selected_price_material = st.sidebar.selectbox(
                "Select commodity material",
                available_price_materials,
                index=default_material_index,
            )

            selected_price_pathway_df = price_pathways_df[
                price_pathways_df["material"].astype(str)
                == selected_price_material
            ].copy()

            selected_price_pathway_df["year"] = pd.to_numeric(
                selected_price_pathway_df["year"],
                errors="coerce",
            )

            selected_price_pathway_df = (
                selected_price_pathway_df.dropna(
                    subset=["year"]
                )
            )

            selected_price_pathway_df["year"] = (
                selected_price_pathway_df["year"].astype(int)
            )

            if selected_price_pathway_df.empty:
                st.warning(
                    f"No commodity-price pathways were found for "
                    f"{selected_price_material}."
                )

            else:
                price_pathway_chart = (
                    selected_price_pathway_df.pivot_table(
                        index="year",
                        columns="price_scenario",
                        values=(
                            "quality_adjusted_price_real_2026_usd_t"
                        ),
                        aggfunc="first",
                    )
                )

                ordered_price_columns = [
                    scenario
                    for scenario in [
                        "Downside",
                        "Reference",
                        "Upside",
                    ]
                    if scenario in price_pathway_chart.columns
                ]

                price_pathway_chart = price_pathway_chart[
                    ordered_price_columns
                ]

                st.subheader(
                    f"{selected_price_material} commodity-price pathways"
                )

                st.line_chart(price_pathway_chart)

                st.caption(
                    "Prices are quality-adjusted and expressed in "
                    "constant 2026 USD per tonne."
                )
            economic_chart_1, economic_chart_2 = (
                st.columns(2)
            )

            with economic_chart_1:
                st.subheader(
                    "Annual gross material-value potential"
                )

                future_value_chart = (
                    annual_valuation_df[
                        [
                            "year",
                            "annual_quality_adjusted_value_billion_usd",
                        ]
                    ]
                    .rename(
                        columns={
                            "annual_quality_adjusted_value_billion_usd":
                            "Quality-adjusted value",
                        }
                    )
                    .set_index("year")
                )

                st.line_chart(future_value_chart)

                st.caption(
                    "Values are in billion constant 2026 USD."
                )

            with economic_chart_2:
                st.subheader(
                    f"Annual present value at {selected_discount_rate}"
                )

                present_value_chart = (
                    annual_valuation_df[
                        [
                            "year",
                            "annual_present_value_billion_usd",
                        ]
                    ]
                    .rename(
                        columns={
                            "annual_present_value_billion_usd":
                            "Present value",
                        }
                    )
                    .set_index("year")
                )

                st.line_chart(present_value_chart)

                st.caption(
                    "Present values use a real discount rate."
                )

            st.subheader(
                "Cumulative present value"
            )

            cumulative_pv_chart = (
                annual_valuation_df[
                    [
                        "year",
                        "cumulative_present_value_billion_usd",
                    ]
                ]
                .rename(
                    columns={
                        "cumulative_present_value_billion_usd":
                        "Cumulative present value",
                    }
                )
                .set_index("year")
            )

            st.line_chart(cumulative_pv_chart)

            # ------------------------------------------------
            # MATERIAL CONTRIBUTION
            # ------------------------------------------------

            available_economic_years = sorted(
                selected_valuation_df[
                    "year"
                ].unique()
            )

            if 2045 in available_economic_years:
                default_economic_year_index = (
                    available_economic_years.index(2045)
                )
            else:
                default_economic_year_index = (
                    len(available_economic_years) - 1
                )

            selected_economic_year = st.sidebar.selectbox(
                "Select economic-analysis year",
                available_economic_years,
                index=default_economic_year_index,
            )

            economic_year_df = selected_valuation_df[
                selected_valuation_df["year"]
                == selected_economic_year
            ].copy()

            material_value_df = (
                economic_year_df.groupby(
                    "material",
                    as_index=False,
                )[
                    [
                        "quality_adjusted_gross_value_real_2026_usd",
                        selected_pv_column,
                    ]
                ]
                .sum()
            )

            material_value_df[
                "quality_adjusted_value_million_usd"
            ] = (
                material_value_df[
                    "quality_adjusted_gross_value_real_2026_usd"
                ]
                / 1_000_000
            )

            material_value_df[
                "present_value_million_usd"
            ] = (
                material_value_df[
                    selected_pv_column
                ]
                / 1_000_000
            )

            material_value_df = material_value_df.sort_values(
                "quality_adjusted_value_million_usd",
                ascending=False,
            )

            st.subheader(
                f"Material contribution in {selected_economic_year}"
            )

            material_value_chart = (
                material_value_df[
                    [
                        "material",
                        "quality_adjusted_value_million_usd",
                    ]
                ]
                .rename(
                    columns={
                        "quality_adjusted_value_million_usd":
                        "Quality-adjusted value",
                    }
                )
                .set_index("material")
            )

            st.bar_chart(material_value_chart)

            st.caption(
                "Material values are shown in million constant 2026 USD."
            )

            # ------------------------------------------------
            # ECONOMIC DATA TABLE
            # ------------------------------------------------

            st.subheader("Economic valuation data")

            economic_table = annual_valuation_df[
                [
                    "year",
                    "annual_material_tonnes",
                    "gross_benchmark_value_real_2026_usd",
                    "quality_adjusted_gross_value_real_2026_usd",
                    selected_pv_column,
                    "cumulative_present_value_real_2026_usd",
                ]
            ].copy()

            economic_table = economic_table.rename(
                columns={
                    "year": "Year",
                    "annual_material_tonnes":
                    "Material quantity (tonnes)",
                    "gross_benchmark_value_real_2026_usd":
                    "Gross benchmark value (constant 2026 USD)",
                    "quality_adjusted_gross_value_real_2026_usd":
                    "Quality-adjusted gross value (constant 2026 USD)",
                    selected_pv_column:
                    f"Present value at {selected_discount_rate} "
                    "(constant 2026 USD)",
                    "cumulative_present_value_real_2026_usd":
                    f"Cumulative present value at "
                    f"{selected_discount_rate} "
                    "(constant 2026 USD)",
                }
            )

            st.dataframe(
                economic_table,
                use_container_width=True,
                hide_index=True,
            )

            economic_csv = economic_table.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download economic valuation CSV",
                data=economic_csv,
                file_name=(
                    f"{selected_region}_"
                    f"{selected_capacity_scenario}_"
                    f"{selected_lifetime_scenario}_"
                    f"{selected_price_scenario}_valuation.csv"
                ),
                mime="text/csv",
            )
            # ------------------------------------------------
            # SENSITIVITY ANALYSIS
            # ------------------------------------------------

            st.subheader(
                "One-at-a-Time Economic Sensitivity Analysis"
            )

            st.caption(
                "This sensitivity analysis represents the central case "
                "defined in pv_eol_config.json. It is not controlled by "
                "the region and scenario selections above."
            )

            sensitivity_metric_columns = {
                "Full-period cumulative present value":
                "cumulative_pv_change_percent",
                "Selected-year snapshot present value":
                "snapshot_pv_change_percent",
            }

            selected_sensitivity_metric = (
                st.sidebar.selectbox(
                    "Select sensitivity result",
                    list(sensitivity_metric_columns.keys()),
                    index=0,
                )
            )

            sensitivity_value_column = (
                sensitivity_metric_columns[
                    selected_sensitivity_metric
                ]
            )

            sensitivity_plot_df = sensitivity_df[
                [
                    "variable",
                    "case",
                    sensitivity_value_column,
                ]
            ].copy()

            sensitivity_plot_df[
                sensitivity_value_column
            ] = pd.to_numeric(
                sensitivity_plot_df[
                    sensitivity_value_column
                ],
                errors="coerce",
            )

            sensitivity_plot_df = (
                sensitivity_plot_df.dropna(
                    subset=[sensitivity_value_column]
                )
            )

            sensitivity_chart_df = (
                sensitivity_plot_df.pivot_table(
                    index="variable",
                    columns="case",
                    values=sensitivity_value_column,
                    aggfunc="first",
                )
            )

            ordered_sensitivity_cases = [
                case
                for case in [
                    "Low",
                    "Adverse",
                    "Favourable",
                    "High",
                ]
                if case in sensitivity_chart_df.columns
            ]

            sensitivity_chart_df = (
                sensitivity_chart_df[
                    ordered_sensitivity_cases
                ]
            )

            sensitivity_chart_df[
                "_maximum_absolute_change"
            ] = (
                sensitivity_chart_df.abs()
                .max(axis=1)
            )

            sensitivity_chart_df = (
                sensitivity_chart_df.sort_values(
                    "_maximum_absolute_change",
                    ascending=False,
                )
                .drop(
                    columns="_maximum_absolute_change"
                )
            )

            st.bar_chart(
                sensitivity_chart_df
            )

            st.caption(
                "Values show the percentage change relative to the "
                "central economic-valuation case. Negative values reduce "
                "the estimated present value; positive values increase it."
            )

            st.subheader(
                "Sensitivity-analysis data"
            )

            sensitivity_table = (
                sensitivity_plot_df.rename(
                    columns={
                        "variable": "Variable",
                        "case": "Case",
                        sensitivity_value_column:
                        "Change from central case (%)",
                    }
                )
            )

            st.dataframe(
                sensitivity_table,
                use_container_width=True,
                hide_index=True,
            )

            sensitivity_csv = (
                sensitivity_table.to_csv(
                    index=False
                ).encode("utf-8")
            )

            st.download_button(
                label="Download sensitivity-analysis CSV",
                data=sensitivity_csv,
                file_name="economic_sensitivity_analysis.csv",
                mime="text/csv",
            )
            
                        # ------------------------------------------------
            # MATERIAL-WISE QUALITY-DISCOUNT SENSITIVITY
            # ------------------------------------------------

            st.subheader(
                "Material-Wise Quality-Discount Sensitivity"
            )

            st.caption(
                "This chart shows how changing the quality discount "
                "for one material at a time affects the economic "
                "present value of the central case."
            )

            material_sensitivity_plot_df = material_sensitivity_df[
                [
                    "material",
                    "case",
                    sensitivity_value_column,
                ]
            ].copy()

            material_sensitivity_plot_df[
                sensitivity_value_column
            ] = pd.to_numeric(
                material_sensitivity_plot_df[
                    sensitivity_value_column
                ],
                errors="coerce",
            )

            material_sensitivity_plot_df = (
                material_sensitivity_plot_df.dropna(
                    subset=[sensitivity_value_column]
                )
            )

            material_sensitivity_chart_df = (
                material_sensitivity_plot_df.pivot_table(
                    index="material",
                    columns="case",
                    values=sensitivity_value_column,
                    aggfunc="first",
                )
            )

            ordered_material_cases = [
                case
                for case in [
                    "Adverse",
                    "Favourable",
                ]
                if case
                in material_sensitivity_chart_df.columns
            ]

            material_sensitivity_chart_df = (
                material_sensitivity_chart_df[
                    ordered_material_cases
                ]
            )

            material_sensitivity_chart_df[
                "_maximum_absolute_change"
            ] = (
                material_sensitivity_chart_df.abs()
                .max(axis=1)
            )

            material_sensitivity_chart_df = (
                material_sensitivity_chart_df.sort_values(
                    "_maximum_absolute_change",
                    ascending=False,
                )
                .drop(
                    columns="_maximum_absolute_change"
                )
            )

            st.bar_chart(
                material_sensitivity_chart_df
            )

            st.caption(
                "Positive values indicate an increase in present value. "
                "Negative values indicate a decrease relative to the "
                "central economic case."
            )

            material_sensitivity_table = (
                material_sensitivity_plot_df.rename(
                    columns={
                        "material": "Material",
                        "case": "Case",
                        sensitivity_value_column:
                        "Change from central case (%)",
                    }
                )
            )

            st.dataframe(
                material_sensitivity_table,
                use_container_width=True,
                hide_index=True,
            )

            material_sensitivity_csv = (
                material_sensitivity_table.to_csv(
                    index=False
                ).encode("utf-8")
            )

            st.download_button(
                label=(
                    "Download material quality-discount "
                    "sensitivity CSV"
                ),
                data=material_sensitivity_csv,
                file_name=(
                    "material_quality_discount_sensitivity.csv"
                ),
                mime="text/csv",
            )
            
            # ------------------------------------------------
            # MODEL ASSUMPTIONS AND VALIDATION
            # ------------------------------------------------

            st.subheader(
                "Economic Model Assumptions and Validation"
            )

            with st.expander(
                "View commodity-price and quality assumptions"
            ):
                assumption_columns = [
                    column
                    for column in [
                        "material",
                        "base_price_usd_t",
                        "quality_discount",
                        "quality_factor",
                        "quality_adjusted_base_price_usd_t",
                        "downside_growth_2027_2035",
                        "downside_growth_2036_2075",
                        "reference_growth_2027_2035",
                        "reference_growth_2036_2075",
                        "upside_growth_2027_2035",
                        "upside_growth_2036_2075",
                        "quality_discount_basis",
                    ]
                    if column in assumptions_df.columns
                ]

                assumptions_table = assumptions_df[
                    assumption_columns
                ].copy()

                st.dataframe(
                    assumptions_table,
                    use_container_width=True,
                    hide_index=True,
                )

                assumptions_csv = assumptions_table.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download economic assumptions CSV",
                    data=assumptions_csv,
                    file_name="economic_model_assumptions.csv",
                    mime="text/csv",
                )

            with st.expander(
                "View internal model-validation checks"
            ):
                validation_table = validation_df.copy()

                if "passed" in validation_table.columns:
                    passed_mask = (
                        validation_table["passed"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["true", "1", "yes"])
                    )

                    if passed_mask.all():
                        st.success(
                            "All internal economic-model checks passed."
                        )

                    else:
                        failed_checks = (
                            validation_table.loc[
                                ~passed_mask,
                                "check",
                            ]
                            .astype(str)
                            .tolist()
                        )

                        st.error(
                            "Some economic-model checks failed: "
                            + ", ".join(failed_checks)
                        )

                st.dataframe(
                    validation_table,
                    use_container_width=True,
                    hide_index=True,
                )

                validation_csv = validation_table.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download model-validation CSV",
                    data=validation_csv,
                    file_name="economic_model_validation.csv",
                    mime="text/csv",
                )
                
                            # ------------------------------------------------
            # ECONOMIC METHODOLOGY AND LIMITATIONS
            # ------------------------------------------------

            with st.expander(
                "Economic methodology, interpretation and limitations"
            ):
                st.markdown(
                    """
### Economic methodology

The economic valuation is calculated by multiplying the projected
annual quantity of each end-of-life PV material by its corresponding
quality-adjusted commodity price.

The model evaluates three commodity-price pathways:

- **Downside**
- **Reference**
- **Upside**

Future monetary values are expressed in constant 2026 USD. Present
values are calculated using real discount rates of 3%, 5% and 7%.

### Interpretation

The reported value represents the potential gross value of materials
contained in projected end-of-life PV waste.

It should not be interpreted as recycling revenue, recycling profit or
net economic benefit.

### Excluded costs and losses

The current valuation does not deduct:

- material recovery losses;
- collection and transportation costs;
- dismantling and sorting costs;
- processing and refining costs;
- capital expenditure;
- operating expenditure;
- taxes and transaction costs.

Therefore, the results represent an upper-level material-value
potential rather than the profitability of a recycling facility.
                    """
                )