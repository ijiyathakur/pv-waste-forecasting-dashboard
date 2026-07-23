#!/usr/bin/env python3
r"""
Complete economic-valuation pipeline for end-of-life crystalline-silicon PV materials.

Purpose
-------
Estimate the quality-adjusted gross material-value potential of six material streams:
Aluminium, Copper, Silver, Silicon, Glass, and Polymers.

The model intentionally does NOT deduct:
- physical recycling/recovery losses,
- collection and transport costs,
- dismantling and processing costs,
- refining costs,
- CAPEX/OPEX, taxes, or transaction costs.

Accordingly, outputs are gross material-value potential, not recycling profit or net cash flow.
Prices and values are expressed in constant 2026 USD. Present values therefore use real
(rather than nominal) discount rates.

The quality discounts and long-run price growth rates are transparent study assumptions.
They are editable in the companion JSON configuration file.

Command-line use
----------------
python pv_eol_economic_pipeline.py \
    --input "pv_material_waste_all_years_long(1).csv" \
    --config "pv_eol_config.json" \
    --output "pv_eol_economic_outputs"

Jupyter use
-----------
from pv_eol_economic_pipeline import run_pipeline
run_pipeline(
    input_csv=r"C:\path\pv_material_waste_all_years_long(1).csv",
    output_dir=r"C:\path\pv_eol_economic_outputs",
    config_path=r"C:\path\pv_eol_config.json",
)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


LOGGER = logging.getLogger("pv_eol_economic_pipeline")

REQUIRED_COLUMNS = {
    "region",
    "capacity_scenario",
    "lifetime_scenario",
    "year",
    "material",
    "annual_material_tonnes",
    "cumulative_material_tonnes",
}

KEY_COLUMNS = [
    "region",
    "capacity_scenario",
    "lifetime_scenario",
    "year",
    "material",
]

SCENARIO_ORDER = ["Downside", "Reference", "Upside"]


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    tables: Path
    charts: Path
    logs: Path


class PipelineError(RuntimeError):
    """Raised when the input/configuration is unsuitable for valuation."""


def configure_logging(log_path: Path) -> None:
    """Configure file and console logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def make_output_paths(output_dir: Path) -> PipelinePaths:
    root = output_dir.resolve()
    tables = root / "tables"
    charts = root / "charts"
    logs = root / "logs"
    for path in (root, tables, charts, logs):
        path.mkdir(parents=True, exist_ok=True)
    return PipelinePaths(root=root, tables=tables, charts=charts, logs=logs)


def load_config(config_path: Path) -> Dict[str, Any]:
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise PipelineError(f"Configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON configuration: {config_path}: {exc}") from exc

    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if "analysis" not in config or "materials" not in config:
        raise PipelineError("Configuration must contain 'analysis' and 'materials' sections.")

    analysis = config["analysis"]
    base_year = int(analysis["base_year"])
    end_year = int(analysis["end_year"])
    if end_year < base_year:
        raise PipelineError("end_year must be greater than or equal to base_year.")

    rates = [float(x) for x in analysis["real_discount_rates"]]
    if not rates or any(rate < 0 or rate >= 1 for rate in rates):
        raise PipelineError("real_discount_rates must be between 0 and 1.")

    central_rate = float(analysis["central_discount_rate"])
    if central_rate not in rates:
        raise PipelineError("central_discount_rate must be included in real_discount_rates.")

    selected_years = [int(x) for x in analysis["selected_years"]]
    if any(year < base_year or year > end_year for year in selected_years):
        raise PipelineError("All selected_years must lie between base_year and end_year.")

    if not config["materials"]:
        raise PipelineError("At least one material must be configured.")

    for material_name, item in config["materials"].items():
        base_price = float(item["base_price_usd_t"])
        discount = float(item["quality_discount"])
        if base_price < 0:
            raise PipelineError(f"Negative base price for {material_name}.")
        if not 0 <= discount <= 1:
            raise PipelineError(
                f"quality_discount for {material_name} must be between 0 and 1."
            )
        growth_rates = item["growth_rates"]
        if set(growth_rates) != set(SCENARIO_ORDER):
            raise PipelineError(
                f"{material_name} must define growth rates for {SCENARIO_ORDER}."
            )
        for scenario in SCENARIO_ORDER:
            for phase in ("2027_2035", "2036_2075"):
                if phase not in growth_rates[scenario]:
                    raise PipelineError(
                        f"Missing {phase} growth rate for {material_name}/{scenario}."
                    )


def load_and_validate_input(input_csv: Path, config: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the input and return (clean dataframe, data-quality report)."""
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError as exc:
        raise PipelineError(f"Input CSV not found: {input_csv}") from exc
    except Exception as exc:
        raise PipelineError(f"Could not read input CSV: {exc}") from exc

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise PipelineError(f"Input CSV is missing required columns: {missing}")

    original_rows = len(df)
    df = df.copy()

    # Normalize text fields without changing scenario spelling beyond whitespace.
    for column in ["region", "capacity_scenario", "lifetime_scenario", "material"]:
        df[column] = df[column].astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["annual_material_tonnes"] = pd.to_numeric(
        df["annual_material_tonnes"], errors="coerce"
    )
    df["cumulative_material_tonnes"] = pd.to_numeric(
        df["cumulative_material_tonnes"], errors="coerce"
    )

    null_required = df[list(REQUIRED_COLUMNS)].isna().sum()
    if int(null_required.sum()) > 0:
        details = null_required[null_required > 0].to_dict()
        raise PipelineError(f"Null/non-numeric values found in required fields: {details}")

    df["year"] = df["year"].astype(int)

    negative_annual = int((df["annual_material_tonnes"] < 0).sum())
    negative_cumulative = int((df["cumulative_material_tonnes"] < 0).sum())
    if negative_annual or negative_cumulative:
        raise PipelineError(
            "Material quantities must be non-negative. "
            f"Negative annual rows={negative_annual}, cumulative rows={negative_cumulative}."
        )

    duplicate_count = int(df.duplicated(KEY_COLUMNS, keep=False).sum())
    if duplicate_count:
        LOGGER.warning(
            "Detected %s rows belonging to duplicate keys; quantities will be summed.",
            duplicate_count,
        )
        passthrough_columns = [
            col
            for col in df.columns
            if col not in KEY_COLUMNS + ["annual_material_tonnes", "cumulative_material_tonnes"]
        ]
        aggregations: Dict[str, Any] = {
            "annual_material_tonnes": "sum",
            "cumulative_material_tonnes": "max",
        }
        aggregations.update({col: "first" for col in passthrough_columns})
        df = df.groupby(KEY_COLUMNS, as_index=False).agg(aggregations)

    configured_csv_materials = {
        str(item["csv_material"]) for item in config["materials"].values()
    }
    available_materials = set(df["material"].unique())
    missing_materials = sorted(configured_csv_materials - available_materials)
    if missing_materials:
        raise PipelineError(
            f"Configured materials are absent from the input CSV: {missing_materials}"
        )

    analysis = config["analysis"]
    base_year = int(analysis["base_year"])
    end_year = int(analysis["end_year"])
    if df["year"].min() > base_year or df["year"].max() < end_year:
        raise PipelineError(
            f"Input year range {df['year'].min()}-{df['year'].max()} does not cover "
            f"the requested valuation range {base_year}-{end_year}."
        )

    # Verify cumulative quantities against annual quantities over the complete input range.
    df = df.sort_values(KEY_COLUMNS).reset_index(drop=True)
    group_cols = ["region", "capacity_scenario", "lifetime_scenario", "material"]
    calculated_cumulative = df.groupby(group_cols, sort=False)[
        "annual_material_tonnes"
    ].cumsum()
    cumulative_abs_error = (
        calculated_cumulative - df["cumulative_material_tonnes"]
    ).abs()

    # Check normalized material composition when the column is present.
    weight_sum_min = np.nan
    weight_sum_max = np.nan
    if "normalized_weight_percent" in df.columns:
        unique_weight_rows = df.drop_duplicates(
            ["region", "capacity_scenario", "lifetime_scenario", "year", "material"]
        )
        weight_sums = unique_weight_rows.groupby(
            ["region", "capacity_scenario", "lifetime_scenario", "year"]
        )["normalized_weight_percent"].sum()
        weight_sum_min = float(weight_sums.min())
        weight_sum_max = float(weight_sums.max())

    report_rows = [
        ("input_rows", original_rows, "rows"),
        ("clean_rows", len(df), "rows"),
        ("duplicate_rows_detected", duplicate_count, "rows"),
        ("minimum_year", int(df["year"].min()), "year"),
        ("maximum_year", int(df["year"].max()), "year"),
        ("regions", int(df["region"].nunique()), "count"),
        ("capacity_scenarios", int(df["capacity_scenario"].nunique()), "count"),
        ("lifetime_scenarios", int(df["lifetime_scenario"].nunique()), "count"),
        ("materials", int(df["material"].nunique()), "count"),
        ("negative_annual_rows", negative_annual, "rows"),
        ("negative_cumulative_rows", negative_cumulative, "rows"),
        ("max_cumulative_absolute_error_t", float(cumulative_abs_error.max()), "tonnes"),
        ("mean_cumulative_absolute_error_t", float(cumulative_abs_error.mean()), "tonnes"),
        ("minimum_normalized_weight_sum_percent", weight_sum_min, "percent"),
        ("maximum_normalized_weight_sum_percent", weight_sum_max, "percent"),
    ]
    quality_report = pd.DataFrame(report_rows, columns=["check", "value", "unit"])

    # Filter only after complete-input cumulative validation.
    df = df[
        df["year"].between(base_year, end_year)
        & df["material"].isin(configured_csv_materials)
    ].copy()

    # Map input material names to canonical configured names.
    csv_to_canonical = {
        str(item["csv_material"]): canonical
        for canonical, item in config["materials"].items()
    }
    df["material"] = df["material"].map(csv_to_canonical)

    if df.empty:
        raise PipelineError("No rows remain after filtering configured years/materials.")

    return df, quality_report


def build_assumption_table(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for material, item in config["materials"].items():
        row: Dict[str, Any] = {
            "material": material,
            "csv_material": item["csv_material"],
            "base_price_usd_t": float(item["base_price_usd_t"]),
            "quality_discount": float(item["quality_discount"]),
            "quality_factor": 1.0 - float(item["quality_discount"]),
            "quality_adjusted_base_price_usd_t": float(item["base_price_usd_t"])
            * (1.0 - float(item["quality_discount"])),
            "quality_discount_basis": item.get("quality_discount_basis", ""),
        }
        for scenario in SCENARIO_ORDER:
            row[f"{scenario.lower()}_growth_2027_2035"] = float(
                item["growth_rates"][scenario]["2027_2035"]
            )
            row[f"{scenario.lower()}_growth_2036_2075"] = float(
                item["growth_rates"][scenario]["2036_2075"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_price_pathways(config: Mapping[str, Any]) -> pd.DataFrame:
    """Construct annual real commodity-price pathways from base year to end year."""
    analysis = config["analysis"]
    base_year = int(analysis["base_year"])
    end_year = int(analysis["end_year"])

    rows: List[Dict[str, Any]] = []
    for material, item in config["materials"].items():
        base_price = float(item["base_price_usd_t"])
        quality_discount = float(item["quality_discount"])

        for price_scenario in SCENARIO_ORDER:
            price = base_price
            for year in range(base_year, end_year + 1):
                if year == base_year:
                    growth_rate = 0.0
                elif year <= 2035:
                    growth_rate = float(
                        item["growth_rates"][price_scenario]["2027_2035"]
                    )
                    price *= 1.0 + growth_rate
                else:
                    growth_rate = float(
                        item["growth_rates"][price_scenario]["2036_2075"]
                    )
                    price *= 1.0 + growth_rate

                rows.append(
                    {
                        "material": material,
                        "price_scenario": price_scenario,
                        "year": year,
                        "base_price_usd_t": base_price,
                        "annual_real_growth_rate": growth_rate,
                        "price_real_2026_usd_t": price,
                        "quality_discount": quality_discount,
                        "quality_factor": 1.0 - quality_discount,
                        "quality_adjusted_price_real_2026_usd_t": price
                        * (1.0 - quality_discount),
                    }
                )

    prices = pd.DataFrame(rows)
    prices["price_scenario"] = pd.Categorical(
        prices["price_scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    return prices.sort_values(["material", "price_scenario", "year"]).reset_index(
        drop=True
    )


def rate_to_label(rate: float) -> str:
    return f"{int(round(rate * 100))}pct"


def calculate_valuation(
    material_data: pd.DataFrame,
    price_pathways: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    analysis = config["analysis"]
    base_year = int(analysis["base_year"])
    discount_rates = [float(x) for x in analysis["real_discount_rates"]]

    valuation = material_data.merge(
        price_pathways,
        on=["material", "year"],
        how="left",
        validate="many_to_many",
    )

    if valuation["price_real_2026_usd_t"].isna().any():
        missing = valuation.loc[
            valuation["price_real_2026_usd_t"].isna(), ["material", "year"]
        ].drop_duplicates()
        raise PipelineError(
            "Price-pathway merge failed for: " + missing.to_dict(orient="records").__repr__()
        )

    valuation["gross_benchmark_value_real_2026_usd"] = (
        valuation["annual_material_tonnes"]
        * valuation["price_real_2026_usd_t"]
    )
    valuation["quality_adjusted_gross_value_real_2026_usd"] = (
        valuation["annual_material_tonnes"]
        * valuation["quality_adjusted_price_real_2026_usd_t"]
    )

    years_from_base = valuation["year"] - base_year
    for rate in discount_rates:
        label = rate_to_label(rate)
        valuation[f"discount_factor_{label}"] = 1.0 / np.power(
            1.0 + rate, years_from_base
        )
        valuation[f"present_value_{label}_real_2026_usd"] = (
            valuation["quality_adjusted_gross_value_real_2026_usd"]
            * valuation[f"discount_factor_{label}"]
        )

    group_cols = [
        "region",
        "capacity_scenario",
        "lifetime_scenario",
        "price_scenario",
        "material",
    ]
    valuation = valuation.sort_values(group_cols + ["year"]).reset_index(drop=True)
    valuation["cumulative_quality_adjusted_value_real_2026_usd"] = valuation.groupby(
        group_cols, observed=True
    )["quality_adjusted_gross_value_real_2026_usd"].cumsum()
    for rate in discount_rates:
        label = rate_to_label(rate)
        valuation[f"cumulative_present_value_{label}_real_2026_usd"] = valuation.groupby(
            group_cols, observed=True
        )[f"present_value_{label}_real_2026_usd"].cumsum()

    preferred_columns = [
        "region",
        "capacity_scenario",
        "lifetime_scenario",
        "price_scenario",
        "year",
        "material",
        "annual_material_tonnes",
        "cumulative_material_tonnes",
        "base_price_usd_t",
        "annual_real_growth_rate",
        "price_real_2026_usd_t",
        "quality_discount",
        "quality_factor",
        "quality_adjusted_price_real_2026_usd_t",
        "gross_benchmark_value_real_2026_usd",
        "quality_adjusted_gross_value_real_2026_usd",
    ]
    for rate in discount_rates:
        label = rate_to_label(rate)
        preferred_columns.extend(
            [
                f"discount_factor_{label}",
                f"present_value_{label}_real_2026_usd",
            ]
        )
    preferred_columns.append("cumulative_quality_adjusted_value_real_2026_usd")
    for rate in discount_rates:
        label = rate_to_label(rate)
        preferred_columns.append(f"cumulative_present_value_{label}_real_2026_usd")

    extra_columns = [col for col in valuation.columns if col not in preferred_columns]
    return valuation[preferred_columns + extra_columns]


def summarize_valuation(
    valuation: pd.DataFrame, config: Mapping[str, Any]
) -> Dict[str, pd.DataFrame]:
    analysis = config["analysis"]
    selected_years = [int(x) for x in analysis["selected_years"]]
    discount_rates = [float(x) for x in analysis["real_discount_rates"]]

    value_columns = [
        "annual_material_tonnes",
        "gross_benchmark_value_real_2026_usd",
        "quality_adjusted_gross_value_real_2026_usd",
    ] + [
        f"present_value_{rate_to_label(rate)}_real_2026_usd"
        for rate in discount_rates
    ]

    total_group_cols = [
        "region",
        "capacity_scenario",
        "lifetime_scenario",
        "price_scenario",
        "year",
    ]
    annual_total = (
        valuation.groupby(total_group_cols, as_index=False, observed=True)[value_columns]
        .sum()
        .sort_values(total_group_cols)
    )

    selected_material = valuation[valuation["year"].isin(selected_years)].copy()
    selected_total = annual_total[annual_total["year"].isin(selected_years)].copy()

    scenario_group_cols = [
        "region",
        "capacity_scenario",
        "lifetime_scenario",
        "price_scenario",
    ]
    full_period = (
        annual_total.groupby(scenario_group_cols, as_index=False, observed=True)[value_columns]
        .sum()
        .rename(
            columns={
                "annual_material_tonnes": "full_period_material_tonnes",
                "gross_benchmark_value_real_2026_usd": "full_period_gross_benchmark_value_real_2026_usd",
                "quality_adjusted_gross_value_real_2026_usd": "full_period_quality_adjusted_value_real_2026_usd",
                **{
                    f"present_value_{rate_to_label(rate)}_real_2026_usd":
                    f"full_period_present_value_{rate_to_label(rate)}_real_2026_usd"
                    for rate in discount_rates
                },
            }
        )
        .sort_values(scenario_group_cols)
    )

    snapshot_sum = (
        selected_total.groupby(scenario_group_cols, as_index=False, observed=True)[value_columns]
        .sum()
        .rename(
            columns={
                "annual_material_tonnes": "selected_year_snapshot_material_tonnes",
                "gross_benchmark_value_real_2026_usd": "selected_year_snapshot_gross_benchmark_value_real_2026_usd",
                "quality_adjusted_gross_value_real_2026_usd": "selected_year_snapshot_quality_adjusted_value_real_2026_usd",
                **{
                    f"present_value_{rate_to_label(rate)}_real_2026_usd":
                    f"selected_year_snapshot_present_value_{rate_to_label(rate)}_real_2026_usd"
                    for rate in discount_rates
                },
            }
        )
        .sort_values(scenario_group_cols)
    )

    material_group_cols = scenario_group_cols + ["material"]
    full_period_by_material = (
        valuation.groupby(material_group_cols, as_index=False, observed=True)[value_columns]
        .sum()
        .rename(
            columns={
                "annual_material_tonnes": "full_period_material_tonnes",
                "gross_benchmark_value_real_2026_usd": "full_period_gross_benchmark_value_real_2026_usd",
                "quality_adjusted_gross_value_real_2026_usd": "full_period_quality_adjusted_value_real_2026_usd",
                **{
                    f"present_value_{rate_to_label(rate)}_real_2026_usd":
                    f"full_period_present_value_{rate_to_label(rate)}_real_2026_usd"
                    for rate in discount_rates
                },
            }
        )
        .sort_values(material_group_cols)
    )

    return {
        "annual_total": annual_total,
        "selected_years_material": selected_material,
        "selected_years_total": selected_total,
        "full_period_scenario_summary": full_period,
        "selected_year_snapshot_summary": snapshot_sum,
        "full_period_by_material": full_period_by_material,
    }


def _evaluate_modified_central_case(
    central_rows: pd.DataFrame,
    base_year: int,
    discount_rate: float,
    price_multiplier: float = 1.0,
    quantity_multiplier: float = 1.0,
    quality_discount_shift: float = 0.0,
    material_to_shift: Optional[str] = None,
) -> Tuple[float, float]:
    """Return cumulative PV and selected-year snapshot PV for a modified central case."""
    data = central_rows.copy()
    adjusted_discount = data["quality_discount"].astype(float)
    if material_to_shift is None:
        adjusted_discount = (adjusted_discount + quality_discount_shift).clip(0.0, 1.0)
    else:
        mask = data["material"].eq(material_to_shift)
        adjusted_discount.loc[mask] = (
            adjusted_discount.loc[mask] + quality_discount_shift
        ).clip(0.0, 1.0)

    modified_future_value = (
        data["annual_material_tonnes"]
        * quantity_multiplier
        * data["price_real_2026_usd_t"]
        * price_multiplier
        * (1.0 - adjusted_discount)
    )
    modified_pv = modified_future_value / np.power(
        1.0 + discount_rate, data["year"] - base_year
    )

    cumulative_pv = float(modified_pv.sum())
    selected_year_mask = data["year"].isin(data.attrs["selected_years"])
    snapshot_pv = float(modified_pv.loc[selected_year_mask].sum())
    return cumulative_pv, snapshot_pv


def run_sensitivity_analysis(
    valuation: pd.DataFrame, config: Mapping[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    analysis = config["analysis"]
    sensitivity = config["sensitivity"]
    base_year = int(analysis["base_year"])
    selected_years = [int(x) for x in analysis["selected_years"]]
    central_rate = float(analysis["central_discount_rate"])

    mask = (
        valuation["region"].eq(str(analysis["central_region"]))
        & valuation["capacity_scenario"].eq(str(analysis["central_capacity_scenario"]))
        & valuation["lifetime_scenario"].eq(str(analysis["central_lifetime_scenario"]))
        & valuation["price_scenario"].astype(str).eq(str(analysis["central_price_scenario"]))
    )
    central_rows = valuation.loc[mask].copy()
    if central_rows.empty:
        available = {
            "regions": sorted(valuation["region"].unique().tolist()),
            "capacity_scenarios": sorted(
                valuation["capacity_scenario"].unique().tolist()
            ),
            "lifetime_scenarios": sorted(
                valuation["lifetime_scenario"].unique().tolist()
            ),
            "price_scenarios": sorted(
                valuation["price_scenario"].astype(str).unique().tolist()
            ),
        }
        raise PipelineError(
            "Central sensitivity case is absent from the data. Available values: "
            f"{available}"
        )
    central_rows.attrs["selected_years"] = selected_years

    baseline_cum, baseline_snapshot = _evaluate_modified_central_case(
        central_rows, base_year, central_rate
    )

    price_change = float(sensitivity["commodity_price_change"])
    quantity_change = float(sensitivity["material_quantity_change"])
    quality_shift = float(
        sensitivity["quality_discount_change_percentage_points"]
    )

    global_cases = [
        (
            "Real discount rate",
            "Low",
            float(min(analysis["real_discount_rates"])),
            1.0,
            1.0,
            0.0,
        ),
        (
            "Real discount rate",
            "High",
            float(max(analysis["real_discount_rates"])),
            1.0,
            1.0,
            0.0,
        ),
        (
            "Commodity prices",
            "Low",
            central_rate,
            1.0 - price_change,
            1.0,
            0.0,
        ),
        (
            "Commodity prices",
            "High",
            central_rate,
            1.0 + price_change,
            1.0,
            0.0,
        ),
        (
            "Material quantities",
            "Low",
            central_rate,
            1.0,
            1.0 - quantity_change,
            0.0,
        ),
        (
            "Material quantities",
            "High",
            central_rate,
            1.0,
            1.0 + quantity_change,
            0.0,
        ),
        (
            "Quality discounts",
            "Favourable",
            central_rate,
            1.0,
            1.0,
            -quality_shift,
        ),
        (
            "Quality discounts",
            "Adverse",
            central_rate,
            1.0,
            1.0,
            quality_shift,
        ),
    ]

    global_rows: List[Dict[str, Any]] = []
    for variable, case, rate, price_mult, qty_mult, q_shift in global_cases:
        cum_pv, snapshot_pv = _evaluate_modified_central_case(
            central_rows,
            base_year,
            rate,
            price_multiplier=price_mult,
            quantity_multiplier=qty_mult,
            quality_discount_shift=q_shift,
        )
        global_rows.append(
            {
                "region": analysis["central_region"],
                "capacity_scenario": analysis["central_capacity_scenario"],
                "lifetime_scenario": analysis["central_lifetime_scenario"],
                "price_scenario": analysis["central_price_scenario"],
                "variable": variable,
                "case": case,
                "discount_rate": rate,
                "price_multiplier": price_mult,
                "quantity_multiplier": qty_mult,
                "quality_discount_shift_percentage_points": q_shift,
                "cumulative_pv_2026_to_2075_real_2026_usd": cum_pv,
                "selected_year_snapshot_pv_real_2026_usd": snapshot_pv,
                "baseline_cumulative_pv_real_2026_usd": baseline_cum,
                "baseline_snapshot_pv_real_2026_usd": baseline_snapshot,
                "cumulative_pv_change_percent": 100.0
                * (cum_pv - baseline_cum)
                / baseline_cum
                if baseline_cum != 0
                else np.nan,
                "snapshot_pv_change_percent": 100.0
                * (snapshot_pv - baseline_snapshot)
                / baseline_snapshot
                if baseline_snapshot != 0
                else np.nan,
            }
        )

    global_sensitivity = pd.DataFrame(global_rows)

    material_rows: List[Dict[str, Any]] = []
    for material in sorted(central_rows["material"].unique()):
        for case, shift in (("Favourable", -quality_shift), ("Adverse", quality_shift)):
            cum_pv, snapshot_pv = _evaluate_modified_central_case(
                central_rows,
                base_year,
                central_rate,
                quality_discount_shift=shift,
                material_to_shift=material,
            )
            material_rows.append(
                {
                    "region": analysis["central_region"],
                    "material": material,
                    "case": case,
                    "quality_discount_shift_percentage_points": shift,
                    "cumulative_pv_2026_to_2075_real_2026_usd": cum_pv,
                    "selected_year_snapshot_pv_real_2026_usd": snapshot_pv,
                    "baseline_cumulative_pv_real_2026_usd": baseline_cum,
                    "baseline_snapshot_pv_real_2026_usd": baseline_snapshot,
                    "cumulative_pv_change_percent": 100.0
                    * (cum_pv - baseline_cum)
                    / baseline_cum
                    if baseline_cum != 0
                    else np.nan,
                    "snapshot_pv_change_percent": 100.0
                    * (snapshot_pv - baseline_snapshot)
                    / baseline_snapshot
                    if baseline_snapshot != 0
                    else np.nan,
                }
            )

    material_quality_sensitivity = pd.DataFrame(material_rows)
    return global_sensitivity, material_quality_sensitivity


def billions_formatter(value: float, _position: int) -> str:
    return f"{value / 1e9:,.1f}"


def millions_formatter(value: float, _position: int) -> str:
    return f"{value / 1e6:,.1f}"


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_price_charts(price_pathways: pd.DataFrame, charts_dir: Path) -> None:
    for material, subset in price_pathways.groupby("material", observed=True):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for scenario in SCENARIO_ORDER:
            series = subset[subset["price_scenario"].astype(str).eq(scenario)]
            ax.plot(
                series["year"],
                series["price_real_2026_usd_t"],
                linewidth=2,
                label=scenario,
            )
        ax.set_title(f"{material}: real commodity-price pathways")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price (constant 2026 USD/t)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        safe_name = material.lower().replace(" ", "_")
        save_figure(fig, charts_dir / f"price_pathway_{safe_name}.png")


def create_valuation_charts(
    valuation: pd.DataFrame,
    summaries: Mapping[str, pd.DataFrame],
    global_sensitivity: pd.DataFrame,
    config: Mapping[str, Any],
    charts_dir: Path,
) -> None:
    analysis = config["analysis"]
    plot_mask = (
        valuation["region"].eq(str(analysis["plot_region"]))
        & valuation["capacity_scenario"].eq(
            str(analysis["plot_capacity_scenario"])
        )
        & valuation["lifetime_scenario"].eq(
            str(analysis["plot_lifetime_scenario"])
        )
    )
    plot_data = valuation.loc[plot_mask].copy()
    if plot_data.empty:
        LOGGER.warning("Plot scenario is absent; valuation charts were skipped.")
        return

    annual = (
        plot_data.groupby(["price_scenario", "year"], as_index=False, observed=True)[
            "quality_adjusted_gross_value_real_2026_usd"
        ]
        .sum()
        .sort_values(["price_scenario", "year"])
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in SCENARIO_ORDER:
        subset = annual[annual["price_scenario"].astype(str).eq(scenario)]
        ax.plot(
            subset["year"],
            subset["quality_adjusted_gross_value_real_2026_usd"],
            linewidth=2,
            label=scenario,
        )
    ax.set_title(
        f"{analysis['plot_region']}: annual quality-adjusted gross material value\n"
        f"{str(analysis['plot_capacity_scenario']).replace('_', ' ')} capacity / "
        f"{str(analysis['plot_lifetime_scenario']).replace('_', ' ')} retirement"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual value (billion constant 2026 USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(billions_formatter))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, charts_dir / "annual_future_material_value_price_scenarios.png")

    central_rate = float(analysis["central_discount_rate"])
    central_label = rate_to_label(central_rate)
    pv_column = f"present_value_{central_label}_real_2026_usd"
    annual_pv = (
        plot_data.groupby(["price_scenario", "year"], as_index=False, observed=True)[
            pv_column
        ]
        .sum()
        .sort_values(["price_scenario", "year"])
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in SCENARIO_ORDER:
        subset = annual_pv[annual_pv["price_scenario"].astype(str).eq(scenario)]
        ax.plot(
            subset["year"],
            subset[pv_column],
            linewidth=2,
            label=scenario,
        )
    ax.set_title(
        f"{analysis['plot_region']}: present value of annual material opportunity\n"
        f"Real discount rate = {central_rate:.0%}"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Present value (billion constant 2026 USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(billions_formatter))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, charts_dir / "annual_present_value_price_scenarios.png")

    selected_years = [int(x) for x in analysis["selected_years"]]
    mix = plot_data[
        plot_data["price_scenario"].astype(str).eq("Reference")
        & plot_data["year"].isin(selected_years)
    ]
    mix_pivot = mix.pivot_table(
        index="year",
        columns="material",
        values="quality_adjusted_gross_value_real_2026_usd",
        aggfunc="sum",
        fill_value=0.0,
        observed=True,
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    mix_pivot.div(1e9).plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(
        f"{analysis['plot_region']}: material contribution in selected years\n"
        "Reference commodity-price pathway"
    )
    ax.set_xlabel("Assessment year")
    ax.set_ylabel("Quality-adjusted gross value (billion constant 2026 USD)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Material", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, charts_dir / "selected_year_material_contribution.png")

    # Global OAT tornado chart using cumulative PV over the complete analysis period.
    tornado = global_sensitivity.pivot(
        index="variable", columns="case", values="cumulative_pv_change_percent"
    )
    lower_candidates = [c for c in ["Low", "Adverse", "Favourable"] if c in tornado]
    upper_candidates = [c for c in ["High", "Favourable", "Adverse"] if c in tornado]

    tornado_rows: List[Dict[str, Any]] = []
    for variable, row in tornado.iterrows():
        values = row.dropna().to_dict()
        if not values:
            continue
        low_case = min(values, key=values.get)
        high_case = max(values, key=values.get)
        tornado_rows.append(
            {
                "variable": variable,
                "low": values[low_case],
                "high": values[high_case],
                "low_case": low_case,
                "high_case": high_case,
                "range": max(abs(values[low_case]), abs(values[high_case])),
            }
        )
    tornado_plot = pd.DataFrame(tornado_rows).sort_values("range", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(tornado_plot))
    ax.barh(y, tornado_plot["low"], label="Lower outcome")
    ax.barh(y, tornado_plot["high"], label="Higher outcome")
    ax.set_yticks(y, tornado_plot["variable"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Change in cumulative present value relative to central case (%)")
    ax.set_title(
        f"One-at-a-time sensitivity: {analysis['central_region']} central case"
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, charts_dir / "sensitivity_tornado_cumulative_pv.png")



def run_internal_validation_checks(
    material_data: pd.DataFrame,
    price_pathways: pd.DataFrame,
    valuation: pd.DataFrame,
    summaries: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Run deterministic consistency checks and return a publication audit table."""
    analysis = config["analysis"]
    base_year = int(analysis["base_year"])
    discount_rates = sorted(float(x) for x in analysis["real_discount_rates"])

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, observed: Any, expected: Any, note: str = "") -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "note": note,
            }
        )

    expected_price_rows = len(config["materials"]) * len(SCENARIO_ORDER) * (
        int(analysis["end_year"]) - base_year + 1
    )
    add_check(
        "price_pathway_row_count",
        len(price_pathways) == expected_price_rows,
        len(price_pathways),
        expected_price_rows,
    )

    base_rows = price_pathways[price_pathways["year"].eq(base_year)]
    max_base_error = 0.0
    for material, item in config["materials"].items():
        expected = float(item["base_price_usd_t"])
        observed = base_rows.loc[
            base_rows["material"].eq(material), "price_real_2026_usd_t"
        ]
        if not observed.empty:
            max_base_error = max(max_base_error, float((observed - expected).abs().max()))
    add_check(
        "base_year_prices_equal_config",
        max_base_error < 1e-9,
        max_base_error,
        "< 1e-9 USD/t absolute error",
    )

    expected_valuation_rows = len(material_data) * len(SCENARIO_ORDER)
    add_check(
        "valuation_row_count",
        len(valuation) == expected_valuation_rows,
        len(valuation),
        expected_valuation_rows,
    )

    negative_value_rows = int(
        (valuation["quality_adjusted_gross_value_real_2026_usd"] < -1e-9).sum()
    )
    add_check(
        "no_negative_quality_adjusted_values",
        negative_value_rows == 0,
        negative_value_rows,
        0,
    )

    excess_rows = int(
        (
            valuation["quality_adjusted_gross_value_real_2026_usd"]
            - valuation["gross_benchmark_value_real_2026_usd"]
            > 1e-6
        ).sum()
    )
    add_check(
        "quality_adjusted_value_not_above_gross_value",
        excess_rows == 0,
        excess_rows,
        0,
    )

    if len(discount_rates) >= 2:
        pv_columns = [
            f"present_value_{rate_to_label(rate)}_real_2026_usd"
            for rate in discount_rates
        ]
        monotonic_violations = 0
        for low_col, high_col in zip(pv_columns[:-1], pv_columns[1:]):
            monotonic_violations += int(
                (valuation[high_col] - valuation[low_col] > 1e-6).sum()
            )
        add_check(
            "present_value_nonincreasing_with_discount_rate",
            monotonic_violations == 0,
            monotonic_violations,
            0,
        )

    combinations = (
        valuation[["region", "capacity_scenario", "lifetime_scenario", "price_scenario"]]
        .drop_duplicates()
        .groupby("region", observed=True)
        .size()
    )
    min_combinations = int(combinations.min())
    max_combinations = int(combinations.max())
    add_check(
        "twenty_seven_scenario_combinations_per_region",
        min_combinations == 27 and max_combinations == 27,
        f"min={min_combinations}, max={max_combinations}",
        "27 per region",
    )

    selected_years = set(int(x) for x in analysis["selected_years"])
    available_years = set(int(x) for x in valuation["year"].unique())
    missing_selected = sorted(selected_years - available_years)
    add_check(
        "selected_years_available",
        not missing_selected,
        missing_selected,
        [],
    )

    summary_rows = len(summaries["full_period_scenario_summary"])
    expected_summary_rows = valuation[\
        ["region", "capacity_scenario", "lifetime_scenario", "price_scenario"]
    ].drop_duplicates().shape[0]
    add_check(
        "full_period_summary_row_count",
        summary_rows == expected_summary_rows,
        summary_rows,
        expected_summary_rows,
    )

    result = pd.DataFrame(checks)
    failures = result.loc[~result["passed"]]
    if not failures.empty:
        raise PipelineError(
            "Internal validation checks failed: "
            + failures[["check", "observed", "expected"]].to_dict(orient="records").__repr__()
        )
    return result


def write_methodology_note(
    output_path: Path,
    input_csv: Path,
    config_path: Path,
    config: Mapping[str, Any],
) -> None:
    analysis = config["analysis"]
    materials = config["materials"]

    lines = [
        "PV EOL ECONOMIC VALUATION - METHODOLOGY AND INTERPRETATION",
        "=" * 67,
        "",
        f"Input CSV: {input_csv}",
        f"Configuration: {config_path}",
        f"Valuation period: {analysis['base_year']} to {analysis['end_year']}",
        f"Selected assessment years: {analysis['selected_years']}",
        "Currency basis: constant 2026 USD",
        "",
        "1. VALUE DEFINITION",
        "The model estimates quality-adjusted gross material-value potential contained in",
        "end-of-life crystalline-silicon PV waste. It does not estimate recycling profit,",
        "net revenue, or net cash flow.",
        "",
        "2. EXCLUDED ITEMS",
        "No deduction is made for physical recovery losses, collection, transport, dismantling,",
        "processing, refining, energy, labour, CAPEX, OPEX, taxes, or transaction costs.",
        "",
        "3. QUALITY ADJUSTMENT",
        "Annual value = annual material tonnes x commodity price x (1 - quality discount).",
        "The discounts are transparent study assumptions. They are not universal values",
        "reported by a single publication.",
        "",
    ]
    for material, item in materials.items():
        lines.append(
            f"- {material}: base price={float(item['base_price_usd_t']):,.2f} USD/t; "
            f"discount={float(item['quality_discount']):.1%}; "
            f"basis={item.get('quality_discount_basis', '')}"
        )

    lines.extend(
        [
            "",
            "4. PRICE PATHWAYS",
            "Downside, Reference, and Upside pathways use a two-phase compound real-growth",
            "model: 2027-2035 and 2036-2075. These pathways are scenario assumptions, not",
            "point forecasts.",
            "",
            "5. PRESENT VALUE",
            "PV_t = future material value_t / (1 + real discount rate)^(t - 2026).",
            f"Real discount rates: {analysis['real_discount_rates']}; central rate: "
            f"{analysis['central_discount_rate']:.1%}.",
            "Because commodity prices are in constant 2026 USD, real discount rates are used.",
            "",
            "6. CUMULATIVE VERSUS SNAPSHOT RESULTS",
            "Full-period cumulative PV sums all annual flows from 2026 through 2075.",
            "The selected-year snapshot sum includes only 2045, 2060, and 2075 and must not",
            "be described as cumulative 2026-2075 revenue.",
            "",
            "7. SENSITIVITY ANALYSIS",
            "The OAT sensitivity changes one input at a time around the central case:",
            "real discount rate (3%/7%), commodity prices (+/-10%), material quantities",
            "(+/-10%), and all quality discounts (+/-10 percentage points, bounded 0-100%).",
            "A separate table changes each material quality discount individually.",
            "",
            "8. CORE REFERENCES USED TO FRAME THE MODEL",
            "- User-supplied economic methodology section (two-phase price pathways, real PV,",
            "  scenario/sensitivity structure, and interpretation as gross material value).",
            "- Deng, R., Chang, N. L., Ouyang, Z., & Chong, C. M. (2019). A techno-economic",
            "  review of silicon photovoltaic module recycling. Renewable and Sustainable",
            "  Energy Reviews, 109, 532-550. DOI: 10.1016/j.rser.2019.04.020.",
            "- Heath, G. A. et al. (2020). Research and development priorities for silicon",
            "  photovoltaic module recycling to support a circular economy. Nature Energy,",
            "  5, 502-510. DOI: 10.1038/s41560-020-0645-2.",
            "- Komoto, K. et al. (2025). Status of PV Module Recycling in IEA PVPS Task 12",
            "  Countries. IEA-PVPS T12-31:2025. DOI: 10.69766/XLFG7020.",
            "- Tammaro, M. et al. (2020). Development and techno-economic analysis of an",
            "  advanced recycling process for photovoltaic panels enabling polymer separation",
            "  and recovery of Ag and Si. Energies, 13, 6690. DOI: 10.3390/en13246690.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    paths: PipelinePaths,
    input_csv: Path,
    config_path: Path,
    config: Mapping[str, Any],
    quality_report: pd.DataFrame,
    assumptions: pd.DataFrame,
    prices: pd.DataFrame,
    valuation: pd.DataFrame,
    summaries: Mapping[str, pd.DataFrame],
    global_sensitivity: pd.DataFrame,
    material_sensitivity: pd.DataFrame,
    validation_checks: pd.DataFrame,
) -> None:
    quality_report.to_csv(paths.tables / "data_quality_report.csv", index=False)
    assumptions.to_csv(paths.tables / "model_assumptions.csv", index=False)
    prices.to_csv(paths.tables / "commodity_price_pathways_2026_2075.csv", index=False)
    valuation.to_csv(paths.tables / "valuation_annual_long.csv", index=False)

    file_names = {
        "annual_total": "valuation_annual_total.csv",
        "selected_years_material": "valuation_selected_years_by_material.csv",
        "selected_years_total": "valuation_selected_years_total.csv",
        "full_period_scenario_summary": "valuation_full_period_2026_2075_summary.csv",
        "selected_year_snapshot_summary": "valuation_selected_year_snapshot_summary.csv",
        "full_period_by_material": "valuation_full_period_by_material.csv",
    }
    for key, frame in summaries.items():
        frame.to_csv(paths.tables / file_names[key], index=False)

    global_sensitivity.to_csv(
        paths.tables / "sensitivity_global_one_at_a_time.csv", index=False
    )
    material_sensitivity.to_csv(
        paths.tables / "sensitivity_quality_discount_by_material.csv", index=False
    )
    validation_checks.to_csv(
        paths.tables / "model_validation_checks.csv", index=False
    )

    write_methodology_note(
        paths.root / "METHODOLOGY_AND_INTERPRETATION.txt",
        input_csv=input_csv,
        config_path=config_path,
        config=config,
    )

    # Keep an exact copy of the configuration used for reproducibility.
    shutil.copy2(config_path, paths.root / "config_used.json")


def create_manifest(paths: PipelinePaths) -> None:
    records: List[Dict[str, Any]] = []
    for path in sorted(paths.root.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "relative_path": str(path.relative_to(paths.root)),
                    "size_bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(records).to_csv(paths.root / "OUTPUT_MANIFEST.csv", index=False)


def run_pipeline(
    input_csv: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> Path:
    """Run the complete pipeline and return the output directory."""
    input_path = Path(input_csv).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()

    paths = make_output_paths(output_path)
    configure_logging(paths.logs / "pipeline.log")

    LOGGER.info("Starting PV EOL economic valuation pipeline")
    LOGGER.info("Input: %s", input_path)
    LOGGER.info("Configuration: %s", config_file)
    LOGGER.info("Output: %s", paths.root)

    config = load_config(config_file)
    material_data, quality_report = load_and_validate_input(input_path, config)
    LOGGER.info("Validated %s material-year rows for valuation", len(material_data))

    assumptions = build_assumption_table(config)
    prices = build_price_pathways(config)
    valuation = calculate_valuation(material_data, prices, config)
    summaries = summarize_valuation(valuation, config)
    global_sensitivity, material_sensitivity = run_sensitivity_analysis(
        valuation, config
    )
    validation_checks = run_internal_validation_checks(
        material_data, prices, valuation, summaries, config
    )

    write_outputs(
        paths=paths,
        input_csv=input_path,
        config_path=config_file,
        config=config,
        quality_report=quality_report,
        assumptions=assumptions,
        prices=prices,
        valuation=valuation,
        summaries=summaries,
        global_sensitivity=global_sensitivity,
        material_sensitivity=material_sensitivity,
        validation_checks=validation_checks,
    )

    create_price_charts(prices, paths.charts)
    create_valuation_charts(
        valuation, summaries, global_sensitivity, config, paths.charts
    )
    create_manifest(paths)

    LOGGER.info("Pipeline completed successfully")
    LOGGER.info("Main long-form valuation rows: %s", len(valuation))
    LOGGER.info("Outputs written to: %s", paths.root)
    return paths.root


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-adjusted economic valuation of EOL c-Si PV materials."
    )
    parser.add_argument(
        "--input",
        default="pv_material_waste_all_years_long.csv",
        help="Path to the long-form material-waste CSV.",
    )
    parser.add_argument(
        "--config",
        default="pv_eol_config.json",
        help="Path to the JSON assumptions/configuration file.",
    )
    parser.add_argument(
        "--output",
        default="pv_eol_economic_outputs",
        help="Output directory.",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        LOGGER.debug("Ignoring unrecognized arguments: %s", unknown)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output_path = run_pipeline(
            input_csv=args.input,
            output_dir=args.output,
            config_path=args.config,
        )
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Defensive top-level handler with traceback in log/console.
        LOGGER.exception("Unexpected pipeline failure")
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Completed. Outputs: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())