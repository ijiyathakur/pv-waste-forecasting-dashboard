# %% [markdown]
# waste accumulation prediction

# %%
#!/usr/bin/env python3
"""
Solar-PV end-of-life (EoL) waste model for India and five states.

FIX APPLIED (see "BUGFIX" comments below)
------------------------------------------
The original script matched historical rows on
`series.eq(capacity_scenario)` (i.e. it looked for historical rows labeled
"Conservative"/"Medium"/"High"). In the actual CSV schema, historical rows
are labeled `series == "Actual"` (a single series, not one per scenario).
Because of that mismatch, the historical filter never matched anything, and
every scenario's installation cohort silently started in 2027 instead of
2014 - all real, already-installed historical capacity (2014-2026,
~150 GW nationally) was dropped from every run with no error or warning.

The fix: match historical rows on `series.eq("Actual")` instead of
`series.eq(capacity_scenario)`. Historical data does not vary by
capacity/loss scenario, so it should be pulled in once and shared across
all scenario runs, while forecast rows (which DO vary by scenario) still
match on `series.eq(capacity_scenario)` as before.

STYLE UPDATE
------------
- Global plotting font switched to Times New Roman.
- The multi-panel figures no longer use a figure-level suptitle or an
  axis-level title. Instead, each individual subplot carries a small
  top-left label reading "<Capacity Scenario> scenario - <Region>"
  (e.g. "Conservative scenario - India").
"""

from __future__ import annotations

import glob
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

# STYLE UPDATE: use Times New Roman (falls back to the next available serif
# font if Times New Roman itself isn't installed on the machine running this
# script) for all text in every chart produced below.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


# =============================================================================
# PARAMETERS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_ROOT = BASE_DIR / "outputs"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BASE_CAPACITY_INPUT_FILE = (
    DATA_DIR
    / "pv_capacity_additions_fiscal_year_model_ready_2014_2050_updated.csv"
)

FORECAST_ADDITIONS_FILE = (
    OUTPUT_ROOT
    / "pv_annual_additions_all_scenarios_2027_2050_gw.csv"
)

UPDATED_CAPACITY_INPUT_FILE = (
    DATA_DIR
    / "pv_capacity_additions_model_ready_updated_from_capacity_model.csv"
)

INPUT_FILENAME_HINT = "pv_capacity_additions"

KAGGLE_INPUT_ROOT = str(DATA_DIR)
KAGGLE_OUTPUT_ROOT = str(OUTPUT_ROOT)

FALLBACK_INPUT_ROOT = str(DATA_DIR)
FALLBACK_OUTPUT_ROOT = str(OUTPUT_ROOT)

OUTPUT_DIR_NAME = "pv_eol_model_outputs"

MASS_TONNES_PER_MW = 70.0
MAX_RETIREMENT_AGE = 80
OPENING_STOCK_MODE = "as_2014_cohort"
APPLY_OPTIONAL_EARLY_FAILURES = True
FIRST_YEAR_FAILURE = 0.005
SECOND_YEAR_FAILURE = 0.005
SHOW_PLOTS_INLINE = False

PLOT_YEAR_MIN = 2014
PLOT_YEAR_MAX = 2075

PLOT_YEAR_TICK_STEP = 5

LINE_MARKER = "o"
LINE_MARKER_SIZE = 3.5

TONNES_PER_MILLION_TONNES = 1_000_000.0


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

CAPACITY_SCENARIOS = ("Conservative", "Medium", "High")
LIFETIME_SCENARIOS = ("Fixed_25", "Early_Loss", "Regular_Loss")

FIXED_LIFETIME_YEARS = 25
WEIBULL_PARAMETERS: Dict[str, Tuple[float, float]] = {
    "Early_Loss": (2.4928, 30.0),
    "Regular_Loss": (5.3759, 30.0),
}

# BUGFIX: historical rows in this CSV schema always carry series == "Actual",
# never "Conservative"/"Medium"/"High". Matching this label (rather than
# capacity_scenario) is what makes the historical filter actually find rows.
HISTORICAL_SERIES_LABEL = "Actual"

REQUIRED_COLUMNS = {
    "region", "fiscal_year_end_year", "data_type", "series",
    "record_type", "capacity_gw",
}

EXPECTED_REGIONS = {
    "India", "Gujarat", "Karnataka", "Maharashtra", "Rajasthan", "Tamil Nadu"
}


# =============================================================================
# INPUT DISCOVERY / VALIDATION
# =============================================================================
def build_updated_capacity_input(
    base_input_path: Path,
    forecast_additions_path: Path,
    output_path: Path,
) -> Path:
    """
    Keep historical capacity records from the existing model-ready file
    and replace its forecast records with the latest capacity-model output.
    """

    if not base_input_path.exists():
        raise FileNotFoundError(
            f"Historical capacity input file not found: {base_input_path}"
        )

    if not forecast_additions_path.exists():
        raise FileNotFoundError(
            f"Forecast annual-additions file not found: "
            f"{forecast_additions_path}"
        )

    base_df = pd.read_csv(base_input_path)
    forecast_df = pd.read_csv(forecast_additions_path)

    required_base_columns = {
        "region",
        "fiscal_year_end_year",
        "data_type",
        "series",
        "record_type",
        "capacity_gw",
    }

    missing_base_columns = (
        required_base_columns.difference(base_df.columns)
    )

    if missing_base_columns:
        raise ValueError(
            "The historical capacity input is missing columns: "
            f"{sorted(missing_base_columns)}"
        )

    required_forecast_columns = {
        "state",
        "year",
        "Conservative",
        "Medium",
        "High",
    }

    missing_forecast_columns = (
        required_forecast_columns.difference(forecast_df.columns)
    )

    if missing_forecast_columns:
        raise ValueError(
            "The capacity-model annual-additions file is missing columns: "
            f"{sorted(missing_forecast_columns)}"
        )

    # Keep only historical records from the original model-ready file.
    historical_df = base_df.loc[
        base_df["data_type"].astype(str).eq("Historical"),
        [
            "region",
            "fiscal_year_end_year",
            "data_type",
            "series",
            "record_type",
            "capacity_gw",
        ],
    ].copy()

    # Convert the latest wide capacity output into the long format
    # required by the waste model.
    forecast_long_df = forecast_df.melt(
        id_vars=["state", "year"],
        value_vars=["Conservative", "Medium", "High"],
        var_name="series",
        value_name="capacity_gw",
    )

    forecast_long_df = forecast_long_df.rename(
        columns={
            "state": "region",
            "year": "fiscal_year_end_year",
        }
    )

    forecast_long_df["data_type"] = "Forecast"
    forecast_long_df["record_type"] = "Annual addition"

    forecast_long_df = forecast_long_df[
        [
            "region",
            "fiscal_year_end_year",
            "data_type",
            "series",
            "record_type",
            "capacity_gw",
        ]
    ]

    forecast_long_df["capacity_gw"] = pd.to_numeric(
        forecast_long_df["capacity_gw"],
        errors="raise",
    )

    # Remove only extremely small negative floating-point values.
    tiny_negative_mask = forecast_long_df["capacity_gw"].between(
        -1e-9,
        0,
        inclusive="left",
    )

    forecast_long_df.loc[
        tiny_negative_mask,
        "capacity_gw",
    ] = 0.0

    significant_negative_mask = (
        forecast_long_df["capacity_gw"] < 0
    )

    if significant_negative_mask.any():
        bad_rows = forecast_long_df.loc[
            significant_negative_mask
        ].head(10)

        raise ValueError(
            "The capacity model generated negative annual additions. "
            "Review these rows before running the waste model:\n"
            + bad_rows.to_string(index=False)
        )

    combined_df = pd.concat(
        [
            historical_df,
            forecast_long_df,
        ],
        ignore_index=True,
    )

    combined_df["fiscal_year_end_year"] = pd.to_numeric(
        combined_df["fiscal_year_end_year"],
        errors="raise",
    ).astype(int)

    combined_df = combined_df.sort_values(
        [
            "region",
            "fiscal_year_end_year",
            "data_type",
            "series",
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nUpdated model-ready capacity input created:"
    )
    print(output_path)

    print(
        "\nHistorical period:"
        f" {historical_df['fiscal_year_end_year'].min()}"
        f"-{historical_df['fiscal_year_end_year'].max()}"
    )

    print(
        "Forecast period:"
        f" {forecast_long_df['fiscal_year_end_year'].min()}"
        f"-{forecast_long_df['fiscal_year_end_year'].max()}"
    )

    return output_path

def find_input_csv(explicit_path: str | None, root: str, hint: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        print(f"Note: INPUT_CSV_PATH '{explicit_path}' not found next to the script; "
              f"searching '{root}' instead...")

    candidates = sorted(glob.glob(os.path.join(root, "**", f"*{hint}*.csv"), recursive=True))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found under {root}. "
            "Place the CSV in the same folder as this script (or update INPUT_CSV_PATH)."
        )
    if len(candidates) > 1:
        print(f"Multiple CSV candidates found under {root}; using the first match:")
        for c in candidates:
            print(f"  - {c}")
    return Path(candidates[0])


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    missing_regions = EXPECTED_REGIONS.difference(set(df["region"].dropna().unique()))
    if missing_regions:
        raise ValueError(f"Expected regions not found in CSV: {sorted(missing_regions)}")

    invalid_capacity = df["capacity_gw"].isna() | (df["capacity_gw"] < 0)
    if invalid_capacity.any():
        bad_rows = df.loc[invalid_capacity].head(10)
        raise ValueError(
            "capacity_gw contains missing or negative values. Example rows:\n"
            + bad_rows.to_string(index=False)
        )

    # BUGFIX / SANITY CHECK: warn loudly (instead of failing silently) if the
    # historical rows ever stop matching HISTORICAL_SERIES_LABEL, so this
    # class of bug can't reoccur unnoticed.
    hist = df.loc[df["data_type"].eq("Historical")]
    if not hist.empty:
        bad_series = set(hist["series"].dropna().unique()) - {HISTORICAL_SERIES_LABEL}
        if bad_series:
            raise ValueError(
                f"Historical rows contain unexpected series label(s) {sorted(bad_series)}; "
                f"expected only '{HISTORICAL_SERIES_LABEL}'. Update HISTORICAL_SERIES_LABEL "
                "or investigate the input file before trusting model output."
            )


# =============================================================================
# RETIREMENT / LIFETIME DISTRIBUTIONS
# =============================================================================

def weibull_retirement_probabilities(shape: float, scale: float, max_age: int) -> pd.Series:
    ages = np.arange(1, max_age + 1, dtype=float)
    cdf_now = 1.0 - np.exp(-np.power(ages / scale, shape))
    cdf_previous = 1.0 - np.exp(-np.power((ages - 1.0) / scale, shape))
    probabilities = cdf_now - cdf_previous

    residual_tail = 1.0 - probabilities.sum()
    probabilities[-1] += residual_tail

    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return pd.Series(probabilities, index=ages.astype(int), name="retirement_fraction")


def build_loss_distributions(
    max_age: int,
    apply_optional_early_failures: bool,
    first_year_failure: float,
    second_year_failure: float,
) -> Dict[str, pd.Series]:
    if max_age < FIXED_LIFETIME_YEARS:
        raise ValueError(f"max_age must be at least {FIXED_LIFETIME_YEARS} years for Fixed_25.")

    fixed = pd.Series(0.0, index=np.arange(1, max_age + 1), name="retirement_fraction")
    fixed.loc[FIXED_LIFETIME_YEARS] = 1.0

    early_shape, early_scale = WEIBULL_PARAMETERS["Early_Loss"]
    early = weibull_retirement_probabilities(early_shape, early_scale, max_age)

    if apply_optional_early_failures:
        front_loaded = first_year_failure + second_year_failure
        if not (0.0 <= front_loaded < 1.0):
            raise ValueError("The sum of optional year-1 and year-2 failures must be < 1.")

        later = early.copy()
        later.loc[1:2] = 0.0
        later /= later.sum()
        early = later * (1.0 - front_loaded)
        early.loc[1] = first_year_failure
        early.loc[2] = second_year_failure

    regular_shape, regular_scale = WEIBULL_PARAMETERS["Regular_Loss"]
    regular = weibull_retirement_probabilities(regular_shape, regular_scale, max_age)

    distributions = {"Fixed_25": fixed, "Early_Loss": early, "Regular_Loss": regular}

    for name, probabilities in distributions.items():
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            raise RuntimeError(f"Retirement probabilities for {name} do not sum to 1.")

    return distributions


# =============================================================================
# INSTALLATION COHORTS & WASTE CALCULATION
# =============================================================================

def _dedupe_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that are replicated across an extra dimension (e.g. a
    historical row appearing under more than one loss_scenario) down to a
    single row per fiscal year. capacity_gw is identical across those
    replicated rows, so `first` is safe and avoids accidentally summing the
    same installation multiple times."""
    if frame.empty:
        return frame
    return (
        frame.groupby("fiscal_year_end_year", as_index=False)["capacity_gw"]
        .first()
    )


def prepare_installation_cohorts(
    df: pd.DataFrame,
    region: str,
    capacity_scenario: str,
    mass_tonnes_per_mw: float,
    opening_stock_mode: str,
) -> pd.DataFrame:
    regional = df.loc[df["region"].eq(region)].copy()

    # BUGFIX: historical rows are labeled series == "Actual" in this CSV
    # schema (a single shared series, not one per capacity scenario), so we
    # match on HISTORICAL_SERIES_LABEL here rather than on capacity_scenario.
    # capacity_scenario is still used below for the *forecast* rows, which do
    # genuinely vary by scenario.
    historical = _dedupe_by_year(regional.loc[
        regional["data_type"].eq("Historical")
        & regional["series"].eq(HISTORICAL_SERIES_LABEL)
        & regional["record_type"].eq("Annual addition"),
        ["fiscal_year_end_year", "capacity_gw"],
    ])

    if opening_stock_mode == "as_2014_cohort":
        opening_stock = _dedupe_by_year(regional.loc[
            regional["data_type"].eq("Historical")
            & regional["series"].eq(HISTORICAL_SERIES_LABEL)
            & regional["record_type"].eq("Opening stock"),
            ["fiscal_year_end_year", "capacity_gw"],
        ])
        historical = pd.concat([opening_stock, historical], ignore_index=True)
    elif opening_stock_mode != "exclude":
        raise ValueError("opening_stock_mode must be 'as_2014_cohort' or 'exclude'.")

    forecast = _dedupe_by_year(regional.loc[
        regional["data_type"].eq("Forecast")
        & regional["series"].eq(capacity_scenario)
        & regional["record_type"].eq("Annual addition"),
        ["fiscal_year_end_year", "capacity_gw"],
    ])

    cohorts = pd.concat([historical, forecast], ignore_index=True)
    cohorts = (
        cohorts.groupby("fiscal_year_end_year", as_index=False)["capacity_gw"]
        .sum()
        .sort_values("fiscal_year_end_year")
    )

    cohorts["capacity_mw"] = cohorts["capacity_gw"] * 1000.0
    cohorts["installed_mass_tonnes"] = cohorts["capacity_mw"] * mass_tonnes_per_mw
    return cohorts


def calculate_annual_waste(
    cohorts: pd.DataFrame,
    retirement_probabilities: pd.Series,
    output_years: Iterable[int],
) -> pd.DataFrame:
    output_years = np.asarray(list(output_years), dtype=int)
    waste_by_year = pd.Series(0.0, index=output_years, dtype=float)

    for row in cohorts.itertuples(index=False):
        installation_year = int(row.fiscal_year_end_year)
        installed_mass = float(row.installed_mass_tonnes)

        retirement_years = installation_year + retirement_probabilities.index.to_numpy()
        cohort_waste = installed_mass * retirement_probabilities.to_numpy()

        valid = np.isin(retirement_years, output_years)
        if valid.any():
            waste_by_year.loc[retirement_years[valid]] += cohort_waste[valid]

    result = waste_by_year.rename("annual_waste_tonnes").reset_index()
    result.columns = ["year", "annual_waste_tonnes"]
    result["cumulative_waste_tonnes"] = result["annual_waste_tonnes"].cumsum()
    return result


# =============================================================================
# FULL MODEL RUN
# =============================================================================

def run_model(
    df: pd.DataFrame,
    mass_tonnes_per_mw: float,
    max_retirement_age: int,
    opening_stock_mode: str,
    apply_optional_early_failures: bool,
    first_year_failure: float,
    second_year_failure: float,
    plot_year_max: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    distributions = build_loss_distributions(
        max_age=max_retirement_age,
        apply_optional_early_failures=apply_optional_early_failures,
        first_year_failure=first_year_failure,
        second_year_failure=second_year_failure,
    )

    regions = sorted(df["region"].dropna().unique(), key=lambda x: (x != "India", x))
    first_installation_year = int(df["fiscal_year_end_year"].min())
    last_installation_year = int(df["fiscal_year_end_year"].max())

    last_output_year = max(last_installation_year + max_retirement_age, plot_year_max)
    output_years = range(first_installation_year, last_output_year + 1)

    model_outputs = []
    summary_rows = []

    total_runs = len(regions) * len(CAPACITY_SCENARIOS) * len(LIFETIME_SCENARIOS)
    run_number = 0

    for region in regions:
        for capacity_scenario in CAPACITY_SCENARIOS:
            cohorts = prepare_installation_cohorts(
                df=df, region=region, capacity_scenario=capacity_scenario,
                mass_tonnes_per_mw=mass_tonnes_per_mw, opening_stock_mode=opening_stock_mode,
            )
            installed_mass_total = cohorts["installed_mass_tonnes"].sum()

            for lifetime_scenario in LIFETIME_SCENARIOS:
                run_number += 1
                annual = calculate_annual_waste(
                    cohorts=cohorts,
                    retirement_probabilities=distributions[lifetime_scenario],
                    output_years=output_years,
                )
                annual.insert(0, "lifetime_scenario", lifetime_scenario)
                annual.insert(0, "capacity_scenario", capacity_scenario)
                annual.insert(0, "region", region)
                model_outputs.append(annual)

                peak_index = annual["annual_waste_tonnes"].idxmax()
                peak_row = annual.loc[peak_index]
                positive = annual.loc[annual["annual_waste_tonnes"] >= 1.0, "year"]

                summary_rows.append({
                    "region": region,
                    "capacity_scenario": capacity_scenario,
                    "lifetime_scenario": lifetime_scenario,
                    "modelled_installed_mass_million_tonnes": installed_mass_total / TONNES_PER_MILLION_TONNES,
                    "total_eol_waste_million_tonnes": annual["annual_waste_tonnes"].sum() / TONNES_PER_MILLION_TONNES,
                    "peak_waste_year": int(peak_row["year"]),
                    "peak_annual_waste_million_tonnes": float(peak_row["annual_waste_tonnes"]) / TONNES_PER_MILLION_TONNES,
                    "first_year_above_1_tonne": int(positive.min()) if not positive.empty else np.nan,
                })

                print(f"Completed {run_number:02d}/{total_runs}: {region} | {capacity_scenario} | {lifetime_scenario}")

    results = pd.concat(model_outputs, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return results, summary


# =============================================================================
# PLOTTING
# =============================================================================

# STYLE UPDATE: centralised font-size constants so axis titles, tick
# numbers, and the in-plot scenario label are all easy to tune together.
AXIS_LABEL_FONTSIZE = 15
AXIS_TICK_FONTSIZE = 16
SCENARIO_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 16


def format_million_tonnes_axis(axis: plt.Axes) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.1f}"))
    axis.yaxis.get_offset_text().set_visible(False)
    axis.grid(True, alpha=0.25)
    # STYLE UPDATE: larger y-axis tick numbers for visibility.
    axis.tick_params(axis="y", which="major", labelsize=AXIS_TICK_FONTSIZE)


def apply_full_year_xaxis(axis: plt.Axes, year_min: int, year_max: int, tick_step: int) -> None:
    axis.set_xlim(year_min, year_max)
    axis.xaxis.set_major_locator(MultipleLocator(tick_step))
    axis.xaxis.set_minor_locator(MultipleLocator(1))
    # STYLE UPDATE: larger x-axis tick numbers for visibility.
    axis.tick_params(axis="x", which="major", labelrotation=45, labelsize=AXIS_TICK_FONTSIZE)
    axis.tick_params(axis="x", which="minor", length=2)
    for label in axis.get_xticklabels(which="major"):
        label.set_ha("right")


def _underline_text(text: str) -> str:
    """Insert a Unicode combining low line after every character so the
    string renders underlined in matplotlib without needing a separate
    line artist or LaTeX rendering."""
    return "".join(ch + "\u0332" for ch in text)


def add_top_left_scenario_label(
    axis: plt.Axes,
    capacity_scenario: str,
    region: str,
    panel_letter: str | None = None,
) -> None:
    """STYLE UPDATE: instead of a per-axis title and a figure suptitle, each
    subplot gets a bold, underlined label in its top-left corner reading
    '(a) <Capacity Scenario> scenario - <Region>' (e.g. '(a) Conservative
    scenario - India'). panel_letter is optional; pass None to omit it."""
    prefix = f"({panel_letter}) " if panel_letter else ""
    axis.text(
        0.02, 0.98,
        _underline_text(f"{prefix}{capacity_scenario} scenario - {region}"),
        transform=axis.transAxes,
        fontsize=SCENARIO_LABEL_FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_loss_probability_table(
    output_dir: Path,
    max_retirement_age: int,
    apply_optional_early_failures: bool,
    first_year_failure: float,
    second_year_failure: float,
) -> None:
    distributions = build_loss_distributions(
        max_age=max_retirement_age,
        apply_optional_early_failures=apply_optional_early_failures,
        first_year_failure=first_year_failure,
        second_year_failure=second_year_failure,
    )

    probability_table = pd.DataFrame({name: series for name, series in distributions.items()})
    probability_table.index.name = "module_age_years"
    probability_table.reset_index().to_csv(output_dir / "loss_probability_curves.csv", index=False)


def plot_all_regions_for_each_capacity_scenario(
    results: pd.DataFrame,
    output_dir: Path,
    show_inline: bool,
    year_min: int,
    year_max: int,
    tick_step: int,
) -> None:
    regions = ["India", "Rajasthan", "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu"]

    for capacity_scenario in CAPACITY_SCENARIOS:
        fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharex=True)
        axes = axes.ravel()

        for panel_index, (ax, region) in enumerate(zip(axes, regions)):
            panel_letter = chr(ord("a") + panel_index)
            subset = results.loc[
                results["region"].eq(region)
                & results["capacity_scenario"].eq(capacity_scenario)
            ]

            for lifetime_scenario in LIFETIME_SCENARIOS:
                curve = subset.loc[subset["lifetime_scenario"].eq(lifetime_scenario)]
                curve = curve.loc[curve["year"].between(year_min, year_max)]
                cumulative_million_tonnes = curve["cumulative_waste_tonnes"] / TONNES_PER_MILLION_TONNES
                ax.plot(
                    curve["year"], cumulative_million_tonnes,
                    linewidth=1.6, marker=LINE_MARKER, markersize=LINE_MARKER_SIZE,
                    markevery=1, label=lifetime_scenario.replace("_", " "),
                )

            # STYLE UPDATE: no per-axis title; top-left in-plot label instead,
            # prefixed with a bold (a)-(f) panel letter.
            add_top_left_scenario_label(ax, capacity_scenario, region, panel_letter)
            ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
            ax.set_ylabel(
                "Cumulative PV waste generated (million tonnes)",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight="bold",
            )
            apply_full_year_xaxis(ax, year_min, year_max, tick_step)
            ax.tick_params(axis="x", which="major", labelbottom=True)
            format_million_tonnes_axis(ax)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=3, frameon=False,
            fontsize=LEGEND_FONTSIZE,
            prop={"weight": "bold", "size": LEGEND_FONTSIZE},
            handlelength=3, markerscale=1.8,
        )
        # STYLE UPDATE: figure-level suptitle removed.
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(output_dir / f"all_regions_{capacity_scenario.lower()}_cumulative_waste.png", dpi=300)
        if show_inline:
            try:
                plt.show()
            except Exception:
                pass
        else:
            plt.close(fig)


def plot_each_region(
    results: pd.DataFrame,
    output_dir: Path,
    show_inline: bool,
    year_min: int,
    year_max: int,
    tick_step: int,
) -> None:
    regions = ["India", "Rajasthan", "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu"]

    for region in regions:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharex=True)

        for ax, capacity_scenario in zip(axes, CAPACITY_SCENARIOS):
            subset = results.loc[
                results["region"].eq(region)
                & results["capacity_scenario"].eq(capacity_scenario)
            ]

            for lifetime_scenario in LIFETIME_SCENARIOS:
                curve = subset.loc[subset["lifetime_scenario"].eq(lifetime_scenario)]
                curve = curve.loc[curve["year"].between(year_min, year_max)]
                cumulative_million_tonnes = curve["cumulative_waste_tonnes"] / TONNES_PER_MILLION_TONNES
                ax.plot(
                    curve["year"], cumulative_million_tonnes,
                    linewidth=1.6, marker=LINE_MARKER, markersize=LINE_MARKER_SIZE,
                    markevery=1, label=lifetime_scenario.replace("_", " "),
                )

            # STYLE UPDATE: no per-axis title; top-left in-plot label instead.
            add_top_left_scenario_label(ax, capacity_scenario, region)
            ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
            ax.set_ylabel(
                "Cumulative PV waste generated (million tonnes)",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight="bold",
            )
            apply_full_year_xaxis(ax, year_min, year_max, tick_step)
            ax.tick_params(axis="x", which="major", labelbottom=True)
            format_million_tonnes_axis(ax)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=3, frameon=False,
            fontsize=LEGEND_FONTSIZE,
            prop={"weight": "bold", "size": LEGEND_FONTSIZE},
            handlelength=3, markerscale=1.8,
        )
        # STYLE UPDATE: figure-level suptitle removed.
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        safe_region = region.lower().replace(" ", "_")
        fig.savefig(output_dir / f"{safe_region}_cumulative_waste.png", dpi=300)
        if show_inline:
            try:
                plt.show()
            except Exception:
                pass
        else:
            plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    input_root = (
        KAGGLE_INPUT_ROOT
        if os.path.isdir(KAGGLE_INPUT_ROOT)
        else FALLBACK_INPUT_ROOT
    )

    output_root = (
        KAGGLE_OUTPUT_ROOT
        if os.path.isdir(KAGGLE_OUTPUT_ROOT)
        else FALLBACK_OUTPUT_ROOT
    )

    input_path = build_updated_capacity_input(
        base_input_path=BASE_CAPACITY_INPUT_FILE,
        forecast_additions_path=FORECAST_ADDITIONS_FILE,
        output_path=UPDATED_CAPACITY_INPUT_FILE,
    )

    output_dir = Path(output_root) / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    if MASS_TONNES_PER_MW <= 0:
        raise ValueError("MASS_TONNES_PER_MW must be positive.")

    if MAX_RETIREMENT_AGE <= 0:
        raise ValueError("MAX_RETIREMENT_AGE must be positive.")

    df = pd.read_csv(input_path)
    validate_input(df)

    print("=" * 76)
    print("EoL SOLAR-PV WASTE ACCUMULATION MODEL")
    print("=" * 76)
    print(f"Input file: {input_path.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Mass conversion assumption: {MASS_TONNES_PER_MW:g} tonnes/MW")
    print(f"Opening-stock treatment: {OPENING_STOCK_MODE}")
    print(f"Optional Early-Loss failures enabled: {APPLY_OPTIONAL_EARLY_FAILURES}")
    print(f"Chart x-axis window: {PLOT_YEAR_MIN}-{PLOT_YEAR_MAX} (labels every {PLOT_YEAR_TICK_STEP}y)")

    if OPENING_STOCK_MODE == "as_2014_cohort":
        print(
            "Warning: The unresolved 2014 opening stock is treated as one 2014 "
            "installation cohort. Interpret its retirement timing cautiously."
        )

    results, summary = run_model(
        df=df,
        mass_tonnes_per_mw=MASS_TONNES_PER_MW,
        max_retirement_age=MAX_RETIREMENT_AGE,
        opening_stock_mode=OPENING_STOCK_MODE,
        apply_optional_early_failures=APPLY_OPTIONAL_EARLY_FAILURES,
        first_year_failure=FIRST_YEAR_FAILURE,
        second_year_failure=SECOND_YEAR_FAILURE,
        plot_year_max=PLOT_YEAR_MAX,
    )

    results.to_csv(output_dir / "pv_eol_waste_annual_tonnes_long.csv", index=False)
    summary.to_csv(output_dir / "pv_eol_summary.csv", index=False)

    save_loss_probability_table(
        output_dir=output_dir,
        max_retirement_age=MAX_RETIREMENT_AGE,
        apply_optional_early_failures=APPLY_OPTIONAL_EARLY_FAILURES,
        first_year_failure=FIRST_YEAR_FAILURE,
        second_year_failure=SECOND_YEAR_FAILURE,
    )
    plot_all_regions_for_each_capacity_scenario(
        results, output_dir, SHOW_PLOTS_INLINE, PLOT_YEAR_MIN, PLOT_YEAR_MAX, PLOT_YEAR_TICK_STEP
    )
    plot_each_region(
        results, output_dir, SHOW_PLOTS_INLINE, PLOT_YEAR_MIN, PLOT_YEAR_MAX, PLOT_YEAR_TICK_STEP
    )

    print("\nModel completed successfully.")
    print(f"Annual results: {output_dir / 'pv_eol_waste_annual_tonnes_long.csv'}")
    print(f"Summary:        {output_dir / 'pv_eol_summary.csv'}")
    print(f"Graphs:         {output_dir}")

    return results, summary


if __name__ == "__main__":
    main()


