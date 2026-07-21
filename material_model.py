# %% [markdown]
# weight distribution of materials

# %%
#!/usr/bin/env python3
"""
PV material inventory analysis from pre-calculated PV end-of-life waste scenarios.

Input CSV expected columns:
    region, capacity_scenario, lifetime_scenario, year,
    annual_waste_tonnes, cumulative_waste_tonnes

Outputs (written under --output):
    tables/
        iea_csi_composition_used.csv         - material composition reference
        pv_material_waste_all_years_long.csv - every region/scenario/year/material
        pv_material_waste_selected_years_long.csv
        pv_material_waste_selected_years_wide.csv
        chart_manifest.csv                   - list of every chart produced
    charts/
        composition/csi_weight_composition_pie_of_pie.png
        material_bars/<metric>/<region>_<year>_<metric>_material_bars.png
        region_comparison/<year>_<metric>_region_comparison.png
    README_outputs.txt

Notes on methodology:
- The IEA reports composition RANGES, not one exact figure summing to 100%.
  This script uses the range midpoints and normalizes them to 100%.
- Results are "contained material inventory" in PV waste, not recovered
  material - no collection or recycling-recovery efficiency is applied.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
    pip install matplotlib numpy
    python pv_material_waste_analysis.py --input your_data.csv --output results

Input resolution order (first one found wins), so this same file works both
from the VS Code "Run" button and from the command line on another machine:
    1. --input PATH, if supplied on the command line.
    2. NOTEBOOK_INPUT_CSV below, IF that path exists on this machine.
    3. Auto-detect: most recently modified *.csv in the working directory
       whose name contains "pv_eol", "pv_waste", or "pv_material".
If none of the three resolve to a real file, the script exits with a clear
error instead of a raw traceback.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.ticker import FuncFormatter

# =============================================================================
# User-editable settings
# =============================================================================
TARGET_YEARS = (2045, 2060, 2075)
CAPACITY_SCENARIO_ORDER = ("Conservative", "Medium", "High")
LIFETIME_SCENARIO_ORDER = ("Fixed_25", "Early_Loss", "Regular_Loss")

# Convenience default for running from VS Code's "Run" button / an
# interactive window, where passing --input is inconvenient. This is used
# ONLY if it exists on the current machine (see resolve_input_csv() below) -
# on any other machine the script transparently falls back to --input or
# auto-detection instead of crashing on a path that doesn't exist there.
NOTEBOOK_INPUT_CSV = (
    r"C:\Users\thaku\trendy\Data-Prediction-Model\pv_eol_model_outputs_20260719_014234_b8899729"
    r"\pv_eol_waste_annual_tonnes_longo.csv"
)

# Filename fragments used for auto-detecting the input CSV when neither
# --input nor NOTEBOOK_INPUT_CSV resolve to a real file. Matched
# case-insensitively against files in the CWD.
AUTODETECT_NAME_HINTS = ("pv_eol", "pv_waste", "pv_material")

# Reference scenario used for the cross-region comparison chart. Chosen as a
# representative "central" combination rather than the most extreme case.
COMPARISON_CAPACITY_SCENARIO = "Medium"
COMPARISON_LIFETIME_SCENARIO = "Regular_Loss"

# Midpoints selected from IEA, "Solar PV Global Supply Chains" (2022),
# c-Si weight-based composition figure.
IEA_SOURCE_RANGES = {
    "Glass": "68-72%",
    "Aluminium": "12-14%",
    "Polymers": "8-10%",
    "Silicon": "3-4%",
    "Copper": "2-4%",
    "Silver": "0.03-0.08%",
    "Zinc": "0.03-0.10%",
    "Lead": "0.01-0.05%",
    "Tin": "0.01-0.05%",
    "Other": "<0.20%",
}

# For bounded ranges, the arithmetic midpoint is used.
# For "Other <0.2%", 0.2% is used as a conservative representative value.
IEA_RAW_MIDPOINT_PERCENT = {
    "Glass": 70.0,
    "Aluminium": 13.0,
    "Polymers": 9.0,
    "Silicon": 3.5,
    "Copper": 3.0,
    "Silver": 0.055,
    "Zinc": 0.065,
    "Lead": 0.030,
    "Tin": 0.030,
    "Other": 0.200,
}

MAJOR_MATERIALS = ("Glass", "Aluminium", "Polymers", "Silicon", "Copper")
MINOR_MATERIALS = ("Silver", "Zinc", "Lead", "Tin", "Other")
ALL_MATERIALS = MAJOR_MATERIALS + MINOR_MATERIALS

IEA_SOURCE_URL = (
    "https://iea.blob.core.windows.net/assets/2d18437f-211d-4504-beeb-"
    "570c4d139e25/SpecialReportonSolarPVGlobalSupplyChains.pdf"
)

REQUIRED_CSV_COLUMNS = {
    "region",
    "capacity_scenario",
    "lifetime_scenario",
    "year",
    "annual_waste_tonnes",
    "cumulative_waste_tonnes",
}

# Consistent, colorblind-friendlier palette used across every chart.
CAPACITY_COLORS = {
    "Conservative": "#3F6DA8",
    "Medium": "#E58A2A",
    "High": "#B5392C",
}
MAJOR_PIE_COLORS = ["#3F6DA8", "#5FA8D3", "#5CB88A", "#E5B93B", "#E58A2A", "#9A9A9A"]
MINOR_PIE_COLORS = ["#B5392C", "#D4A017", "#7A5CA8", "#6E6E6E", "#2B2B2B"]
REGION_PALETTE = [
    "#3F6DA8", "#E58A2A", "#5CB88A", "#B5392C", "#7A5CA8",
    "#5FA8D3", "#E5B93B", "#6E6E6E", "#8C564B", "#17A2B8",
]


# =============================================================================
# Styling
# =============================================================================
def apply_professional_style() -> None:
    """Central place for all chart styling so every figure looks consistent."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.6,
            "axes.axisbelow": True,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.frameon": False,
            "legend.fontsize": 14,
        }
    )


def strip_top_right_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Utilities
# =============================================================================
def normalize_percentages(raw_percent: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(raw_percent.values()))
    if total <= 0:
        raise ValueError("Composition percentages must sum to a positive value.")
    return {material: value * 100.0 / total for material, value in raw_percent.items()}


def safe_filename(text: str) -> str:
    valid = []
    for char in text.strip():
        if char.isalnum() or char in ("-", "_"):
            valid.append(char)
        elif char.isspace():
            valid.append("_")
    return "".join(valid).strip("_") or "output"


def compact_number(value: float, _position: int = 0) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if abs_value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.1f}k"
    if abs_value >= 1:
        return f"{value:.0f}"
    if abs_value == 0:
        return "0"
    return f"{value:.3g}"


def millions_label(value: float) -> str:
    """Data-label formatter that always expresses the value in millions of
    tonnes, rounded to one decimal place, regardless of magnitude (used only
    for the numbers printed above bars, not the axis-tick formatting)."""
    return f"{round(float(value) / 1e6, 1):.1f}M"


def label_lifetime(name: str) -> str:
    return {
        "Fixed_25": "Fixed lifetime (25 y)",
        "Early_Loss": "Early-loss Weibull",
        "Regular_Loss": "Regular-loss Weibull",
    }.get(name, name.replace("_", " "))


def metric_title(metric: str) -> str:
    if metric == "annual":
        return "Annual material generation"
    if metric == "cumulative":
        return "Cumulative material inventory"
    raise ValueError(f"Unsupported metric: {metric}")


def material_value_key(metric: str) -> str:
    """Single source of truth for annual/cumulative field-name lookups."""
    if metric == "annual":
        return "annual_material_tonnes"
    if metric == "cumulative":
        return "cumulative_material_tonnes"
    raise ValueError(f"Unsupported metric: {metric}")


# =============================================================================
# Data loading and validation
# =============================================================================
def read_waste_csv(input_csv: Path) -> List[dict]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_CSV_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(sorted(missing_columns))
            )

        rows: List[dict] = []
        for line_number, row in enumerate(reader, start=2):
            # DictReader fills missing trailing fields with None (short rows).
            # Also treat blank cells ("") as missing rather than letting them
            # fail later with a confusing float() ValueError.
            missing_values = [
                col for col in REQUIRED_CSV_COLUMNS
                if row.get(col) is None or str(row.get(col)).strip() == ""
            ]
            if missing_values:
                raise ValueError(
                    f"CSV line {line_number} is missing a value for: "
                    + ", ".join(sorted(missing_values))
                )

            try:
                rows.append(
                    {
                        "region": row["region"].strip(),
                        "capacity_scenario": row["capacity_scenario"].strip(),
                        "lifetime_scenario": row["lifetime_scenario"].strip(),
                        "year": int(float(row["year"])),
                        "annual_waste_tonnes": float(row["annual_waste_tonnes"]),
                        "cumulative_waste_tonnes": float(row["cumulative_waste_tonnes"]),
                    }
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric data at CSV line {line_number}: {exc}") from exc

    if not rows:
        raise ValueError("Input CSV contains no data rows.")
    return rows


def validate_target_combinations(rows: Sequence[dict], target_years: Sequence[int]) -> None:
    available = {
        (row["region"], row["capacity_scenario"], row["lifetime_scenario"], row["year"])
        for row in rows
    }
    regions = sorted({row["region"] for row in rows})
    missing: List[Tuple[str, str, str, int]] = []
    for region in regions:
        for year in target_years:
            for cap in CAPACITY_SCENARIO_ORDER:
                for life in LIFETIME_SCENARIO_ORDER:
                    key = (region, cap, life, year)
                    if key not in available:
                        missing.append(key)
    if missing:
        preview = "; ".join(map(str, missing[:8]))
        raise ValueError(
            f"Missing {len(missing)} requested region/scenario/year combinations. "
            f"Every region must have data for all of {CAPACITY_SCENARIO_ORDER} x "
            f"{LIFETIME_SCENARIO_ORDER} for each target year {tuple(target_years)}. "
            f"First missing entries (region, capacity, lifetime, year): {preview}"
        )


def autodetect_input_csv(search_dir: Path) -> Path | None:
    """Best-effort auto-detect. Returns None (never raises) so callers can
    fold this into a resolution chain and produce one combined error message
    instead of a premature failure."""
    candidates = [
        path for path in search_dir.glob("*.csv")
        if any(hint in path.name.lower() for hint in AUTODETECT_NAME_HINTS)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def resolve_input_csv(cli_input: str | None, search_dir: Path) -> Path:
    """Three-tier input resolution, in priority order:
    1. --input, if the user passed one explicitly.
    2. NOTEBOOK_INPUT_CSV, but ONLY if that exact path exists on this
       machine (so the script never crashes on a personal path when run
       elsewhere - it just silently moves on to the next option).
    3. Auto-detected *.csv in the working directory.
    Raises FileNotFoundError with a message listing every path that was
    tried if none of the three resolve.
    """
    attempted: List[str] = []

    if cli_input:
        path = Path(cli_input).expanduser().resolve()
        attempted.append(f"--input: {path}")
        if path.exists():
            return path

    if NOTEBOOK_INPUT_CSV:
        notebook_path = Path(NOTEBOOK_INPUT_CSV)
        attempted.append(f"NOTEBOOK_INPUT_CSV: {notebook_path}")
        if notebook_path.exists():
            print(f"Using NOTEBOOK_INPUT_CSV: {notebook_path}")
            return notebook_path.resolve()

    autodetected = autodetect_input_csv(search_dir)
    if autodetected is not None:
        print(f"Auto-detected input CSV: {autodetected}")
        return autodetected
    attempted.append(
        f"auto-detect: no *.csv in '{search_dir}' matched {AUTODETECT_NAME_HINTS}"
    )

    raise FileNotFoundError(
        "Could not resolve an input CSV. Tried, in order:\n  "
        + "\n  ".join(attempted)
        + "\nPass --input path/to/file.csv explicitly to fix this."
    )


# =============================================================================
# Material calculations
# =============================================================================
def build_material_rows(
    waste_rows: Sequence[dict],
    normalized_percent: Mapping[str, float],
) -> List[dict]:
    result: List[dict] = []
    for row in waste_rows:
        for material in ALL_MATERIALS:
            fraction = normalized_percent[material] / 100.0
            result.append(
                {
                    "region": row["region"],
                    "capacity_scenario": row["capacity_scenario"],
                    "lifetime_scenario": row["lifetime_scenario"],
                    "year": row["year"],
                    "material": material,
                    "source_range": IEA_SOURCE_RANGES[material],
                    "raw_midpoint_weight_percent": IEA_RAW_MIDPOINT_PERCENT[material],
                    "normalized_weight_percent": normalized_percent[material],
                    "annual_material_tonnes": row["annual_waste_tonnes"] * fraction,
                    "cumulative_material_tonnes": row["cumulative_waste_tonnes"] * fraction,
                }
            )
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_composition_csv(path: Path, normalized_percent: Mapping[str, float]) -> None:
    rows = []
    for material in ALL_MATERIALS:
        rows.append(
            {
                "material": material,
                "iea_reported_range": IEA_SOURCE_RANGES[material],
                "selected_raw_midpoint_percent": IEA_RAW_MIDPOINT_PERCENT[material],
                "normalized_weight_percent_used": normalized_percent[material],
                "normalized_mass_fraction_used": normalized_percent[material] / 100.0,
                "source_url": IEA_SOURCE_URL,
            }
        )
    write_csv(
        path,
        rows,
        (
            "material",
            "iea_reported_range",
            "selected_raw_midpoint_percent",
            "normalized_weight_percent_used",
            "normalized_mass_fraction_used",
            "source_url",
        ),
    )


def index_material_rows(
    material_rows: Sequence[dict], metric: str
) -> Dict[Tuple[str, int, str, str, str], float]:
    value_key = material_value_key(metric)
    index: Dict[Tuple[str, int, str, str, str], float] = {}
    for row in material_rows:
        key = (
            row["region"],
            int(row["year"]),
            row["capacity_scenario"],
            row["lifetime_scenario"],
            row["material"],
        )
        index[key] = float(row[value_key])
    return index


# =============================================================================
# Chart 1: pie-of-pie composition reference
# =============================================================================
def create_composition_pie(normalized_percent: Mapping[str, float], output_dir: Path) -> Path:
    major_values = [normalized_percent[m] for m in MAJOR_MATERIALS]
    minor_total = sum(normalized_percent[m] for m in MINOR_MATERIALS)
    main_values = major_values + [minor_total]
    minor_values = [normalized_percent[m] for m in MINOR_MATERIALS]

    fig = plt.figure(figsize=(13.5, 7.5))
    ax_main = fig.add_axes([0.05, 0.14, 0.56, 0.73])
    ax_minor = fig.add_axes([0.66, 0.30, 0.27, 0.42])

    def main_autopct(percent: float) -> str:
        return f"{percent:.2f}%" if percent >= 5.0 else ""

    wedges_main, _, autotexts_main = ax_main.pie(
        main_values,
        autopct=main_autopct,
        startangle=270,
        counterclock=False,
        colors=MAJOR_PIE_COLORS,
        wedgeprops={"edgecolor": "white", "linewidth": 1.1},
        textprops={"fontsize": 11, "color": "white"},
        pctdistance=0.72,
    )
    for text in autotexts_main:
        text.set_fontweight("bold")

    wedges_minor, _, _ = ax_minor.pie(
        minor_values,
        autopct=lambda p: f"{p * minor_total / 100.0:.3f}%",
        startangle=90,
        counterclock=False,
        colors=MINOR_PIE_COLORS,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        textprops={"fontsize": 9, "color": "white"},
        pctdistance=0.70,
    )

    # NOTE: top titles intentionally removed for both the main pie and the
    # expanded minor-material pie (previously ax_main.set_title(...) and
    # ax_minor.set_title(...)). Panel labels (a)/(b) below still identify
    # each chart.

    minor_wedge = wedges_main[-1]
    theta1, theta2 = minor_wedge.theta1, minor_wedge.theta2
    for theta_deg, y_minor in ((theta1, 0.72), (theta2, -0.72)):
        theta_rad = math.radians(theta_deg)
        connection = ConnectionPatch(
            xyA=(math.cos(theta_rad), math.sin(theta_rad)),
            coordsA=ax_main.transData,
            xyB=(-0.96, y_minor),
            coordsB=ax_minor.transData,
            color="#8A8A8A",
            linewidth=1.1,
        )
        fig.add_artist(connection)

    # Sub-figure numbering: (a) in front of (to the left of) the main pie,
    # (b) in front of (to the left of) the expanded minor-material pie -
    # vertically centered on each chart rather than sitting above it.
    ax_main.text(
        -0.08, 0.5, "(a)", transform=ax_main.transAxes,
        fontsize=18, fontweight="bold", ha="right", va="center",
    )
    ax_minor.text(
        -0.14, 0.5, "(b)", transform=ax_minor.transAxes,
        fontsize=18, fontweight="bold", ha="right", va="center",
    )

    # --- Custom legends (brackets with the IEA source ranges removed, and
    # no explanatory line underneath) ------------------------------------
    major_handles = [
        Patch(facecolor=MAJOR_PIE_COLORS[i], edgecolor="white",
              label=f"{material}: {normalized_percent[material]:.3f}% used")
        for i, material in enumerate(MAJOR_MATERIALS)
    ]
    major_handles.append(
        Patch(facecolor=MAJOR_PIE_COLORS[-1], edgecolor="white",
              label=f"Minor materials combined: {minor_total:.3f}% used")
    )
    fig.legend(
        handles=major_handles,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.01),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    minor_handles = [
        Patch(facecolor=MINOR_PIE_COLORS[i], edgecolor="white",
              label=f"{material}: {normalized_percent[material]:.3f}% used")
        for i, material in enumerate(MINOR_MATERIALS)
    ]
    fig.legend(
        handles=minor_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.01),
        ncol=1,
        frameon=False,
        fontsize=10,
    )
    # Matplotlib only keeps the most recent legend attached to a figure by
    # default when using fig.legend() twice, so re-add the first one as an
    # artist to keep both on screen.
    fig.add_artist(fig.legends[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "csi_weight_composition_pie_of_pie.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


# =============================================================================
# Chart 2: per-region/year grouped bar chart (major + minor materials)
# =============================================================================
def plot_grouped_material_bars(
    region: str,
    year: int,
    metric: str,
    material_index: Mapping[Tuple[str, int, str, str, str], float],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(
        2, 3, figsize=(20, 11.5), sharex="row",
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )

    bar_width = 0.24
    scenario_offsets = np.array([-bar_width, 0.0, bar_width])

    for column, lifetime in enumerate(LIFETIME_SCENARIO_ORDER):
        for row_index, materials in enumerate((MAJOR_MATERIALS, MINOR_MATERIALS)):
            ax = axes[row_index, column]
            x = np.arange(len(materials), dtype=float)

            for scenario_index, capacity in enumerate(CAPACITY_SCENARIO_ORDER):
                values = [
                    material_index.get((region, year, capacity, lifetime, material), 0.0)
                    for material in materials
                ]
                bars = ax.bar(
                    x + scenario_offsets[scenario_index], values, width=bar_width,
                    label=capacity, color=CAPACITY_COLORS[capacity],
                    edgecolor="white", linewidth=0.6, zorder=3,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(materials, rotation=25, ha="right", fontsize=16)
            ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
            ax.tick_params(axis="y", labelsize=16)
            ax.set_ylabel(f"Tonnes ({region})" if column == 0 else "", fontsize=22, fontweight="bold")
            strip_top_right_spines(ax)
            ax.grid(axis="y", zorder=0)

            if row_index == 0:
                ax.set_title(label_lifetime(lifetime), fontsize=20, fontweight="bold", pad=10)
                ax.margins(y=0.15)
            else:
                # Minor constituents differ strongly in magnitude; symlog
                # keeps zeros visible while still showing silver/lead/tin
                # beside the much larger "Other" fraction.
                ax.set_yscale("symlog", linthresh=1.0)

            panel_label = "Major materials" if row_index == 0 else "Minor materials (symlog scale)"
            ax.text(
                0.5, 0.96, panel_label, transform=ax.transAxes, va="top", ha="center",
                fontsize=14, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.85, "pad": 2.5},
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, fontsize=16, title="Installation capacity scenario", title_fontsize=16,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_fontfamily("Times New Roman")

    # NOTE: top figure heading (fig.suptitle), the bottom explanatory
    # caption line, and the (a)-(f) panel numbering have all been removed.
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.14, wspace=0.20, hspace=0.36)

    base = f"{safe_filename(region)}_{year}_{metric}_material_bars"
    png_path = output_dir / f"{base}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return png_path


# =============================================================================
# Chart 3: cross-region comparison (stacked bars, one reference scenario)
# =============================================================================
def plot_region_comparison(
    year: int,
    metric: str,
    regions: Sequence[str],
    material_index: Mapping[Tuple[str, int, str, str, str], float],
    output_dir: Path,
) -> Path:
    """Single summary chart: total material stack per region for one year,
    under one reference scenario. Complements the per-region detail charts
    with an at-a-glance cross-region comparison."""
    cap, life = COMPARISON_CAPACITY_SCENARIO, COMPARISON_LIFETIME_SCENARIO

    totals = {
        region: sum(
            material_index.get((region, year, cap, life, m), 0.0) for m in ALL_MATERIALS
        )
        for region in regions
    }
    ordered_regions = sorted(regions, key=lambda r: totals[r], reverse=True)

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(ordered_regions) + 3), 7))
    x = np.arange(len(ordered_regions), dtype=float)
    bottoms = np.zeros(len(ordered_regions))
    material_colors = dict(zip(ALL_MATERIALS, MAJOR_PIE_COLORS[:5] + MINOR_PIE_COLORS))

    for material in ALL_MATERIALS:
        values = np.array(
            [material_index.get((r, year, cap, life, material), 0.0) for r in ordered_regions]
        )
        ax.bar(
            x, values, bottom=bottoms, width=0.62, label=material,
            color=material_colors[material], edgecolor="white", linewidth=0.5, zorder=3,
        )
        bottoms += values

    for xi, total in zip(x, (totals[r] for r in ordered_regions)):
        ax.annotate(
            compact_number(total), xy=(xi, total), xytext=(0, 4),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ordered_regions, rotation=20, ha="right", fontsize=16, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Tonnes", fontsize=16, fontweight="bold")
    strip_top_right_spines(ax)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=16)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{year}_{metric}_region_comparison.png"
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return png_path


# =============================================================================
# Main workflow
# =============================================================================
def run_analysis(
    input_csv: Path,
    output_dir: Path,
    target_years: Sequence[int],
    metrics: Sequence[str],
) -> None:
    apply_professional_style()

    rows = read_waste_csv(input_csv)
    validate_target_combinations(rows, target_years)

    normalized_percent = normalize_percentages(IEA_RAW_MIDPOINT_PERCENT)
    material_rows = build_material_rows(rows, normalized_percent)

    tables_dir = output_dir / "tables"
    charts_dir = output_dir / "charts"
    pie_dir = charts_dir / "composition"
    bars_dir = charts_dir / "material_bars"
    comparison_dir = charts_dir / "region_comparison"

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_composition_csv(tables_dir / "iea_csi_composition_used.csv", normalized_percent)

    all_fields = (
        "region", "capacity_scenario", "lifetime_scenario", "year", "material",
        "source_range", "raw_midpoint_weight_percent", "normalized_weight_percent",
        "annual_material_tonnes", "cumulative_material_tonnes",
    )
    write_csv(tables_dir / "pv_material_waste_all_years_long.csv", material_rows, all_fields)

    target_set = set(target_years)
    selected_rows = [row for row in material_rows if int(row["year"]) in target_set]
    write_csv(tables_dir / "pv_material_waste_selected_years_long.csv", selected_rows, all_fields)

    wide_records: Dict[Tuple[str, str, str, int, str], dict] = {}
    for metric in metrics:
        value_key = material_value_key(metric)
        for row in selected_rows:
            key = (row["region"], row["capacity_scenario"], row["lifetime_scenario"], int(row["year"]), metric)
            record = wide_records.setdefault(
                key,
                {
                    "region": key[0], "capacity_scenario": key[1], "lifetime_scenario": key[2],
                    "year": key[3], "metric": key[4],
                },
            )
            record[f"{row['material']}_tonnes"] = row[value_key]

    wide_fields = ("region", "capacity_scenario", "lifetime_scenario", "year", "metric") + tuple(
        f"{material}_tonnes" for material in ALL_MATERIALS
    )
    ordered_wide = sorted(
        wide_records.values(),
        key=lambda r: (
            r["region"], int(r["year"]),
            LIFETIME_SCENARIO_ORDER.index(r["lifetime_scenario"]),
            CAPACITY_SCENARIO_ORDER.index(r["capacity_scenario"]), r["metric"],
        ),
    )
    write_csv(tables_dir / "pv_material_waste_selected_years_wide.csv", ordered_wide, wide_fields)

    create_composition_pie(normalized_percent, pie_dir)

    regions = sorted({row["region"] for row in rows})
    generated_charts: List[dict] = []
    for metric in metrics:
        material_index = index_material_rows(material_rows, metric)
        metric_output = bars_dir / metric
        for region in regions:
            for year in target_years:
                png_path = plot_grouped_material_bars(region, int(year), metric, material_index, metric_output)
                generated_charts.append(
                    {"chart_type": "region_detail", "region": region, "year": int(year),
                     "metric": metric, "png_file": str(png_path.relative_to(output_dir))}
                )

        for year in target_years:
            png_path = plot_region_comparison(int(year), metric, regions, material_index, comparison_dir)
            generated_charts.append(
                {"chart_type": "region_comparison", "region": "ALL", "year": int(year),
                 "metric": metric, "png_file": str(png_path.relative_to(output_dir))}
            )

    write_csv(
        tables_dir / "chart_manifest.csv", generated_charts,
        ("chart_type", "region", "year", "metric", "png_file"),
    )

    notes = output_dir / "README_outputs.txt"
    notes.write_text(
        "\n".join(
            [
                "PV MATERIAL WASTE ANALYSIS",
                "==========================",
                f"Input: {input_csv}",
                f"Target years: {', '.join(map(str, target_years))}",
                f"Metrics plotted: {', '.join(metrics)}",
                f"Regions detected: {', '.join(regions)}",
                "",
                "Composition method:",
                "- IEA c-Si weight-based percentage ranges were converted to midpoint values.",
                "- Midpoints were normalized to 100% because the source ranges are approximate",
                "  and do not sum exactly to 100%.",
                "- 'Other <0.2%' was represented as 0.2% before normalization.",
                "",
                "Chart types produced:",
                "- composition/  : IEA reference pie-of-pie (material % basis, not tonnes).",
                "- material_bars/: one 2x3 grouped-bar figure per region/year/metric.",
                "- region_comparison/: one stacked bar per year comparing regions under a",
                f"  single reference scenario ({COMPARISON_CAPACITY_SCENARIO} capacity x "
                f"{COMPARISON_LIFETIME_SCENARIO}).",
                "",
                "Interpretation:",
                "- Outputs are contained material tonnes in the modeled PV waste.",
                "- They are NOT recovered tonnes; collection/recycling efficiency is not applied.",
                "",
                f"Source: {IEA_SOURCE_URL}",
            ]
        ),
        encoding="utf-8",
    )

    print("Analysis completed successfully.")
    print(f"Output directory: {output_dir}")
    print(f"Regions: {', '.join(regions)}")
    print(f"Target years: {', '.join(map(str, target_years))}")
    print(f"Charts generated: {len(generated_charts)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PV waste scenarios into c-Si material inventories and charts.",
        allow_abbrev=False,
    )
    parser.add_argument("--input", default=None, help="Path to the structured PV waste CSV.")
    parser.add_argument("--output", default="pv_material_waste_outputs", help="Output directory.")
    parser.add_argument(
        "--years", nargs="+", type=int, default=list(TARGET_YEARS),
        help="Target years for bar charts. Default: 2045 2060 2075",
    )
    parser.add_argument(
        "--metric", choices=("annual", "cumulative", "both"), default="cumulative",
        help="Waste metric to plot. Default: cumulative",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring unrecognized arguments:", " ".join(unknown), file=sys.stderr)

    metrics = ("annual", "cumulative") if args.metric == "both" else (args.metric,)
    input_csv = resolve_input_csv(args.input, Path.cwd())

    run_analysis(
        input_csv=input_csv,
        output_dir=Path(args.output).expanduser().resolve(),
        target_years=tuple(args.years),
        metrics=metrics,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


