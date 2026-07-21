# %% [markdown]
# Logistic s curve

# %%
# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

try:
    from IPython.display import display
except ImportError:
    display = print

plt.rcParams["font.family"] = "Times New Roman"
# ============================================================
# USER SETTINGS
# ============================================================

# CHANGED FOR VS CODE: Kaggle notebooks mount input files under a fixed
# /kaggle/input/ path that this script used to os.walk() looking for
# TARGET_FILES. That path doesn't exist outside Kaggle. On a local machine
# / VS Code, point this at wherever your CSVs actually live. Default below
# assumes the CSVs sit in the same folder as this script - change if not.
DATA_DIR = BASE_DIR / "data"

TARGET_STATE = "All"

TARGET_FILES = [
    "gujarat_pv_model_input_constraints.csv",
    "india_pv_model_input_constraints.csv",
    "maharashtra_pv_model_input_constraints.csv",
    "tamilnadu_pv_model_input_constraints.csv",
]

# FIX #1: the state/region for every file is now derived from the filename
# itself, once, here - instead of trusting each CSV to carry its own correct
# 'state'/'country' column. This is what actually caused the bug: any file
# missing that column (confirmed with Karnataka in testing) silently got
# folded into "India" in some pandas versions, or crashed with
# "'float' object has no attribute 'upper'" in others, because the old
# fallback (df["state"].replace({'nan': 'India', ...})) only reliably catches
# a literal string 'nan', which isn't guaranteed after concatenating files
# where the column doesn't exist at all in one of them.
FILENAME_TO_STATE = {
    "gujarat_pv_model_input_constraints.csv": "Gujarat",
    "india_pv_model_input_constraints.csv": "India",
    "maharashtra_pv_model_input_constraints.csv": "Maharashtra",
    "tamilnadu_pv_model_input_constraints.csv": "Tamil Nadu",
}

HISTORICAL_START_YEAR = 2014
HISTORICAL_END_YEAR = 2026
FORECAST_START_YEAR = 2027
FORECAST_END_YEAR = 2050

# Conversion used only for graph and forecast-table outputs.
MW_PER_GW = 1000.0

SCENARIO_ORDER = ["Conservative", "Medium", "High"]

# ============================================================
# 1. CSV VALIDATION
# ============================================================

def validate_pv_csv(df, expected_scenarios=SCENARIO_ORDER):
    df = df.copy()
    df.columns = df.columns.str.strip()

    required_cols = ["state", "year", "scenario", "carrying_capacity_mw", "total_installed_capacity_mw"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["state"] = df["state"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["carrying_capacity_mw"] = pd.to_numeric(df["carrying_capacity_mw"], errors="raise")
    df["total_installed_capacity_mw"] = pd.to_numeric(df["total_installed_capacity_mw"], errors="raise")

    # FIX #2: the state column is now always set explicitly from the filename
    # map before this function ever runs (see ingestion below), so it should
    # never legitimately be missing/NaN here. If it somehow still is, that's
    # a real data problem worth stopping on rather than silently guessing
    # "India" - the old behavior is exactly what caused another region's data
    # to get merged into India's numbers without any warning.
    bad_state_mask = df["state"].isin(["nan", "None", "", "<NA>"]) | df["state"].isna()
    if bad_state_mask.any():
        bad_rows = df.loc[bad_state_mask, ["year", "scenario"]].to_dict("records")
        raise ValueError(
            f"{bad_state_mask.sum()} row(s) have no identifiable state/region after ingestion "
            f"(examples: {bad_rows[:5]}). Refusing to guess - check the source file(s)."
        )

    errors, warnings, rows = [], [], []

    for (state, scenario), g in df.groupby(["state", "scenario"], sort=False, dropna=False):
        g = g.sort_values("year")

        # Real data-quality checks now actually populate warnings
        # (previously this list was declared but never written to).
        dup_years = g["year"][g["year"].duplicated()].unique()
        if len(dup_years) > 0:
            warnings.append(f"{state}/{scenario}: duplicate year(s) {list(dup_years)} - check for merged/duplicated rows.")

        if not g["total_installed_capacity_mw"].is_monotonic_increasing:
            warnings.append(f"{state}/{scenario}: total_installed_capacity_mw is not monotonically increasing - unexpected for cumulative capacity.")

        L = float(g["carrying_capacity_mw"].iloc[0])
        if g["total_installed_capacity_mw"].max() > L:
            warnings.append(f"{state}/{scenario}: installed capacity ({g['total_installed_capacity_mw'].max():.0f} MW) already exceeds the stated ceiling L ({L:.0f} MW) - the logistic fit for this scenario will be unreliable.")

        rows.append({
            "state": state,
            "scenario": scenario,
            "rows": len(g),
            "first_year": int(g["year"].min()),
            "last_year": int(g["year"].max()),
            "carrying_capacity_mw": L,
            "max_installed_capacity_mw": float(g["total_installed_capacity_mw"].max()),
        })

    summary = pd.DataFrame(rows)
    if errors:
        raise ValueError("CSV failed validation:\n" + "\n".join(errors))
    return df, summary, warnings

# ============================================================
# 2. MATHEMATICAL MODELS
# ============================================================

def logistic_model(t, L, k, t0):
    t = np.asarray(t, dtype=float)
    z = np.clip(-k * (t - t0), -700, 700)
    return L / (1.0 + np.exp(z))

def logistic_residuals(params, t, y_true, L):
    k, t0 = params
    return logistic_model(t, L, k, t0) - y_true

def fit_logistic_params(t, y, L):
    fraction = np.clip(y / L, 1e-8, 1 - 1e-8)
    logit_y = np.log(fraction / (1 - fraction))
    slope, intercept = np.polyfit(t, logit_y, 1)
    x0 = [float(np.clip(slope, 1e-4, 1.0)), float(np.clip(-intercept / slope, -50.0, 80.0))]
    return least_squares(logistic_residuals, x0=x0, args=(t, y, L), bounds=([1e-6, -100.0], [5.0, 100.0]))

def exp_model(t, a, r):
    return a * np.exp(r * t)

def exp_residuals(params, t, y_true):
    a, r = params
    return exp_model(t, a, r) - y_true

def score_model(y_true, y_pred):
    # FIX #3: guard against an empty test set. Previously, if SPLIT_YEAR was
    # at or past the last year in the data, y_test/y_pred_test were empty
    # arrays and sklearn's mean_squared_error raised
    # "ValueError: Found array with 0 sample(s)" - crashing the whole run.
    if len(y_true) == 0:
        return np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    return rmse, r2

# ============================================================
# 3. FULL-DATA MODEL FITTING AND EVALUATION
# ============================================================

def evaluate_full_data_by_scenario(df, scenario_order=SCENARIO_ORDER):
    """
    Fit the logistic model using every historical observation from
    2014 through 2026. No train/test split is applied.
    """
    results = []
    fitted_models = {}

    scenarios = [s for s in scenario_order if s in df["scenario"].unique()]

    for scenario in scenarios:
        g = df[df["scenario"] == scenario].sort_values("year").copy()

        L = float(g["carrying_capacity_mw"].iloc[0])
        years = g["year"].to_numpy(dtype=int)
        y = g["total_installed_capacity_mw"].to_numpy(dtype=float)

        historical_mask = (
            (years >= HISTORICAL_START_YEAR) &
            (years <= HISTORICAL_END_YEAR)
        )

        years_fit = years[historical_mask]
        y_fit = y[historical_mask]

        if len(y_fit) < 3:
            print(
                f"  [skip] {scenario}: fewer than 3 observations between "
                f"{HISTORICAL_START_YEAR} and {HISTORICAL_END_YEAR}."
            )
            continue

        base_year = int(years_fit.min())
        t_fit = years_fit - base_year

        fit = fit_logistic_params(t_fit, y_fit, L)
        k_opt, t0_opt = fit.x

        y_pred_fit = logistic_model(t_fit, L, k_opt, t0_opt)
        rmse_fit, r2_fit = score_model(y_fit, y_pred_fit)

        results.append({
            "scenario": scenario,
            "L_mw": L,
            "k_full": k_opt,
            "t0_year": t0_opt + base_year,
            "fit_rmse": rmse_fit,
            "fit_r2": r2_fit,
        })

        fitted_models[scenario] = {
            "L": L,
            "k": k_opt,
            "t0_relative": t0_opt,
            "base_year": base_year,
            "historical_start_year": int(years_fit.min()),
            "historical_end_year": int(years_fit.max()),
        }

    return pd.DataFrame(results), fitted_models

# ============================================================
# 4. UNIFIED PLOT FUNCTION
# ============================================================

def plot_all_scenarios_combined(df, fitted_models, state_name, forecast_end_year=FORECAST_END_YEAR):
    if not fitted_models:
        print(f"  [skip plot] {state_name}: no scenario could be fit.")
        return

    first_scen = list(fitted_models.keys())[0]
    g = df[df["scenario"] == first_scen].sort_values("year").copy()

    years = g["year"].to_numpy(dtype=int)
    y = g["total_installed_capacity_mw"].to_numpy(dtype=float)

    base_year = fitted_models[first_scen]["base_year"]

    historical_mask = (
        (years >= HISTORICAL_START_YEAR) &
        (years <= HISTORICAL_END_YEAR)
    )

    years_hist = years[historical_mask]
    y_hist = y[historical_mask]
    t_hist = years_hist - base_year

    curve_years = np.arange(years_hist.min(), forecast_end_year + 1)
    curve_t = curve_years - base_year

    exp_fit = least_squares(
        exp_residuals,
        x0=[max(y_hist[0], 1e-3), 0.1],
        args=(t_hist, y_hist)
    )
    exp_y = exp_model(curve_t, *exp_fit.x)

    max_y_plot = df["carrying_capacity_mw"].max() * 1.1
    exp_y_clipped = np.where(exp_y > max_y_plot, np.nan, exp_y)

    y_hist_gw = y_hist / MW_PER_GW
    exp_y_clipped_gw = exp_y_clipped / MW_PER_GW

    plt.figure(figsize=(14, 8))

    # CHANGED: state/country name now prefixed onto the historical-data legend label
# CHANGED: removed the (2014-2026) year range from the historical label
    plt.scatter(
        years_hist,
        y_hist_gw,
        color="black",
        s=80,
        label=f"{state_name} Historical Data",
        zorder=5
    )

    # CHANGED: dropped "(Infinite Limit)" from the legend text
    plt.plot(
        curve_years,
        exp_y_clipped_gw,
        color="grey",
        linestyle=":",
        linewidth=2.5,
        label="Unconstrained Exponential"
    )

    colors = {
        "Conservative": "#d62728",
        "Medium": "#ff7f0e",
        "High": "#2ca02c"
    }

    for scenario, model in fitted_models.items():
        c = colors.get(scenario, "blue")
        L, k, t0 = model["L"], model["k"], model["t0_relative"]

        curve_y = logistic_model(curve_t, L, k, t0)
        curve_y_gw = curve_y / MW_PER_GW
        L_gw = L / MW_PER_GW

        plt.plot(
            curve_years,
            curve_y_gw,
            color=c,
            linewidth=2.5,
            label=f"{scenario} S-Curve"
        )
        plt.axhline(
            L_gw,
            color=c,
            linestyle="--",
            alpha=0.5,
            label=f"{scenario} Ceiling ({L_gw:.1f} GW)"
        )

    plt.axvline(
        HISTORICAL_END_YEAR + 0.5,
        color="black",
        linestyle="-.",
        alpha=0.5,
        label="Forecast Start"
    )

    # CHANGED: title heading removed entirely (was plt.title(...) here)

    # CHANGED: bold axis labels
    # CHANGED: axis labels in Times New Roman, size 18, bold
    plt.xlabel("Year", fontsize=18, fontweight="bold", fontname="Times New Roman")
    plt.ylabel("Installed Capacity (GW)", fontsize=18, fontweight="bold", fontname="Times New Roman")

    # CHANGED: increased the size of the year numbers (x-axis) and
    # capacity numbers (y-axis) tick labels for better visibility
    plt.xticks(fontsize=14, fontname="Times New Roman")
    plt.yticks(fontsize=14, fontname="Times New Roman")

    # CHANGED: legend font size doubled (9 -> 18)
    legend = plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=18,
        framealpha=0.85,
        borderpad=0.4,
        labelspacing=0.35,
        handlelength=1.6
    )

    # CHANGED: bold just the first legend entry (historical data point)
    legend.get_texts()[0].set_fontweight("bold")

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ============================================================
# 4.5 PER-YEAR FORECAST TABLE ACROSS ALL 3 SCENARIOS (GW, CUMULATIVE)
# ============================================================
# Everything above reports the full-data fitted parameters and fit metrics.
# This adds an actual year-by-year forecast value for each of the
# 3 scenarios, from FORECAST_START_YEAR through FORECAST_END_YEAR, using
# each scenario's already-fitted logistic curve - i.e. the actual numbers
# the plot's S-curves are drawing, laid out in a table you can read/export.
#
# IMPORTANT: this table is intentionally left UNROUNDED (full precision).
# Rounding only happens at display() time or in section 4.6 below, AFTER
# annual additions have been derived from these values - never before.

def generate_yearly_forecast_table(
    fitted_models,
    forecast_start_year=FORECAST_START_YEAR,
    forecast_end_year=FORECAST_END_YEAR
):
    if not fitted_models:
        return pd.DataFrame()

    years = np.arange(forecast_start_year, forecast_end_year + 1)

    forecast_df = pd.DataFrame({"year": years})
    for scenario, model in fitted_models.items():
        t = years - model["base_year"]
        forecast_mw = logistic_model(
            t,
            model["L"],
            model["k"],
            model["t0_relative"]
        )
        forecast_df[scenario] = forecast_mw / MW_PER_GW  # full precision GW, not rounded

    return forecast_df

# ============================================================
# 4.6 ANNUAL ADDITIONS TABLE (full-precision diff, rounded after)
# ============================================================
# A_t = C_t - C_{t-1}, computed on the UNROUNDED cumulative model output
# from generate_yearly_forecast_table(). Rounding is applied only to the
# resulting additions series (and only for display/export), never to the
# cumulative values before differencing.
#
# FIX: the first forecast year's addition is now anchored to the ACTUAL
# observed 2026 capacity, not the fitted curve's value at 2026:
#
#   A_2027 = C_2027^forecast - C_2026^actual
#   A_2028 = C_2028^forecast - C_2027^forecast
#   A_t     = C_t^forecast   - C_{t-1}^forecast      for t >= 2028
#
# The fitted logistic curve is not guaranteed to pass exactly through the
# observed 2026 value (it's a least-squares fit over 2014-2026, not an
# interpolation pinned to the last point), so using the model's own 2026
# estimate as the baseline can introduce a small artificial jump/dip into
# the very first addition that has nothing to do with actual 2027 growth.
# Every year from 2028 onward is unaffected by this - it's purely a
# forecast-vs-forecast diff either way - so only the 2027 baseline changes.

def generate_annual_additions_table(
    fitted_models,
    yearly_forecast_df,          # unrounded cumulative GW table from 4.5
    state_df,                    # raw validated data for this state (has actual capacities)
    forecast_start_year=FORECAST_START_YEAR,
    forecast_end_year=FORECAST_END_YEAR
):
    if not fitted_models or yearly_forecast_df.empty:
        return pd.DataFrame()

    years = np.arange(forecast_start_year, forecast_end_year + 1)
    additions_df = pd.DataFrame({"year": years})

    for scenario, model in fitted_models.items():
        if scenario not in yearly_forecast_df.columns:
            continue

        # Baseline C_{t-1} for the FIRST forecast year only: the actual
        # observed capacity at HISTORICAL_END_YEAR (2026), taken directly
        # from the raw data - full precision, not a rounded display value,
        # and not the fitted curve's estimate at that year.
        actual_row = state_df[
            (state_df["scenario"] == scenario) &
            (state_df["year"] == HISTORICAL_END_YEAR)
        ]

        if actual_row.empty:
            # No actual observation at HISTORICAL_END_YEAR for this
            # scenario - fall back to the fitted curve's value so the
            # pipeline doesn't crash, but flag it since this is exactly
            # the case the fix above is meant to avoid.
            print(
                f"  [warn] {scenario}: no actual observation for "
                f"{HISTORICAL_END_YEAR} - falling back to the fitted "
                f"curve's estimate as the 2027 addition baseline."
            )
            t_prev0 = np.array([HISTORICAL_END_YEAR - model["base_year"]])
            c_prev0_mw = logistic_model(t_prev0, model["L"], model["k"], model["t0_relative"])[0]
        else:
            c_prev0_mw = float(actual_row["total_installed_capacity_mw"].iloc[0])

        c_prev0_gw = c_prev0_mw / MW_PER_GW  # actual 2026 capacity, full precision

        cumulative_gw = yearly_forecast_df[scenario].to_numpy(dtype=float)  # unrounded forecast
        # prev_gw[0] = actual 2026 (real anchor); prev_gw[1:] = forecast[:-1] (forecast-on-forecast)
        prev_gw = np.concatenate(([c_prev0_gw], cumulative_gw[:-1]))

        annual_additions_gw = cumulative_gw - prev_gw   # diff BEFORE rounding
        additions_df[scenario] = annual_additions_gw    # round only when displaying/exporting

    return additions_df

# %%
# ============================================================
# 5. FILE DISCOVERY (LOCAL, NOT KAGGLE)
# ============================================================

# CHANGED FOR VS CODE: the original os.walk("/kaggle/input/") only works
# inside a Kaggle notebook, where input datasets get auto-mounted there.
# Locally, we just look directly inside DATA_DIR (set above) - no need to
# walk a whole tree unless you want to. If your CSVs are scattered across
# subfolders, switch the loop below back to os.walk(DATA_DIR).
print(f"--- Looking for data files in: {DATA_DIR} ---")
found_files = []

for file in TARGET_FILES:
    full_path = os.path.join(DATA_DIR, file)
    if os.path.isfile(full_path):
        found_files.append(full_path)
        print(f"Found: {full_path}")
    else:
        print(f"Not found (skipping): {full_path}")

if not found_files:
    raise ValueError(
        f"Could not find any of the target CSV files in {DATA_DIR}. "
        f"Update DATA_DIR at the top of this script to point at the folder "
        f"containing your CSVs, or place the CSVs next to this script."
    )

raw_dfs = []
for file_path in found_files:
    try:
        temp_df = pd.read_csv(file_path)
        temp_df.columns = temp_df.columns.str.strip()

        # FIX: drop whatever the file itself thinks its state/country is (if
        # anything) and set it authoritatively from the filename, for every
        # file - not just the India one. This is the single change that
        # removes the whole class of "region silently merged into India" /
        # "crash on a NaN state" bugs.
        filename = os.path.basename(file_path)
        state_name = FILENAME_TO_STATE.get(filename)
        if state_name is None:
            print(f"Warning: {filename} is not in FILENAME_TO_STATE - skipping to avoid mislabeling its region.")
            continue

        for stale_col in ("state", "country", "Country", "State"):
            if stale_col in temp_df.columns:
                temp_df = temp_df.drop(columns=[stale_col])
        temp_df["state"] = state_name

        raw_dfs.append(temp_df)
    except Exception as e:
        print(f"Warning: Could not load {file_path}. Error: {e}")

if not raw_dfs:
    raise ValueError("No valid CSV files were loaded. Please check DATA_DIR / TARGET_FILES.")

master_df = pd.concat(raw_dfs, ignore_index=True)

df, validation_summary, validation_warnings = validate_pv_csv(master_df)

print("\n--- Master CSV Validation Summary ---")
display(validation_summary)

if validation_warnings:
    print("\nValidation warnings:")
    for warning in validation_warnings:
        print("-", warning)

all_available_states = df['state'].unique()

if TARGET_STATE.lower() == "all":
    states_to_process = all_available_states
else:
    states_to_process = [s for s in all_available_states if s.lower() == TARGET_STATE.lower()]
    if not states_to_process:
        raise ValueError(f"Target state '{TARGET_STATE}' was not found in any of the loaded CSV files! Available states are: {', '.join(all_available_states)}")

print(f"\n============================================================")
print(f" INITIALIZING MODEL PIPELINE FOR: {', '.join(states_to_process)}")
print(f"============================================================")

# %%
all_states_yearly_forecasts = []  # collects the per-year, all-scenarios CUMULATIVE table for every state
all_states_yearly_additions = []  # NEW: collects the per-year, all-scenarios ANNUAL ADDITIONS table for every state

for state in states_to_process:
    print(f"\n\n{'='*60}")
    print(f" REGION: {state.upper()}")
    print(f"{'='*60}")

    state_df = df[df['state'] == state].copy()

    fit_results, fitted_models = evaluate_full_data_by_scenario(state_df)

    print(f"\n--- {state} Model Evaluation (Full Historical Fit: {HISTORICAL_START_YEAR}-{HISTORICAL_END_YEAR}) ---")
    display(fit_results.round({"L_mw": 0, "k_full": 4, "t0_year": 1, "fit_rmse": 0, "fit_r2": 4}))

    plot_all_scenarios_combined(state_df, fitted_models, state_name=state, forecast_end_year=FORECAST_END_YEAR)

    # Build and show the year-by-year CUMULATIVE forecast across all 3
    # scenarios for this state (full precision), then stash it so we can
    # also build one combined table for every state at the end.
    yearly_forecast = generate_yearly_forecast_table(
        fitted_models,
        forecast_start_year=FORECAST_START_YEAR,
        forecast_end_year=FORECAST_END_YEAR
    )
    if not yearly_forecast.empty:
        print(f"\n--- {state} Forecast by Year (All Scenarios, GW, {FORECAST_START_YEAR}-{FORECAST_END_YEAR}) ---")
        display(yearly_forecast.round(1))  # rounded only for display

        # NEW: annual additions, diffed on the UNROUNDED cumulative series
        # above (yearly_forecast), rounded only for display/export below.
        # state_df is passed so 2027's addition can anchor to the ACTUAL
        # observed 2026 capacity rather than the fitted curve's estimate.
        yearly_additions = generate_annual_additions_table(
            fitted_models, yearly_forecast, state_df,
            forecast_start_year=FORECAST_START_YEAR,
            forecast_end_year=FORECAST_END_YEAR
        )
        print(f"\n--- {state} Annual Additions by Year (All Scenarios, GW, {FORECAST_START_YEAR}-{FORECAST_END_YEAR}) ---")
        display(yearly_additions.round(2))  # rounded only for display

        yearly_forecast_labeled = yearly_forecast.copy()
        yearly_forecast_labeled.insert(0, "state", state)
        all_states_yearly_forecasts.append(yearly_forecast_labeled)

        yearly_additions_labeled = yearly_additions.copy()
        yearly_additions_labeled.insert(0, "state", state)
        all_states_yearly_additions.append(yearly_additions_labeled)  # NEW

# %%
# One combined year-by-year, all-scenarios CUMULATIVE forecast table across
# every state that was processed, plus a CSV export next to the input files.
# Saved UNROUNDED (full precision) - matches how it already behaved.
if all_states_yearly_forecasts:
    combined_yearly_forecast = pd.concat(all_states_yearly_forecasts, ignore_index=True)

    print(f"\n\n{'='*60}")
    print(f" COMBINED FORECAST - ALL STATES, ALL SCENARIOS, {FORECAST_START_YEAR}-{FORECAST_END_YEAR}")
    print(f"{'='*60}")
    display(combined_yearly_forecast.round(1))  # rounded only for display

    output_path = (
    OUTPUT_DIR
    / "pv_forecast_logistic_4_regions_2027_2050_gw.csv"
)
    combined_yearly_forecast.to_csv(output_path, index=False)  # unrounded - full precision
    print(f"\nSaved combined cumulative forecast table (full precision) to: {output_path}")

# %%
# NEW: one combined year-by-year, all-scenarios ANNUAL ADDITIONS table
# across every state, plus a CSV export. This is the series the waste
# model should actually consume. It is derived by differencing the
# UNROUNDED cumulative forecast (A_t = C_t - C_{t-1}) and is itself saved
# UNROUNDED - rounding, if you want it, should happen only when the numbers
# are displayed or reported, never before they're diffed or fed downstream.
if all_states_yearly_additions:
    combined_yearly_additions = pd.concat(all_states_yearly_additions, ignore_index=True)

    print(f"\n\n{'='*60}")
    print(f" COMBINED ANNUAL ADDITIONS - ALL STATES, ALL SCENARIOS, {FORECAST_START_YEAR}-{FORECAST_END_YEAR}")
    print(f"{'='*60}")
    display(combined_yearly_additions.round(2))  # rounded only for display

    additions_output_path = (
    OUTPUT_DIR
    / "pv_annual_additions_logistic_4_regions_2027_2050_gw.csv"
)
    combined_yearly_additions.to_csv(additions_output_path, index=False)  # unrounded - full precision
    print(f"\nSaved combined annual-additions table (full precision) to: {additions_output_path}")

# %% [markdown]
# Bass Diffusion

# %%
# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

try:
    from IPython.display import display
except ImportError:
    display = print

# ============================================================
# GLOBAL FONT SETTINGS
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"

# Ensures every display() call below shows full-precision numbers instead of
# pandas' default truncated view. Change the decimal count if you want fewer
# / more digits shown.
pd.set_option("display.float_format", lambda x: f"{x:.10f}")

# ============================================================
# USER SETTINGS
# ============================================================

# The input CSV files are expected to be in the same folder as this script.
# Change DATA_DIR if your CSV files are stored somewhere else.
DATA_DIR = BASE_DIR / "data"

TARGET_STATE = "All"

TARGET_FILES = [
    "karnataka_pv_model_input_constraints.csv",
    "rajasthan_pv_model_input_constraints.csv",
]

# The state name is assigned from the filename so that missing or incorrect
# state columns inside the source CSV cannot merge one region into another.
FILENAME_TO_STATE = {
    "karnataka_pv_model_input_constraints.csv": "Karnataka",
    "rajasthan_pv_model_input_constraints.csv": "Rajasthan",
}

HISTORICAL_START_YEAR = 2014
HISTORICAL_END_YEAR = 2026
FORECAST_START_YEAR = 2027
FORECAST_END_YEAR = 2050

# Model fitting remains in MW. Conversion to GW is applied only to displayed
# and exported forecast tables and graphs.
MW_PER_GW = 1000.0

SCENARIO_ORDER = ["Conservative", "Medium", "High"]

# Standard Bass parameter bounds.
P_LOWER_BOUND = 1e-6
P_UPPER_BOUND = 1.0
Q_LOWER_BOUND = 1e-6
Q_UPPER_BOUND = 5.0
BOUND_TOLERANCE = 1.01

# ============================================================
# 1. CSV VALIDATION
# ============================================================


def validate_pv_csv(df, expected_scenarios=SCENARIO_ORDER):
    df = df.copy()
    df.columns = df.columns.str.strip()

    required_cols = [
        "state",
        "year",
        "scenario",
        "carrying_capacity_mw",
        "total_installed_capacity_mw",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["state"] = df["state"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["carrying_capacity_mw"] = pd.to_numeric(
        df["carrying_capacity_mw"], errors="raise"
    )
    df["total_installed_capacity_mw"] = pd.to_numeric(
        df["total_installed_capacity_mw"], errors="raise"
    )

    bad_state_mask = (
        df["state"].isin(["nan", "None", "", "<NA>"])
        | df["state"].isna()
    )
    if bad_state_mask.any():
        bad_rows = df.loc[bad_state_mask, ["year", "scenario"]].to_dict("records")
        raise ValueError(
            f"{bad_state_mask.sum()} row(s) have no identifiable state/region "
            f"after ingestion (examples: {bad_rows[:5]})."
        )

    errors = []
    warnings = []
    rows = []

    for (state, scenario), g in df.groupby(
        ["state", "scenario"], sort=False, dropna=False
    ):
        g = g.sort_values("year")

        dup_years = g["year"][g["year"].duplicated()].unique()
        if len(dup_years) > 0:
            warnings.append(
                f"{state}/{scenario}: duplicate year(s) {list(dup_years)}."
            )

        if not g["total_installed_capacity_mw"].is_monotonic_increasing:
            warnings.append(
                f"{state}/{scenario}: cumulative installed capacity is not "
                "monotonically increasing."
            )

        unique_L = g["carrying_capacity_mw"].dropna().unique()
        if len(unique_L) != 1:
            errors.append(
                f"{state}/{scenario}: carrying_capacity_mw must be constant "
                f"within a scenario, but found {len(unique_L)} values."
            )
            continue

        L = float(unique_L[0])
        max_capacity = float(g["total_installed_capacity_mw"].max())

        if L <= 0:
            errors.append(f"{state}/{scenario}: carrying capacity must be positive.")

        if max_capacity > L:
            warnings.append(
                f"{state}/{scenario}: installed capacity ({max_capacity:.0f} MW) "
                f"already exceeds the stated ceiling ({L:.0f} MW); the Bass "
                "fit will be unreliable."
            )

        present_scenarios = set(
            df.loc[df["state"] == state, "scenario"].dropna().unique()
        )
        missing_scenarios = [
            s for s in expected_scenarios if s not in present_scenarios
        ]
        if missing_scenarios:
            warnings.append(
                f"{state}: missing expected scenario(s): {missing_scenarios}."
            )

        rows.append(
            {
                "state": state,
                "scenario": scenario,
                "rows": len(g),
                "first_year": int(g["year"].min()),
                "last_year": int(g["year"].max()),
                "carrying_capacity_mw": L,
                "max_installed_capacity_mw": max_capacity,
            }
        )

    if errors:
        raise ValueError("CSV failed validation:\n" + "\n".join(errors))

    summary = pd.DataFrame(rows)
    return df, summary, list(dict.fromkeys(warnings))


# ============================================================
# 2. MATHEMATICAL MODELS
# ============================================================


def bass_model(t, m, p, q):
    """Standard Bass cumulative diffusion curve with fixed market potential m."""
    t = np.asarray(t, dtype=float)
    p_safe = max(float(p), P_LOWER_BOUND)

    exp_term = np.exp(np.clip(-(p_safe + q) * t, -700, 700))
    numerator = 1.0 - exp_term
    denominator = 1.0 + (q / p_safe) * exp_term

    return m * numerator / denominator


def bass_residuals(params, t, y_true, m):
    p, q = params
    return bass_model(t, m, p, q) - y_true


def fit_bass_params(t, y, m):
    """Fit p and q while keeping market potential m fixed by the scenario."""
    x0 = [0.03, 0.38]

    return least_squares(
        bass_residuals,
        x0=x0,
        args=(t, y, m),
        bounds=(
            [P_LOWER_BOUND, Q_LOWER_BOUND],
            [P_UPPER_BOUND, Q_UPPER_BOUND],
        ),
        max_nfev=100000,
    )


def exp_model(t, a, r):
    return a * np.exp(np.clip(r * t, -700, 700))


def exp_residuals(params, t, y_true):
    a, r = params
    return exp_model(t, a, r) - y_true


def score_model(y_true, y_pred):
    """Return in-sample RMSE and R-squared for full-data calibration."""
    if len(y_true) == 0:
        return np.nan, np.nan

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    return rmse, r2


# ============================================================
# 3. FULL-DATA STANDARD BASS FITTING
# ============================================================
# Model selection and train/test evaluation were already completed in
# kag.ipynb. This final script performs NO train/test split. It recalibrates
# the selected Standard Bass model using every historical observation from
# 2014 through 2026 and uses that full-data fit for the 2027-2050 forecasts.


def evaluate_full_data_by_scenario(df, scenario_order=SCENARIO_ORDER):
    results = []
    fitted_models = {}

    scenarios = [s for s in scenario_order if s in df["scenario"].unique()]

    for scenario in scenarios:
        g = df[df["scenario"] == scenario].sort_values("year").copy()

        L = float(g["carrying_capacity_mw"].iloc[0])
        years = g["year"].to_numpy(dtype=int)
        y = g["total_installed_capacity_mw"].to_numpy(dtype=float)

        historical_mask = (
            (years >= HISTORICAL_START_YEAR)
            & (years <= HISTORICAL_END_YEAR)
        )

        years_fit = years[historical_mask]
        y_fit = y[historical_mask]

        if len(y_fit) < 3:
            print(
                f"  [skip] {scenario}: fewer than 3 observations between "
                f"{HISTORICAL_START_YEAR} and {HISTORICAL_END_YEAR}."
            )
            continue

        base_year = int(years_fit.min())
        t_fit = years_fit - base_year

        fit = fit_bass_params(t_fit, y_fit, L)
        p_opt, q_opt = fit.x

        if not fit.success:
            print(
                f"  [warning] {scenario}: optimizer did not report successful "
                f"convergence: {fit.message}"
            )

        q_at_lower_bound = q_opt <= Q_LOWER_BOUND * BOUND_TOLERANCE
        if q_at_lower_bound:
            print(
                f"  [warning] {scenario}: q reached its lower bound "
                f"({q_opt:.2e}). The imitation effect is effectively zero; "
                "interpret the fitted Bass parameters cautiously."
            )

        y_pred_fit = bass_model(t_fit, L, p_opt, q_opt)
        rmse_fit, r2_fit = score_model(y_fit, y_pred_fit)

        results.append(
            {
                "scenario": scenario,
                "L_mw": L,
                "p_full": p_opt,
                "q_full": q_opt,
                "q_at_lower_bound": q_at_lower_bound,
                "fit_rmse": rmse_fit,
                "fit_r2": r2_fit,
            }
        )

        fitted_models[scenario] = {
            "L": L,
            "p": p_opt,
            "q": q_opt,
            "base_year": base_year,
            "historical_start_year": int(years_fit.min()),
            "historical_end_year": int(years_fit.max()),
        }

    return pd.DataFrame(results), fitted_models


# ============================================================
# 4. UNIFIED PLOT FUNCTION
# ============================================================


def plot_all_scenarios_combined(
    df,
    fitted_models,
    state_name,
    forecast_end_year=FORECAST_END_YEAR,
):
    if not fitted_models:
        print(f"  [skip plot] {state_name}: no scenario could be fit.")
        return

    first_scen = list(fitted_models.keys())[0]
    g = df[df["scenario"] == first_scen].sort_values("year").copy()

    years = g["year"].to_numpy(dtype=int)
    y = g["total_installed_capacity_mw"].to_numpy(dtype=float)

    base_year = fitted_models[first_scen]["base_year"]

    historical_mask = (
        (years >= HISTORICAL_START_YEAR)
        & (years <= HISTORICAL_END_YEAR)
    )

    years_hist = years[historical_mask]
    y_hist = y[historical_mask]
    t_hist = years_hist - base_year

    curve_years = np.arange(years_hist.min(), forecast_end_year + 1)
    curve_t = curve_years - base_year

    # Exponential is retained only as an unconstrained visual benchmark.
    exp_fit = least_squares(
        exp_residuals,
        x0=[max(y_hist[0], 1e-3), 0.1],
        args=(t_hist, y_hist),
        max_nfev=100000,
    )
    exp_y = exp_model(curve_t, *exp_fit.x)

    max_y_plot = df["carrying_capacity_mw"].max() * 1.1
    exp_y_clipped = np.where(exp_y > max_y_plot, np.nan, exp_y)

    y_hist_gw = y_hist / MW_PER_GW
    exp_y_clipped_gw = exp_y_clipped / MW_PER_GW

    plt.figure(figsize=(14, 8))

    # State name prefixed onto the historical-data legend label; the
    # (2014-2026) year range has been removed from that label.
    plt.scatter(
        years_hist,
        y_hist_gw,
        color="black",
        s=80,
        label=f"{state_name} Historical Data",
        zorder=5,
    )

    # "(Infinite Limit)" dropped from the legend text.
    plt.plot(
        curve_years,
        exp_y_clipped_gw,
        color="grey",
        linestyle=":",
        linewidth=2.5,
        label="Unconstrained Exponential",
    )

    colors = {
        "Conservative": "#d62728",
        "Medium": "#ff7f0e",
        "High": "#2ca02c",
    }

    for scenario, model in fitted_models.items():
        c = colors.get(scenario, "blue")
        L = model["L"]
        p = model["p"]
        q = model["q"]

        curve_y = bass_model(curve_t, L, p, q)
        curve_y_gw = curve_y / MW_PER_GW
        L_gw = L / MW_PER_GW

        plt.plot(
            curve_years,
            curve_y_gw,
            color=c,
            linewidth=2.5,
            label=f"{scenario} S-Curve (Standard Bass)",
        )
        plt.axhline(
            L_gw,
            color=c,
            linestyle="--",
            alpha=0.5,
            label=f"{scenario} Ceiling ({L_gw:.1f} GW)",
        )

    plt.axvline(
        HISTORICAL_END_YEAR + 0.5,
        color="black",
        linestyle="-.",
        alpha=0.5,
        label="Forecast Start",
    )

    # Title heading intentionally removed (no plt.title(...) call).

    # Bold axis labels, Times New Roman, size 18 for visibility.
    plt.xlabel("Year", fontsize=18, fontweight="bold", fontname="Times New Roman")
    plt.ylabel("Installed Capacity (GW)", fontsize=18, fontweight="bold", fontname="Times New Roman")

    # Increased size of the year numbers (x-axis) and capacity numbers
    # (y-axis) tick labels for better visibility.
    plt.xticks(fontsize=16, fontname="Times New Roman")
    plt.yticks(fontsize=16, fontname="Times New Roman")

    # Legend font size doubled (9 -> 18).
    legend = plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=18,
        framealpha=0.85,
        borderpad=0.4,
        labelspacing=0.35,
        handlelength=1.6,
    )

    # Bold just the first legend entry (historical data point).
    legend.get_texts()[0].set_fontweight("bold")

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4.5 CUMULATIVE FORECAST TABLE (GW)
# ============================================================
# The table is kept at full precision. Rounding is applied only for display,
# never before calculating annual additions or saving downstream model inputs.


def generate_yearly_forecast_table(
    fitted_models,
    forecast_start_year=FORECAST_START_YEAR,
    forecast_end_year=FORECAST_END_YEAR,
):
    if not fitted_models:
        return pd.DataFrame()

    years = np.arange(forecast_start_year, forecast_end_year + 1)
    forecast_df = pd.DataFrame({"year": years})

    for scenario, model in fitted_models.items():
        t = years - model["base_year"]
        forecast_mw = bass_model(
            t,
            model["L"],
            model["p"],
            model["q"],
        )
        forecast_df[scenario] = forecast_mw / MW_PER_GW

    return forecast_df


# ============================================================
# 4.6 ANNUAL ADDITIONS TABLE (GW)
# ============================================================
# A_2027 = C_2027_forecast - C_2026_actual
# A_2028 = C_2028_forecast - C_2027_forecast
# A_t    = C_t_forecast    - C_(t-1)_forecast for t >= 2028
#
# Differencing is performed on full-precision cumulative values. Rounding is
# applied only when displaying results.


def generate_annual_additions_table(
    fitted_models,
    yearly_forecast_df,
    state_df,
    forecast_start_year=FORECAST_START_YEAR,
    forecast_end_year=FORECAST_END_YEAR,
):
    if not fitted_models or yearly_forecast_df.empty:
        return pd.DataFrame()

    years = np.arange(forecast_start_year, forecast_end_year + 1)
    additions_df = pd.DataFrame({"year": years})

    for scenario, model in fitted_models.items():
        if scenario not in yearly_forecast_df.columns:
            continue

        actual_row = state_df[
            (state_df["scenario"] == scenario)
            & (state_df["year"] == HISTORICAL_END_YEAR)
        ]

        if actual_row.empty:
            print(
                f"  [warn] {scenario}: no actual observation for "
                f"{HISTORICAL_END_YEAR}; falling back to the fitted Bass "
                "curve for the first annual-addition baseline."
            )
            t_previous = np.array(
                [HISTORICAL_END_YEAR - model["base_year"]], dtype=float
            )
            previous_capacity_mw = bass_model(
                t_previous,
                model["L"],
                model["p"],
                model["q"],
            )[0]
        else:
            previous_capacity_mw = float(
                actual_row["total_installed_capacity_mw"].iloc[0]
            )

        previous_capacity_gw = previous_capacity_mw / MW_PER_GW
        cumulative_gw = yearly_forecast_df[scenario].to_numpy(dtype=float)

        previous_gw = np.concatenate(
            ([previous_capacity_gw], cumulative_gw[:-1])
        )
        annual_additions_gw = cumulative_gw - previous_gw

        additions_df[scenario] = annual_additions_gw

    return additions_df


# %%
# ============================================================
# 5. FILE DISCOVERY AND INGESTION
# ============================================================

print(f"--- Looking for data files in: {DATA_DIR} ---")
found_files = []

for filename in TARGET_FILES:
    full_path = os.path.join(DATA_DIR, filename)
    if os.path.isfile(full_path):
        found_files.append(full_path)
        print(f"Found: {full_path}")
    else:
        print(f"Not found (skipping): {full_path}")

if not found_files:
    raise ValueError(
        f"Could not find any target CSV files in {DATA_DIR}. Update DATA_DIR "
        "or place the CSV files in the same folder as this script."
    )

raw_dfs = []

for file_path in found_files:
    try:
        temp_df = pd.read_csv(file_path)
        temp_df.columns = temp_df.columns.str.strip()

        filename = os.path.basename(file_path)
        state_name = FILENAME_TO_STATE.get(filename)

        if state_name is None:
            print(
                f"Warning: {filename} is not in FILENAME_TO_STATE; skipping "
                "to avoid assigning the wrong region."
            )
            continue

        for stale_col in ("state", "country", "Country", "State"):
            if stale_col in temp_df.columns:
                temp_df = temp_df.drop(columns=[stale_col])

        temp_df["state"] = state_name
        raw_dfs.append(temp_df)

    except Exception as exc:
        print(f"Warning: Could not load {file_path}. Error: {exc}")

if not raw_dfs:
    raise ValueError("No valid CSV files were loaded.")

master_df = pd.concat(raw_dfs, ignore_index=True)
df, validation_summary, validation_warnings = validate_pv_csv(master_df)

print("\n--- Master CSV Validation Summary ---")
display(validation_summary)

if validation_warnings:
    print("\nValidation warnings:")
    for warning in validation_warnings:
        print("-", warning)

all_available_states = df["state"].unique()

if TARGET_STATE.lower() == "all":
    states_to_process = all_available_states
else:
    states_to_process = [
        state
        for state in all_available_states
        if state.lower() == TARGET_STATE.lower()
    ]
    if not states_to_process:
        raise ValueError(
            f"Target state '{TARGET_STATE}' was not found. Available states: "
            f"{', '.join(all_available_states)}"
        )

print("\n============================================================")
print(f" INITIALIZING MODEL PIPELINE FOR: {', '.join(states_to_process)}")
print("============================================================")


# %%
# ============================================================
# 6. MODEL FITTING, PLOTTING, AND STATE-LEVEL OUTPUTS
# ============================================================

all_states_yearly_forecasts = []
all_states_yearly_additions = []

for state in states_to_process:
    print(f"\n\n{'=' * 60}")
    print(f" REGION: {state.upper()}")
    print(f"{'=' * 60}")

    state_df = df[df["state"] == state].copy()

    # No train/test split is performed in this final script.
    fit_results, fitted_models = evaluate_full_data_by_scenario(state_df)

    print(
        f"\n--- {state} Model Evaluation "
        f"(Full Historical Fit: {HISTORICAL_START_YEAR}-{HISTORICAL_END_YEAR}) ---"
    )
    display(fit_results)  # full precision, actual model output (no rounding)

    plot_all_scenarios_combined(
        state_df,
        fitted_models,
        state_name=state,
        forecast_end_year=FORECAST_END_YEAR,
    )

    yearly_forecast = generate_yearly_forecast_table(
        fitted_models,
        forecast_start_year=FORECAST_START_YEAR,
        forecast_end_year=FORECAST_END_YEAR,
    )

    if not yearly_forecast.empty:
        print(
            f"\n--- {state} Forecast by Year "
            f"(All Scenarios, GW, {FORECAST_START_YEAR}-{FORECAST_END_YEAR}) ---"
        )
        display(yearly_forecast)  # full precision, actual model output (no rounding)

        yearly_additions = generate_annual_additions_table(
            fitted_models,
            yearly_forecast,
            state_df,
            forecast_start_year=FORECAST_START_YEAR,
            forecast_end_year=FORECAST_END_YEAR,
        )

        print(
            f"\n--- {state} Annual Additions by Year "
            f"(All Scenarios, GW, {FORECAST_START_YEAR}-{FORECAST_END_YEAR}) ---"
        )
        display(yearly_additions)  # full precision, actual model output (no rounding)

        yearly_forecast_labeled = yearly_forecast.copy()
        yearly_forecast_labeled.insert(0, "state", state)
        all_states_yearly_forecasts.append(yearly_forecast_labeled)

        yearly_additions_labeled = yearly_additions.copy()
        yearly_additions_labeled.insert(0, "state", state)
        all_states_yearly_additions.append(yearly_additions_labeled)


# %%
# ============================================================
# 7. COMBINED CUMULATIVE FORECAST CSV
# ============================================================

if all_states_yearly_forecasts:
    combined_yearly_forecast = pd.concat(
        all_states_yearly_forecasts,
        ignore_index=True,
    )

    print(f"\n\n{'=' * 60}")
    print(
        f" COMBINED FORECAST - ALL STATES, ALL SCENARIOS, "
        f"{FORECAST_START_YEAR}-{FORECAST_END_YEAR}"
    )
    print(f"{'=' * 60}")
    display(combined_yearly_forecast)  # full precision, actual model output (no rounding)

    forecast_output_path = (
    OUTPUT_DIR
    / "pv_forecast_bass_2_regions_2027_2050_gw.csv"
)
    combined_yearly_forecast.to_csv(forecast_output_path, index=False)

    print(
        "\nSaved combined cumulative forecast table (full precision) to: "
        f"{forecast_output_path}"
    )


# %%
# ============================================================
# 8. COMBINED ANNUAL-ADDITIONS CSV
# ============================================================

if all_states_yearly_additions:
    combined_yearly_additions = pd.concat(
        all_states_yearly_additions,
        ignore_index=True,
    )

    print(f"\n\n{'=' * 60}")
    print(
        f" COMBINED ANNUAL ADDITIONS - ALL STATES, ALL SCENARIOS, "
        f"{FORECAST_START_YEAR}-{FORECAST_END_YEAR}"
    )
    print(f"{'=' * 60}")
    display(combined_yearly_additions)  # full precision, actual model output (no rounding)

    additions_output_path = (
    OUTPUT_DIR
    / "pv_annual_additions_bass_2_regions_2027_2050_gw.csv"
)
    combined_yearly_additions.to_csv(additions_output_path, index=False)

    print(
        "\nSaved combined annual-additions table (full precision) to: "
        f"{additions_output_path}"
    )
# ============================================================
# 9. COMBINE LOGISTIC AND BASS OUTPUTS FOR STREAMLIT
# ============================================================

logistic_forecast_file = (
    OUTPUT_DIR
    / "pv_forecast_logistic_4_regions_2027_2050_gw.csv"
)

bass_forecast_file = (
    OUTPUT_DIR
    / "pv_forecast_bass_2_regions_2027_2050_gw.csv"
)

logistic_additions_file = (
    OUTPUT_DIR
    / "pv_annual_additions_logistic_4_regions_2027_2050_gw.csv"
)

bass_additions_file = (
    OUTPUT_DIR
    / "pv_annual_additions_bass_2_regions_2027_2050_gw.csv"
)

final_forecast_file = (
    OUTPUT_DIR
    / "pv_forecast_all_scenarios_2027_2050_gw.csv"
)

final_additions_file = (
    OUTPUT_DIR
    / "pv_annual_additions_all_scenarios_2027_2050_gw.csv"
)

intermediate_files = [
    logistic_forecast_file,
    bass_forecast_file,
    logistic_additions_file,
    bass_additions_file,
]

missing_files = [
    file_path
    for file_path in intermediate_files
    if not file_path.exists()
]

if missing_files:
    missing_text = "\n".join(
        str(file_path)
        for file_path in missing_files
    )

    raise FileNotFoundError(
        "Cannot create the final capacity files because these "
        f"intermediate files are missing:\n{missing_text}"
    )


# Combine cumulative-capacity forecasts
logistic_forecast_df = pd.read_csv(
    logistic_forecast_file
)

bass_forecast_df = pd.read_csv(
    bass_forecast_file
)

final_forecast_df = pd.concat(
    [
        logistic_forecast_df,
        bass_forecast_df,
    ],
    ignore_index=True,
)

final_forecast_df = final_forecast_df.drop_duplicates(
    subset=["state", "year"],
    keep="last",
)

final_forecast_df = final_forecast_df[
    [
        "state",
        "year",
        "Conservative",
        "Medium",
        "High",
    ]
]

final_forecast_df = final_forecast_df.sort_values(
    ["state", "year"]
)

final_forecast_df.to_csv(
    final_forecast_file,
    index=False,
)


# Combine annual-capacity additions
logistic_additions_df = pd.read_csv(
    logistic_additions_file
)

bass_additions_df = pd.read_csv(
    bass_additions_file
)

final_additions_df = pd.concat(
    [
        logistic_additions_df,
        bass_additions_df,
    ],
    ignore_index=True,
)

final_additions_df = final_additions_df.drop_duplicates(
    subset=["state", "year"],
    keep="last",
)

final_additions_df = final_additions_df[
    [
        "state",
        "year",
        "Conservative",
        "Medium",
        "High",
    ]
]

final_additions_df = final_additions_df.sort_values(
    ["state", "year"]
)

final_additions_df.to_csv(
    final_additions_file,
    index=False,
)


# Confirm that all six required regions are present
expected_regions = {
    "Gujarat",
    "India",
    "Karnataka",
    "Maharashtra",
    "Rajasthan",
    "Tamil Nadu",
}

available_regions = set(
    final_forecast_df["state"].astype(str).unique()
)

missing_regions = expected_regions - available_regions

if missing_regions:
    raise ValueError(
        "The final capacity output is missing these regions: "
        f"{sorted(missing_regions)}"
    )

print("\nFinal Streamlit capacity files created successfully.")

print(f"\nCumulative capacity file:\n{final_forecast_file}")
print(f"\nAnnual additions file:\n{final_additions_file}")

print("\nRegions included:")

for region in sorted(available_regions):
    print(f"- {region}")
