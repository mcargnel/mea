"""
Compare ML presets (light, default, heavy) across sample sizes (500, 2500, 10000).

For each n_units, produces:
  1. A console summary table showing Bias, RMSE, Coverage per scenario × model × preset
  2. A grouped bar chart comparing RMSE across presets (one figure per n_units)
  3. A combined CSV with all configs for further analysis

Usage:
    uv run src/postprocessing/compare_presets.py
    uv run src/postprocessing/compare_presets.py -o results/preset_comparison
    uv run src/postprocessing/compare_presets.py --presets light default heavy v_heavy
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
N_UNITS_LIST = [500, 2500, 10000]
DEFAULT_PRESETS = ["light", "default", "heavy"]
N_ITERS = 2000

SCENARIO_ORDER = [1, 4, 7, 2, 5, 8, 10, 11, 12, 3, 6, 9]
MODEL_ORDER = ["DML-Chang", "DML-Multi", "TWFE"]

SCENARIO_DESC = {
    1: "S1: Simple, 2-per",
    2: "S2: Simple, 6-per",
    3: "S3: Simple, stagg",
    4: "S4: Mid, 2-per",
    5: "S5: Mid, 6-per",
    6: "S6: Mid, stagg",
    7: "S7: Complex, 2-per",
    8: "S8: Complex, 6-per",
    9: "S9: Complex, stagg",
    10: "S10: Simple, const",
    11: "S11: Mid, const",
    12: "S12: Complex, const",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
})


def load_all_summaries(presets, n_iters=N_ITERS):
    """Load summary CSVs for all (n_units, preset) combos into one DataFrame."""
    frames = []
    for n in N_UNITS_LIST:
        for preset in presets:
            path = os.path.join(RESULTS_DIR, f"{n}_{preset}", f"summary_n{n_iters}.csv")
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping.")
                continue
            df = pd.read_csv(path)
            df["n_units"] = n
            df["preset"] = preset
            frames.append(df)
    if not frames:
        print("ERROR: No summary files found.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def print_comparison_tables(combined, presets):
    """Print a formatted comparison table for each n_units."""
    for n in N_UNITS_LIST:
        sub = combined[combined["n_units"] == n]
        if sub.empty:
            continue

        print(f"\n{'='*90}")
        print(f"  n_units = {n}")
        print(f"{'='*90}")

        # Build pivot: index = (scenario, model), columns = preset, values = bias/rmse/coverage
        header_parts = ["Scenario  Model       "]
        for p in presets:
            header_parts.append(f"  {'Bias':>7s}  {'RMSE':>6s}  {'Cov':>5s}")
        preset_header = "  |  ".join(f"--- {p.upper():^20s} ---" for p in presets)
        print(f"{'':24s}{preset_header}")
        print("-" * 90)

        for scen in SCENARIO_ORDER:
            scen_sub = sub[sub["scenario"] == scen].copy()
            if scen_sub.empty:
                continue
            models_here = [m for m in MODEL_ORDER if m in scen_sub["model"].values]
            for model in models_here:
                row_str = f"  {scen:>2d}       {model:<12s}"
                for preset in presets:
                    cell = scen_sub[(scen_sub["model"] == model) & (scen_sub["preset"] == preset)]
                    if cell.empty:
                        row_str += f"  {'---':>7s}  {'---':>6s}  {'---':>5s}"
                    else:
                        r = cell.iloc[0]
                        bias_str = f"{r['mean_bias']:+.4f}"
                        rmse_str = f"{r['rmse']:.4f}"
                        cov_val = r["coverage_rate"]
                        cov_str = f"{cov_val:.2f}" if pd.notna(cov_val) else "  N/A"
                        row_str += f"  {bias_str:>8s}  {rmse_str:>6s}  {cov_str:>5s}"
                print(row_str)
            print()


def plot_rmse_comparison(combined, presets, output_dir):
    """For each n_units, create a grouped bar chart comparing RMSE across presets."""
    colors = {
        "light": "#4C72B0",
        "default": "#55A868",
        "heavy": "#C44E52",
        "v_heavy": "#8172B2",
    }

    for n in N_UNITS_LIST:
        sub = combined[combined["n_units"] == n]
        if sub.empty:
            continue

        # Build list of (scenario, model) pairs in order
        labels = []
        for scen in SCENARIO_ORDER:
            scen_sub = sub[sub["scenario"] == scen]
            models_here = [m for m in MODEL_ORDER if m in scen_sub["model"].values]
            for model in models_here:
                labels.append((scen, model))

        n_groups = len(labels)
        n_presets = len(presets)
        bar_width = 0.8 / n_presets
        x = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(16, 6))

        for j, preset in enumerate(presets):
            rmse_vals = []
            for scen, model in labels:
                cell = sub[(sub["scenario"] == scen) & (sub["model"] == model) & (sub["preset"] == preset)]
                rmse_vals.append(cell.iloc[0]["rmse"] if not cell.empty else 0)
            offset = (j - (n_presets - 1) / 2) * bar_width
            ax.bar(x + offset, rmse_vals, bar_width, label=preset,
                   color=colors.get(preset, "#999999"), edgecolor="white", linewidth=0.5)

        # X-axis labels
        tick_labels = [f"S{s}\n{m}" for s, m in labels]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, ha="center")

        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE by ML Preset — n = {n}")
        ax.legend(title="ML Preset")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        out_path = os.path.join(output_dir, f"preset_rmse_n{n}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_bias_comparison(combined, presets, output_dir):
    """For each n_units, create a grouped bar chart comparing absolute bias across presets."""
    colors = {
        "light": "#4C72B0",
        "default": "#55A868",
        "heavy": "#C44E52",
        "v_heavy": "#8172B2",
    }

    for n in N_UNITS_LIST:
        sub = combined[combined["n_units"] == n]
        if sub.empty:
            continue

        labels = []
        for scen in SCENARIO_ORDER:
            scen_sub = sub[sub["scenario"] == scen]
            models_here = [m for m in MODEL_ORDER if m in scen_sub["model"].values]
            for model in models_here:
                labels.append((scen, model))

        n_groups = len(labels)
        n_presets = len(presets)
        bar_width = 0.8 / n_presets
        x = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(16, 6))

        for j, preset in enumerate(presets):
            bias_vals = []
            for scen, model in labels:
                cell = sub[(sub["scenario"] == scen) & (sub["model"] == model) & (sub["preset"] == preset)]
                bias_vals.append(abs(cell.iloc[0]["mean_bias"]) if not cell.empty else 0)
            offset = (j - (n_presets - 1) / 2) * bar_width
            ax.bar(x + offset, bias_vals, bar_width, label=preset,
                   color=colors.get(preset, "#999999"), edgecolor="white", linewidth=0.5)

        tick_labels = [f"S{s}\n{m}" for s, m in labels]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, ha="center")

        ax.set_ylabel("|Bias|")
        ax.set_title(f"Absolute Bias by ML Preset — n = {n}")
        ax.legend(title="ML Preset")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        out_path = os.path.join(output_dir, f"preset_bias_n{n}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_coverage_comparison(combined, presets, output_dir):
    """For each n_units, create a grouped bar chart comparing coverage across presets."""
    colors = {
        "light": "#4C72B0",
        "default": "#55A868",
        "heavy": "#C44E52",
        "v_heavy": "#8172B2",
    }

    for n in N_UNITS_LIST:
        sub = combined[combined["n_units"] == n]
        if sub.empty:
            continue

        labels = []
        for scen in SCENARIO_ORDER:
            scen_sub = sub[sub["scenario"] == scen]
            models_here = [m for m in MODEL_ORDER if m in scen_sub["model"].values]
            for model in models_here:
                labels.append((scen, model))

        n_groups = len(labels)
        n_presets = len(presets)
        bar_width = 0.8 / n_presets
        x = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(16, 6))

        for j, preset in enumerate(presets):
            cov_vals = []
            for scen, model in labels:
                cell = sub[(sub["scenario"] == scen) & (sub["model"] == model) & (sub["preset"] == preset)]
                if not cell.empty and pd.notna(cell.iloc[0]["coverage_rate"]):
                    cov_vals.append(cell.iloc[0]["coverage_rate"])
                else:
                    cov_vals.append(np.nan)
            offset = (j - (n_presets - 1) / 2) * bar_width
            ax.bar(x + offset, cov_vals, bar_width, label=preset,
                   color=colors.get(preset, "#999999"), edgecolor="white", linewidth=0.5)

        # Reference line at 0.95
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1, alpha=0.7, label="95% nominal")

        tick_labels = [f"S{s}\n{m}" for s, m in labels]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, ha="center")

        ax.set_ylabel("Coverage Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Coverage Rate by ML Preset — n = {n}")
        ax.legend(title="ML Preset", loc="lower left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        out_path = os.path.join(output_dir, f"preset_coverage_n{n}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare ML presets across sample sizes.")
    parser.add_argument("-o", "--output", type=str, default=os.path.join(RESULTS_DIR, "preset_comparison"),
                        help="Output directory for plots and CSV (default: results/preset_comparison)")
    parser.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS,
                        help="ML presets to compare (default: light default heavy)")
    parser.add_argument("--n-iters", type=int, default=N_ITERS,
                        help="Number of iterations in summary filename (default: 2000)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading summaries...")
    combined = load_all_summaries(args.presets, args.n_iters)
    print(f"  Loaded {len(combined)} rows across {combined[['n_units','preset']].drop_duplicates().shape[0]} configs.\n")

    # Print tables to console
    print_comparison_tables(combined, args.presets)

    # Save combined CSV
    csv_path = os.path.join(args.output, "all_presets_combined.csv")
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved combined CSV: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_rmse_comparison(combined, args.presets, args.output)
    plot_bias_comparison(combined, args.presets, args.output)
    plot_coverage_comparison(combined, args.presets, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
