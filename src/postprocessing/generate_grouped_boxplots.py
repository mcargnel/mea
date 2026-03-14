"""
Generate boxplot figures grouped by scenario block, with all sample sizes
on the same canvas. Produces 4 figures matching the Results subsections:
  1. Two-period scenarios (1, 4, 7)
  2. Six-period non-staggered (2, 5, 8)
  3. Constant treatment effect (10, 11, 12)
  4. Staggered (3, 6, 9)

Each figure is a 3x3 grid: rows = scenarios (simple → mid → complex),
columns = sample sizes (500, 2500, 10000).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

SCENARIO_GROUPS = {
    "two_period": {
        "scenarios": [1, 4, 7],
        "title": "Two-Period Scenarios",
        "filename": "boxplot_two_period.png",
    },
    "six_period": {
        "scenarios": [2, 5, 8],
        "title": "Six-Period Non-Staggered Scenarios",
        "filename": "boxplot_six_period.png",
    },
    "constant_te": {
        "scenarios": [10, 11, 12],
        "title": "Constant Treatment Effect Scenarios",
        "filename": "boxplot_constant_te.png",
    },
    "staggered": {
        "scenarios": [3, 6, 9],
        "title": "Staggered Scenarios",
        "filename": "boxplot_staggered.png",
    },
}

SAMPLE_SIZES = [500, 2500, 10000]
SAMPLE_SIZE_LABELS = ["$n = 500$", "$n = 2{,}500$", "$n = 10{,}000$"]

COMPLEXITY_LABELS = {
    1: "Simple", 2: "Simple", 3: "Simple", 10: "Simple",
    4: "Mid", 5: "Mid", 6: "Mid", 11: "Mid",
    7: "Complex", 8: "Complex", 9: "Complex", 12: "Complex",
}

COLORS = {
    "TWFE": "#4878D0",
    "DML-Chang": "#EE854A",
    "DML-Multi": "#6ACC64",
}


def load_data(results_dir, preset, iterations):
    """Load and concatenate parquet files across sample sizes."""
    frames = []
    for n in SAMPLE_SIZES:
        path = os.path.join(results_dir, f"{n}_{preset}", f"all_results_n{iterations}.parquet")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        df = pd.read_parquet(path)
        df["n_sample"] = n
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No parquet files found in {results_dir} for preset '{preset}'.")
    combined = pd.concat(frames, ignore_index=True)
    combined["estimation_error"] = combined["coef"] - combined["true_att"]
    return combined


def generate_group_figure(df, group_cfg, output_dir):
    """Generate a single 3x3 figure for one scenario group."""
    scenarios = group_cfg["scenarios"]
    nrows = len(scenarios)
    ncols = len(SAMPLE_SIZES)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows),
                             sharex=False, sharey="row")
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, scen in enumerate(scenarios):
        scen_df = df[df["scenario"] == scen]
        models_in_scen = [m for m in ["TWFE", "DML-Chang", "DML-Multi"]
                          if m in scen_df["model"].values]

        for col, n in enumerate(SAMPLE_SIZES):
            ax = axes[row, col]
            cell_df = scen_df[scen_df["n_sample"] == n]

            data_to_plot = []
            labels = []
            colors = []
            for model in models_in_scen:
                model_df = cell_df[cell_df["model"] == model]
                if not model_df.empty:
                    data_to_plot.append(model_df["estimation_error"].values)
                    labels.append(model)
                    colors.append(COLORS.get(model, "gray"))

            if data_to_plot:
                bp = ax.boxplot(
                    data_to_plot,
                    tick_labels=labels,
                    showfliers=True,
                    patch_artist=True,
                    capprops=dict(color="black"),
                    whiskerprops=dict(color="black"),
                    flierprops=dict(marker="o", markerfacecolor="gray",
                                    markersize=2, alpha=0.3),
                    medianprops=dict(color="black", linewidth=1.5),
                )
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.7)

            ax.axhline(0, color="black", linestyle="--", linewidth=1, zorder=0)

            # Column headers (top row only)
            if row == 0:
                ax.set_title(SAMPLE_SIZE_LABELS[col])

            # Row labels (left column only)
            if col == 0:
                complexity = COMPLEXITY_LABELS[scen]
                ax.set_ylabel(f"Scenario {scen} ({complexity})")

            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(group_cfg["title"], fontsize=15, fontweight="bold", y=1.01)
    fig.supylabel("Estimation Error ($\\hat{\\tau}$ - True ATT)", fontsize=12, x=-0.01)
    fig.tight_layout()

    output_path = os.path.join(output_dir, group_cfg["filename"])
    fig.savefig(output_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate grouped boxplots (one figure per scenario block, all sample sizes)."
    )
    parser.add_argument("--results-dir", "-r", type=str, default="results",
                        help="Root results directory (default: results)")
    parser.add_argument("--preset", "-p", type=str, default="light",
                        help="ML preset (default: light)")
    parser.add_argument("--iterations", "-i", type=int, default=2000,
                        help="Number of iterations (determines parquet filename, default: 2000)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for figures (default: results/)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(args.results_dir, args.preset, args.iterations)
    print(f"Loaded {len(df):,} rows across sample sizes {SAMPLE_SIZES}")

    for group_key, group_cfg in SCENARIO_GROUPS.items():
        generate_group_figure(df, group_cfg, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
