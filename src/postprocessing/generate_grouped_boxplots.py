"""
Generate boxplot figures grouped by scenario block, with all sample sizes
on the same canvas. Produces 4 figures matching the Results subsections:
  1. Constant treatment effect (old 10, 11, 12 -> new 1, 2, 3)
  2. Two-period dynamic (old 1, 4, 7 -> new 4, 5, 6)
  3. Six-period non-staggered dynamic (old 2, 5, 8 -> new 7, 8, 9)
  4. Staggered dynamic (old 3, 6, 9 -> new 10, 11, 12)

Each figure is a 3x3 grid: rows = scenarios (simple -> mid -> complex),
columns = sample sizes (500, 2500, 10000).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import (
    COMPLEXITY_LABELS,
    MODEL_COLORS,
    OLD_TO_NEW,
    boxplot_props,
    setup_plot_style,
)

setup_plot_style(font_size=11)

SCENARIO_GROUPS = {
    "constant_te": {
        "scenarios": [10, 11, 12],
        "title": "Constant Treatment Effect Scenarios",
        "filename": "boxplot_constant_te.png",
    },
    "two_period": {
        "scenarios": [1, 4, 7],
        "title": "Two-Period Scenarios",
        "filename": "boxplot_two_period.png",
    },
    "six_period": {
        "scenarios": [2, 5, 8],
        "title": "Six-Period Non-Staggered Dynamic Scenarios",
        "filename": "boxplot_six_period.png",
    },
    "staggered": {
        "scenarios": [3, 6, 9],
        "title": "Staggered Dynamic Scenarios",
        "filename": "boxplot_staggered.png",
    },
}

SAMPLE_SIZES = [500, 2500, 10000]
SAMPLE_SIZE_LABELS = ["$n = 500$", "$n = 2{,}500$", "$n = 10{,}000$"]


def load_data(results_dir: Path, preset: str, iterations: int) -> pd.DataFrame:
    """Load and concatenate parquet files across sample sizes."""
    frames = []
    for n in SAMPLE_SIZES:
        path = results_dir / f"{n}_{preset}" / f"all_results_n{iterations}.parquet"
        if not path.exists():
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


def generate_group_figure(df: pd.DataFrame, group_cfg: dict, output_dir: Path) -> None:
    scenarios = group_cfg["scenarios"]
    nrows, ncols = len(scenarios), len(SAMPLE_SIZES)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows),
                             sharex=False, sharey="row")
    if nrows == 1:
        axes = axes[np.newaxis, :]

    props = boxplot_props(median_color="black", flier_size=2, median_linewidth=1.5)

    for row, scen in enumerate(scenarios):
        scen_df = df[df["scenario"] == scen]
        models_in_scen = [m for m in ["TWFE", "DML-Chang", "DML-Multi"]
                          if m in scen_df["model"].values]

        for col, n in enumerate(SAMPLE_SIZES):
            ax = axes[row, col]
            cell_df = scen_df[scen_df["n_sample"] == n]

            data_to_plot, labels, colors = [], [], []
            for model in models_in_scen:
                model_df = cell_df[cell_df["model"] == model]
                if not model_df.empty:
                    data_to_plot.append(model_df["estimation_error"].values)
                    labels.append(model)
                    colors.append(MODEL_COLORS.get(model, "gray"))

            if data_to_plot:
                bp = ax.boxplot(data_to_plot, tick_labels=labels, showfliers=True,
                                patch_artist=True, **props)
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.7)

            ax.axhline(0, color="black", linestyle="--", linewidth=1, zorder=0)

            if row == 0:
                ax.set_title(SAMPLE_SIZE_LABELS[col])
            if col == 0:
                ax.set_ylabel(f"Scenario {OLD_TO_NEW[scen]} ({COMPLEXITY_LABELS[scen]})")
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(group_cfg["title"], fontsize=15, fontweight="bold", y=1.01)
    fig.supylabel(r"Estimation Error ($\hat{\tau}$ - True ATT)", fontsize=12, x=-0.01)
    fig.tight_layout()

    output_path = output_dir / group_cfg["filename"]
    fig.savefig(output_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grouped boxplots (one figure per scenario block, all sample sizes)."
    )
    parser.add_argument("--results-dir", "-r", type=str, default="output/simulations",
                        help="Root results directory (default: output/simulations)")
    parser.add_argument("--preset", "-p", type=str, default="light",
                        help="ML preset (default: light)")
    parser.add_argument("--iterations", "-i", type=int, default=2000,
                        help="Number of iterations (determines parquet filename, default: 2000)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for figures (default: <results-dir>)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(results_dir, args.preset, args.iterations)
    print(f"Loaded {len(df):,} rows across sample sizes {SAMPLE_SIZES}")

    for group_cfg in SCENARIO_GROUPS.values():
        generate_group_figure(df, group_cfg, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
