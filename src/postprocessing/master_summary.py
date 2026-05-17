import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import (
    OLD_TO_NEW,
    SCENARIO_META,
    boxplot_props,
    setup_plot_style,
)

setup_plot_style(font_size=12)


def generate_summary_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Write a CSV summary of TWFE vs DML across all 12 scenarios."""
    df = summary_df.sort_values(by=["scenario", "model"]).copy()
    df["scenario"] = df["scenario"].astype(int)
    desc = df["scenario"].map(SCENARIO_META)
    out = pd.DataFrame({
        "scenario": df["scenario"].values,
        "model": df["model"].values,
        "complexity": [d[0] for d in desc],
        "staggered": [d[1] for d in desc],
        "dynamics": [d[2] for d in desc],
        "bias": df["mean_bias"].round(4).values,
        "rmse": df["rmse"].round(4).values,
        "coverage": df["coverage_rate"].round(2).values,
    })
    out.to_csv(output_path, index=False)
    print(f"Saved summary CSV to: {output_path}")


def generate_boxplots(raw_df: pd.DataFrame, output_path: Path) -> None:
    """4x3 grid of estimation-error boxplots for all 12 scenarios."""
    raw_df["estimation_error"] = raw_df["coef"] - raw_df["true_att"]

    scenarios = sorted(raw_df["scenario"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharex=False, sharey=True)
    axes = axes.flatten()

    models = raw_df["model"].unique()
    props = boxplot_props(median_color="red", flier_size=3)

    for i, scen in enumerate(scenarios):
        ax = axes[i]
        scen_df = raw_df[raw_df["scenario"] == scen]

        data_to_plot, labels = [], []
        for model in models:
            model_df = scen_df[scen_df["model"] == model]
            if not model_df.empty:
                data_to_plot.append(model_df["estimation_error"].values)
                labels.append(model)

        ax.boxplot(data_to_plot, tick_labels=labels, showfliers=True, **props)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=0)

        new_num = OLD_TO_NEW.get(scen, scen)
        ax.set_title(f"Scenario {new_num}")
        if i % 3 == 0:
            ax.set_ylabel(r"Estimation Error ($\hat{\tau}$ - True ATT)")

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved master boxplot canvas to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate summary table and master boxplot from Monte Carlo results."
    )
    parser.add_argument("--dir", "-d", type=str, required=True,
                        help="Directory containing all_results_n{i}.parquet and summary_n{i}.csv")
    parser.add_argument("--iterations", "-i", type=int, default=100,
                        help="Number of iterations run (determines filename, default 100)")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    raw_parquet = base_dir / f"all_results_n{args.iterations}.parquet"
    summary_csv = base_dir / f"summary_n{args.iterations}.csv"

    if summary_csv.exists():
        generate_summary_table(pd.read_csv(summary_csv), base_dir / "summary_table.csv")
    else:
        print(f"Summary CSV not found at {summary_csv}. Cannot generate summary table.")

    if raw_parquet.exists():
        generate_boxplots(pd.read_parquet(raw_parquet), base_dir / "master_boxplot.png")
    else:
        print(f"Raw results Parquet not found at {raw_parquet}. Cannot generate plots.")
