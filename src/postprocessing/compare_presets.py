"""
Compare ML presets (light, default, heavy) across sample sizes (500, 2500, 10000).

For each n_units, produces:
  1. A console summary table showing Bias, RMSE, Coverage per scenario x model x preset
  2. Grouped bar charts comparing RMSE / |Bias| / Coverage across presets
  3. A combined CSV with all configs for further analysis

Usage:
    uv run src/postprocessing/compare_presets.py
    uv run src/postprocessing/compare_presets.py -o output/simulations/preset_comparison
    uv run src/postprocessing/compare_presets.py --presets light default heavy v_heavy
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from plot_utils import (
    MODEL_ORDER,
    OLD_TO_NEW,
    SCENARIO_ORDER,
    grouped_bar_plot,
    setup_plot_style,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "output" / "simulations"
N_UNITS_LIST = [500, 2500, 10000]
DEFAULT_PRESETS = ["light", "default", "heavy"]
N_ITERS = 2000

setup_plot_style(font_size=11)


def load_all_summaries(presets: list[str], n_iters: int = N_ITERS) -> pd.DataFrame:
    """Load summary CSVs for all (n_units, preset) combos into one DataFrame."""
    frames = []
    for n in N_UNITS_LIST:
        for preset in presets:
            path = RESULTS_DIR / f"{n}_{preset}" / f"summary_n{n_iters}.csv"
            if not path.exists():
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


def print_comparison_tables(combined: pd.DataFrame, presets: list[str]) -> None:
    """Print a formatted comparison table for each n_units."""
    for n in N_UNITS_LIST:
        sub = combined[combined["n_units"] == n]
        if sub.empty:
            continue

        print(f"\n{'='*90}")
        print(f"  n_units = {n}")
        print(f"{'='*90}")

        preset_header = "  |  ".join(f"--- {p.upper():^20s} ---" for p in presets)
        print(f"{'':24s}{preset_header}")
        print("-" * 90)

        for scen in SCENARIO_ORDER:
            scen_sub = sub[sub["scenario"] == scen].copy()
            if scen_sub.empty:
                continue
            models_here = [m for m in MODEL_ORDER if m in scen_sub["model"].values]
            for model in models_here:
                row_str = f"  {OLD_TO_NEW[scen]:>2d}       {model:<12s}"
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


def plot_all_metrics(combined: pd.DataFrame, presets: list[str], output_dir: Path) -> None:
    """For each n_units, render RMSE / |Bias| / Coverage grouped bar charts."""
    for n in N_UNITS_LIST:
        sub = combined[combined["n_units"] == n]
        if sub.empty:
            continue

        grouped_bar_plot(
            sub, presets, lambda r: r["rmse"],
            ylabel="RMSE",
            title=f"RMSE by ML Preset - n = {n}",
            output_path=output_dir / f"preset_rmse_n{n}.png",
        )
        grouped_bar_plot(
            sub, presets, lambda r: abs(r["mean_bias"]),
            ylabel="|Bias|",
            title=f"Absolute Bias by ML Preset - n = {n}",
            output_path=output_dir / f"preset_bias_n{n}.png",
        )
        grouped_bar_plot(
            sub, presets,
            lambda r: r["coverage_rate"] if pd.notna(r["coverage_rate"]) else float("nan"),
            ylabel="Coverage Rate",
            title=f"Coverage Rate by ML Preset - n = {n}",
            output_path=output_dir / f"preset_coverage_n{n}.png",
            ylim=(0, 1.05),
            hline=(0.95, "95% nominal"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ML presets across sample sizes.")
    parser.add_argument("-o", "--output", type=str, default=str(RESULTS_DIR / "preset_comparison"),
                        help="Output directory for plots and CSV (default: results/preset_comparison)")
    parser.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS,
                        help="ML presets to compare (default: light default heavy)")
    parser.add_argument("--n-iters", type=int, default=N_ITERS,
                        help="Number of iterations in summary filename (default: 2000)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading summaries...")
    combined = load_all_summaries(args.presets, args.n_iters)
    print(f"  Loaded {len(combined)} rows across "
          f"{combined[['n_units','preset']].drop_duplicates().shape[0]} configs.\n")

    print_comparison_tables(combined, args.presets)

    csv_path = output_dir / "all_presets_combined.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved combined CSV: {csv_path}")

    print("\nGenerating plots...")
    plot_all_metrics(combined, args.presets, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
