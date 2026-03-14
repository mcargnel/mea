"""Regenerate simulation result tables in 04_simulations.md from CSV files.

Produces two kinds of tables:
  - A single combined RMSE table (with True ATT) across all sample sizes,
    placed between <!-- TABLE:COMBINED --> markers.
  - Per-sample-size detailed tables (Bias, RMSE, Coverage) for the appendix,
    placed between <!-- TABLE:N --> markers.

Usage:
    uv run src/postprocessing/update_sim_tables.py
    uv run src/postprocessing/update_sim_tables.py --preset heavy
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
MD_FILE = ROOT / "book" / "markdown" / "04_simulations.md"
RESULTS_DIR = ROOT / "results"

SAMPLE_SIZES = [500, 2500, 10000]

# Row ordering: 2-period, 6-period non-staggered, constant TE, staggered
SCENARIO_ORDER = [1, 4, 7, 2, 5, 8, 10, 11, 12, 3, 6, 9]

STAGGERED = {3, 6, 9}


def load_summary(n: int, preset: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{n}_{preset}" / "summary_n2000.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def format_combined_table(dfs: dict[int, pd.DataFrame]) -> str:
    """Build a single pipe-table with True ATT and RMSE columns for each sample size."""
    lines = [
        "| Scenario | Model     | True ATT | RMSE (500) | RMSE (2,500) | RMSE (10,000) |",
        "| :------- | :-------- | -------: | ---------: | -----------: | ------------: |",
    ]

    for scenario in SCENARIO_ORDER:
        dml_name = "DML-Multi" if scenario in STAGGERED else "DML-Chang"
        model_order = {dml_name: 0, "TWFE": 1}

        # Get models present for this scenario (from any sample size)
        ref_df = dfs[SAMPLE_SIZES[0]]
        rows = ref_df[ref_df["scenario"] == scenario].copy()
        if rows.empty:
            continue
        rows = rows.sort_values("model", key=lambda s: s.map(model_order))

        first = True
        for _, ref_row in rows.iterrows():
            model = ref_row["model"]
            sc_col = str(scenario) if first else ""

            # True ATT: show only on first row of each scenario
            if first:
                att = ref_row["mean_true_att"]
                att_s = f"{att:.3f}"
            else:
                att_s = ""

            # RMSE for each sample size
            rmse_cells = []
            for n in SAMPLE_SIZES:
                cell = dfs[n][(dfs[n]["scenario"] == scenario) & (dfs[n]["model"] == model)]
                if not cell.empty:
                    rmse_cells.append(f"{cell.iloc[0]['rmse']:.3f}")
                else:
                    rmse_cells.append("---")

            lines.append(
                f"| {sc_col:<8s} | {model:<9s} | {att_s:>8s} | {rmse_cells[0]:>10s} | {rmse_cells[1]:>12s} | {rmse_cells[2]:>13s} |"
            )
            first = False

    return "\n".join(lines)


def format_detail_table(df: pd.DataFrame) -> str:
    """Build a detailed pipe-table with Bias, RMSE, and Coverage for one sample size."""
    lines = [
        "| Scenario | Model     |   Bias |  RMSE | Coverage |",
        "| :------- | :-------- | -----: | ----: | -------: |",
    ]

    for scenario in SCENARIO_ORDER:
        rows = df[df["scenario"] == scenario].copy()
        if rows.empty:
            continue

        dml_name = "DML-Multi" if scenario in STAGGERED else "DML-Chang"
        model_order = {dml_name: 0, "TWFE": 1}
        rows = rows.sort_values("model", key=lambda s: s.map(model_order))

        first = True
        for _, row in rows.iterrows():
            sc_col = str(scenario) if first else ""
            model = row["model"]
            bias = round(row["mean_bias"], 3) + 0.0
            rmse = round(row["rmse"], 3)
            cov = row["coverage_rate"]

            bias_s = f"{bias:6.3f}"
            rmse_s = f"{rmse:.3f}"
            cov_s = f"{cov:8.3f}" if pd.notna(cov) else "     N/A"

            lines.append(
                f"| {sc_col:<8s} | {model:<9s} | {bias_s} | {rmse_s} | {cov_s} |"
            )
            first = False

    return "\n".join(lines)


def update_marker(marker: str, table_str: str, md_text: str) -> str:
    """Replace content between <!-- marker --> and <!-- /marker --> tags."""
    pattern = re.compile(
        rf"(<!-- {marker} -->\n).*?(\n<!-- /{marker} -->)",
        re.DOTALL,
    )
    new_text, count = pattern.subn(rf"\g<1>{table_str}\g<2>", md_text)
    if count == 0:
        print(f"  WARNING: Marker <!-- {marker} --> not found, skipping.")
    return new_text


def main():
    parser = argparse.ArgumentParser(description="Update simulation tables in markdown")
    parser.add_argument(
        "--preset", default="light", help="ML preset name (default: light)"
    )
    args = parser.parse_args()

    md_text = MD_FILE.read_text()

    # Load all summaries
    dfs = {}
    for n in SAMPLE_SIZES:
        dfs[n] = load_summary(n, args.preset)

    # Combined RMSE table (main text)
    combined = format_combined_table(dfs)
    md_text = update_marker("TABLE:COMBINED", combined, md_text)
    print("Updated combined RMSE table")

    # Per-sample-size detail tables (appendix)
    for n in SAMPLE_SIZES:
        detail = format_detail_table(dfs[n])
        md_text = update_marker(f"TABLE:{n}", detail, md_text)
        print(f"Updated detail table for n={n}")

    MD_FILE.write_text(md_text)
    print(f"Wrote {MD_FILE}")


if __name__ == "__main__":
    main()
