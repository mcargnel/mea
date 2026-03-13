"""Fix TWFE true_att in staggered scenarios (3, 6, 9).

Problem: TWFE used compute_true_att() (simple average) while DML-Multi used
compute_true_att_eventstudy() (eventstudy-weighted). Both should use the
eventstudy-weighted ATT so they are compared against the same estimand.

This script:
  1. Reads each parquet file in results/
  2. For staggered scenarios, copies DML-Multi's true_att to TWFE rows
     (per iteration), then recalculates bias and covers_true
  3. Regenerates the summary CSV using summarize_mc()
"""

import os
import glob
import numpy as np
import pandas as pd

STAGGERED_SCENARIOS = [3, 6, 9]


def fix_parquet(parquet_path: str) -> bool:
    """Patch a single parquet file. Returns True if changes were made."""
    df = pd.read_parquet(parquet_path)

    # Check if staggered scenarios exist in this file
    stagg_mask = df["scenario"].isin(STAGGERED_SCENARIOS)
    if not stagg_mask.any():
        print(f"  No staggered scenarios found, skipping.")
        return False

    changed = False
    for sid in STAGGERED_SCENARIOS:
        sc_mask = df["scenario"] == sid

        dml_rows = df[sc_mask & (df["model"] == "DML-Multi")]
        twfe_rows = df[sc_mask & (df["model"] == "TWFE")]

        if dml_rows.empty or twfe_rows.empty:
            continue

        # Build a mapping: iteration -> DML-Multi true_att
        dml_att_by_iter = dml_rows.set_index("iteration")["true_att"]

        # For each TWFE row, replace true_att with DML-Multi's value
        twfe_idx = twfe_rows.index
        twfe_iters = df.loc[twfe_idx, "iteration"]
        new_att = twfe_iters.map(dml_att_by_iter)

        if new_att.isna().any():
            n_missing = new_att.isna().sum()
            print(f"  WARNING: {n_missing} TWFE iterations in scenario {sid} "
                  f"have no matching DML-Multi iteration.")
            # Keep original for those
            new_att = new_att.fillna(df.loc[twfe_idx, "true_att"])

        old_att = df.loc[twfe_idx, "true_att"].values
        df.loc[twfe_idx, "true_att"] = new_att.values
        df.loc[twfe_idx, "bias"] = df.loc[twfe_idx, "coef"] - new_att.values
        df.loc[twfe_idx, "covers_true"] = (
            (df.loc[twfe_idx, "ci_low"] <= new_att.values)
            & (new_att.values <= df.loc[twfe_idx, "ci_high"])
        )

        n_changed = (old_att != new_att.values).sum()
        print(f"  Scenario {sid}: patched {n_changed} TWFE rows")
        if n_changed > 0:
            changed = True

    if changed:
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved patched parquet.")

    return changed


def regenerate_summary(parquet_path: str):
    """Regenerate the summary CSV from the (patched) parquet."""
    from monte_carlo_sim import summarize_mc

    df = pd.read_parquet(parquet_path)
    summary = summarize_mc(df)

    # Derive summary CSV path from parquet path
    directory = os.path.dirname(parquet_path)
    basename = os.path.basename(parquet_path)
    # all_results_n2000.parquet -> summary_n2000.csv
    n_str = basename.replace("all_results_", "").replace(".parquet", "")
    summary_path = os.path.join(directory, f"summary_{n_str}.csv")

    summary.to_csv(summary_path, index=False)
    print(f"  Regenerated {summary_path}")


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    parquet_files = sorted(glob.glob(os.path.join(results_dir, "**/*.parquet"), recursive=True))

    if not parquet_files:
        print("No parquet files found in results/")
        return

    print(f"Found {len(parquet_files)} parquet files.\n")

    for pf in parquet_files:
        rel = os.path.relpath(pf, results_dir)
        print(f"Processing {rel}...")
        changed = fix_parquet(pf)
        if changed:
            regenerate_summary(pf)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
