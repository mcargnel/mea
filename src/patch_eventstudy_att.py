"""Patch existing parquet results to fix DML-Multi true ATT estimand.

For staggered scenarios (3, 6, 9), the eventstudy aggregation in
DoubleMLDIDMulti targets a different estimand than the sample-weighted
ATT that was stored. This script:

1. Regenerates data for each (scenario, iteration) of DML-Multi staggered rows
2. Computes the eventstudy-weighted true ATT
3. Updates true_att, bias, covers_true in the parquet
4. Re-summarizes and overwrites the summary CSV

Estimator outputs (coef, se, ci_low, ci_high) are NOT modified.
Original parquets are backed up to *.parquet.bak before overwriting.

Usage:
    uv run src/patch_eventstudy_att.py                  # patch all
    uv run src/patch_eventstudy_att.py --dry-run        # preview only
    uv run src/patch_eventstudy_att.py --dirs 500_light 2500_light  # specific configs
"""

import argparse
import glob
import os
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from monte_carlo_sim import (
    SCENARIOS,
    compute_true_att,
    compute_true_att_eventstudy,
    data_gen,
    summarize_mc,
)

STAGGERED_SCENARIOS = [sid for sid, s in SCENARIOS.items() if s["staggered"]]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_VALIDATION_SAMPLES = 5


def parse_n_units_from_folder(folder_name: str) -> int:
    """Extract n_units from folder name like '500_light' -> 500."""
    match = re.match(r"^(\d+)_", folder_name)
    if not match:
        raise ValueError(f"Cannot parse n_units from folder name: {folder_name}")
    return int(match.group(1))


def validate_stored_att(
    df: pd.DataFrame, target_mask: pd.Series, n_units_override: int
) -> bool:
    """Validate that regenerated sample-weighted ATT matches stored values.

    Checks N_VALIDATION_SAMPLES random DML-Multi staggered rows.
    Returns True if all match within tolerance, False otherwise.
    """
    target_rows = df[target_mask]
    n_check = min(N_VALIDATION_SAMPLES, len(target_rows))
    sample = target_rows.sample(n=n_check, random_state=42)

    for _, row in sample.iterrows():
        scenario_id = int(row["scenario"])
        iteration = int(row["iteration"])
        seed = 1000 * scenario_id + iteration
        df_regen = data_gen(scenario_id, seed=seed, n_units_override=n_units_override)
        regen_att = compute_true_att(df_regen)

        if not np.isclose(regen_att, row["true_att"], atol=1e-6):
            print(
                f"  VALIDATION FAILED: scenario={scenario_id}, iter={iteration}: "
                f"stored={row['true_att']:.8f}, regenerated={regen_att:.8f}"
            )
            return False

    print(f"  Validation passed ({n_check} samples checked)")
    return True


def patch_parquet(parquet_path: str, dry_run: bool = False, n_jobs: int = -1) -> dict:
    """Patch a single parquet file. Returns before/after stats dict."""
    folder_name = os.path.basename(os.path.dirname(parquet_path))
    n_units_override = parse_n_units_from_folder(folder_name)
    print(f"\n{'='*60}")
    print(f"Processing: {parquet_path}")
    print(f"  Config: n_units_override={n_units_override}, folder={folder_name}")

    df = pd.read_parquet(parquet_path)

    # Identify target rows
    target_mask = (df["model"] == "DML-Multi") & (
        df["scenario"].isin(STAGGERED_SCENARIOS)
    )
    n_target = target_mask.sum()
    print(f"  Total rows: {len(df)}, DML-Multi staggered rows to patch: {n_target}")

    if n_target == 0:
        print("  No rows to patch, skipping.")
        return None

    # --- Before stats ---
    before_stats = (
        df[target_mask]
        .groupby("scenario")
        .agg(mean_bias=("bias", "mean"), coverage=("covers_true", "mean"))
    )

    # --- Validation ---
    if not validate_stored_att(df, target_mask, n_units_override):
        print("  ABORTING: validation failed for this parquet.")
        return None

    # --- Compute eventstudy ATTs ---
    target_rows = df[target_mask]
    pairs = target_rows[["scenario", "iteration"]].drop_duplicates()
    pairs_list = [
        (int(r.scenario), int(r.iteration))
        for r in pairs.itertuples(index=False)
    ]
    print(f"  Regenerating data for {len(pairs_list)} (scenario, iteration) pairs "
          f"using {n_jobs} workers...")

    t_start = time.time()

    def _compute_one(scenario_id, iteration):
        seed = 1000 * scenario_id + iteration
        df_regen = data_gen(scenario_id, seed=seed, n_units_override=n_units_override)
        return (scenario_id, iteration), compute_true_att_eventstudy(df_regen)

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one)(s, i) for s, i in pairs_list
    )
    es_att_map = dict(results_list)

    elapsed = time.time() - t_start
    rate = len(pairs_list) / elapsed if elapsed > 0 else 0
    print(f"  Data regeneration done in {elapsed:.1f}s ({rate:.1f} pairs/s)")

    # --- Apply patches ---
    target_indices = df.index[target_mask]
    new_true_att = df.loc[target_indices].apply(
        lambda r: es_att_map[(int(r["scenario"]), int(r["iteration"]))], axis=1
    )
    df.loc[target_indices, "true_att"] = new_true_att.values
    df.loc[target_indices, "bias"] = (
        df.loc[target_indices, "coef"] - new_true_att.values
    )
    df.loc[target_indices, "covers_true"] = (
        df.loc[target_indices, "ci_low"] <= new_true_att.values
    ) & (new_true_att.values <= df.loc[target_indices, "ci_high"])

    # --- After stats ---
    after_stats = (
        df[target_mask]
        .groupby("scenario")
        .agg(mean_bias=("bias", "mean"), coverage=("covers_true", "mean"))
    )

    print("\n  Before patch:")
    for scen in STAGGERED_SCENARIOS:
        if scen in before_stats.index:
            b = before_stats.loc[scen]
            print(f"    Scenario {scen}: mean_bias={b['mean_bias']:+.4f}, coverage={b['coverage']:.4f}")
    print("  After patch:")
    for scen in STAGGERED_SCENARIOS:
        if scen in after_stats.index:
            a = after_stats.loc[scen]
            print(f"    Scenario {scen}: mean_bias={a['mean_bias']:+.4f}, coverage={a['coverage']:.4f}")

    if dry_run:
        print("  DRY RUN: not writing files.")
        return {"before": before_stats, "after": after_stats}

    # --- Backup & save parquet ---
    backup_path = parquet_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(parquet_path, backup_path)
        print(f"  Backup saved: {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")

    df.to_parquet(parquet_path, index=False)
    print(f"  Patched parquet saved: {parquet_path}")

    # --- Re-summarize ---
    summary = summarize_mc(df)
    # Infer n_iterations from parquet filename (e.g., all_results_n2000.parquet -> 2000)
    parquet_basename = os.path.basename(parquet_path)
    n_iter_match = re.search(r"n(\d+)", parquet_basename)
    if n_iter_match:
        n_iters_str = n_iter_match.group(1)
    else:
        n_iters_str = "unknown"

    summary_path = os.path.join(
        os.path.dirname(parquet_path), f"summary_n{n_iters_str}.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")

    return {"before": before_stats, "after": after_stats}


def main():
    parser = argparse.ArgumentParser(
        description="Patch DML-Multi true ATT in existing parquet results."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="Specific result folder names to patch (e.g., 500_light 2500_default). "
        "Default: all folders with parquet files.",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for data regeneration (-1 = all cores).",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(RESULTS_DIR)
    print(f"Results directory: {results_dir}")

    # Discover parquet files
    if args.dirs:
        parquet_files = []
        for d in args.dirs:
            pattern = os.path.join(results_dir, d, "all_results_n*.parquet")
            parquet_files.extend(sorted(glob.glob(pattern)))
    else:
        pattern = os.path.join(results_dir, "*", "all_results_n*.parquet")
        parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print("No parquet files found.")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet file(s) to process:")
    for p in parquet_files:
        print(f"  {p}")

    if args.dry_run:
        print("\n*** DRY RUN MODE — no files will be modified ***")

    # Process each
    all_stats = {}
    for pf in parquet_files:
        stats = patch_parquet(pf, dry_run=args.dry_run, n_jobs=args.jobs)
        if stats:
            folder = os.path.basename(os.path.dirname(pf))
            all_stats[folder] = stats

    # Final summary
    print(f"\n{'='*60}")
    print("DONE. Patched configs:", list(all_stats.keys()) if all_stats else "none")


if __name__ == "__main__":
    main()
