from data_generation_clean import mldid_staggered_did
import logging
import time
import warnings

import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from doubleml import DoubleMLData, DoubleMLIRM
from lightgbm import LGBMClassifier, LGBMRegressor
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Covariates used in all estimators
# ============================================================================
COVARIATES = ["X1", "X2", "X3", "X4"]

# ============================================================================
# LightGBM complexity presets for DML estimators
# ============================================================================
ML_PRESETS = {
    "light": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1},
    "default": {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1},
    "heavy": {"n_estimators": 1000, "max_depth": 3, "learning_rate": 0.05},
}


def twfe_estimate(df: pd.DataFrame) -> dict:
    """Estimate ATT via Two-Way Fixed Effects regression on simulated data.

    Fits: Y_it = alpha_i + theta_t + delta * post_treat_it + eps_it
    where post_treat_it = 1[G_i > 0 and period >= G_i].

    Args:
        df: Simulated panel from mldid_staggered_did.

    Returns:
        Dict with coef, se, ci_low, ci_high.
    """
    df_ols = df.copy()
    # Binary post-treatment indicator: 1 if treated and in post-treatment period
    df_ols["post_treat"] = ((df_ols["G"] > 0) & (df_ols["period"] >= df_ols["G"])).astype(int)

    df_ols = df_ols.set_index(["id", "period"])

    model = PanelOLS(
        dependent=df_ols["Y"],
        exog=df_ols[["post_treat"]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    results = model.fit(cov_type="clustered", cluster_entity=True)

    return {
        "model": "TWFE",
        "coef": results.params["post_treat"],
        "se": results.std_errors["post_treat"],
        "ci_low": results.conf_int().loc["post_treat", "lower"],
        "ci_high": results.conf_int().loc["post_treat", "upper"],
    }


def dml_chang_estimate(df: pd.DataFrame, covariates: list = None, ml_preset: str = "default") -> dict:
    """Estimate ATT via Chang (2020) cross-sectional DML-DiD.

    Strategy for non-staggered (2-period) designs:
        1. Compute Delta Y = Y_post - Y_pre for each unit.
        2. Use DoubleMLIRM with score='ATTE' to estimate the ATT,
           conditioning on baseline covariates.

    For staggered designs with T > 2, uses the earliest treatment cohort
    vs. the pre-period (period 1) as a simplified 2-period comparison.

    Args:
        df: Simulated panel from mldid_staggered_did.
        covariates: List of covariate column names. Defaults to COVARIATES.

    Returns:
        Dict with coef, se, ci_low, ci_high.
    """
    if covariates is None:
        covariates = COVARIATES

    # Identify the first post-treatment period for simple pre/post split
    min_G = df[df["G"] > 0]["G"].min()
    pre_period = min_G - 1  # last pre-treatment period

    # Pre-treatment outcome (period before first treatment)
    y_pre = df[df["period"] == pre_period].set_index("id")["Y"]
    # Post-treatment outcome (at treatment period)
    y_post = df[df["period"] == min_G].set_index("id")["Y"]

    # Delta Y
    delta_y = y_post - y_pre
    delta_y.name = "delta_y"

    # Covariates and treatment from pre-period (time-invariant, just pick one)
    unit_data = df[df["period"] == pre_period].set_index("id")[covariates + ["treat"]]

    # Combine
    ml_df = unit_data.join(delta_y).dropna().reset_index(drop=True)

    dml_data = DoubleMLData(
        data=ml_df,
        y_col="delta_y",
        d_cols="treat",
        x_cols=covariates,
    )

    hp = ML_PRESETS[ml_preset]
    ml_g = LGBMRegressor(**hp, verbose=-1, random_state=42, n_jobs=1)
    ml_m = LGBMClassifier(**hp, verbose=-1, random_state=42, n_jobs=1)

    dml_irm = DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        score="ATTE",
        n_folds=5,
        n_rep=1,
    )
    dml_irm.fit()

    return {
        "model": "DML-Chang",
        "coef": dml_irm.coef[0],
        "se": dml_irm.se[0],
        "ci_low": dml_irm.confint().iloc[0, 0],
        "ci_high": dml_irm.confint().iloc[0, 1],
    }


def dml_multi_estimate(df: pd.DataFrame, covariates: list = None, ml_preset: str = "default") -> dict:
    """Estimate ATT via DoubleMLDIDMulti (Hatamyar et al., 2023).

    Uses the staggered group-time ATT framework with ML nuisance estimation.
    Aggregates to an overall ATT via event study aggregation.

    Args:
        df: Simulated panel from mldid_staggered_did.
        covariates: List of covariate column names. Defaults to COVARIATES.

    Returns:
        Dict with coef, se, ci_low, ci_high.
    """
    if covariates is None:
        covariates = COVARIATES

    df_dml = df.copy()

    # DoubleMLPanelData expects d_cols to be the treatment group column (G)
    # G=0 is never-treated, G>0 is the treatment adoption period
    panel_data = dml.DoubleMLPanelData(
        data=df_dml,
        y_col="Y",
        d_cols="G",
        t_col="period",
        id_col="id",
        x_cols=covariates,
    )

    hp = ML_PRESETS[ml_preset]
    ml_g = LGBMRegressor(**hp, verbose=-1, random_state=42, n_jobs=1)
    ml_m = LGBMClassifier(**hp, verbose=-1, random_state=42, n_jobs=1)

    dml_did = dml.did.DoubleMLDIDMulti(
        obj_dml_data=panel_data,
        ml_g=ml_g,
        ml_m=ml_m,
        gt_combinations="standard",
        control_group="never_treated",
        n_folds=5,
        n_rep=1,
        score="observational",
    )

    dml_did.fit()
    agg = dml_did.aggregate("eventstudy")
    summary = agg.overall_summary

    return {
        "model": "DML-Multi",
        "coef": summary["coef"].values[0],
        "se": summary["se"].values[0] if "se" in summary.columns else np.nan,
        "ci_low": summary["2.5 %"].values[0],
        "ci_high": summary["97.5 %"].values[0],
    }


SCENARIOS = {
    # --- Simple: linear confounding, no covariate effect on Y(0), weak PS ---
    1: {
        "n_periods": 2,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "simple",
        "outcome_complexity": "simple",
        "ps_strength": 0.25,
    },
    2: {
        "n_periods": 6,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "simple",
        "outcome_complexity": "simple",
        "ps_strength": 0.25,
    },
    3: {
        "n_periods": 6,
        "staggered": True,
        "random_assignment": False,
        "confounding_complexity": "simple",
        "outcome_complexity": "simple",
        "ps_strength": 0.25,
    },
    # --- Mid: interaction in confounding, linear outcome, moderate PS ---
    4: {
        "n_periods": 2,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "mid",
        "outcome_complexity": "mid",
        "ps_strength": 0.5,
    },
    5: {
        "n_periods": 6,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "mid",
        "outcome_complexity": "mid",
        "ps_strength": 0.5,
    },
    6: {
        "n_periods": 6,
        "staggered": True,
        "random_assignment": False,
        "confounding_complexity": "mid",
        "outcome_complexity": "mid",
        "ps_strength": 0.5,
    },
    # --- Complex: nonlinear confounding, nonlinear outcome, strong PS ---
    7: {
        "n_periods": 2,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "complex",
        "outcome_complexity": "complex",
        "ps_strength": 1.0,
    },
    8: {
        "n_periods": 6,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "complex",
        "outcome_complexity": "complex",
        "ps_strength": 1.0,
    },
    9: {
        "n_periods": 6,
        "staggered": True,
        "random_assignment": False,
        "confounding_complexity": "complex",
        "outcome_complexity": "complex",
        "ps_strength": 1.0,
    },
    # --- Constant treatment effect variants (mirror 2, 5, 8 with te_form="constant") ---
    10: {
        "n_periods": 6,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "simple",
        "outcome_complexity": "simple",
        "ps_strength": 0.25,
        "te_form": "constant",
    },
    11: {
        "n_periods": 6,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "mid",
        "outcome_complexity": "mid",
        "ps_strength": 0.5,
        "te_form": "constant",
    },
    12: {
        "n_periods": 6,
        "staggered": False,
        "random_assignment": False,
        "confounding_complexity": "complex",
        "outcome_complexity": "complex",
        "ps_strength": 1.0,
        "te_form": "constant",
    },
}


def data_gen(scenario_id, seed=42, n_units_override=None):
    """Generate data for a given scenario number."""
    params = SCENARIOS[scenario_id].copy()
    params["seed"] = seed
    if n_units_override is not None:
        params["n_units"] = n_units_override
    return mldid_staggered_did(**params)


def compute_true_att(df: pd.DataFrame) -> float:
    """Compute the true ATT from simulated counterfactual outcomes.

    Uses the oracle columns y0 and y1 available in simulated data.
    ATT = E[Y(1) - Y(0) | treated, post-treatment].

    Args:
        df: Simulated panel with y0, y1 columns.

    Returns:
        True ATT as a float.
    """
    post_treated = df[(df["treat"] == 1) & (df["period"] >= df["G"])]
    return (post_treated["y1"] - post_treated["y0"]).mean()


def run_monte_carlo(
    scenario_id: int,
    n_iterations: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run Monte Carlo simulation for a given scenario.

    For each replication, generates a new dataset (with a different seed),
    computes the true ATT from counterfactuals, and runs all applicable
    estimators. Stores per-replication results including bias and coverage.

    Estimator selection:
        - TWFE: always run
        - DML-Chang: run for non-staggered scenarios
        - DML-Multi: run for staggered scenarios

    Args:
        scenario_id: Key into the SCENARIOS dictionary (1-9).
        n_iterations: Number of MC replications.
        verbose: If True, print progress updates.

    Returns:
        DataFrame with one row per (replication, estimator), containing:
        scenario, iteration, model, coef, se, ci_low, ci_high,
        true_att, bias, covers_true.
    """
    scenario = SCENARIOS[scenario_id]
    is_staggered = scenario["staggered"]
    results = []
    t_start = time.time()

    for i in range(n_iterations):
        t_iter = time.time()
        seed = 1000 * scenario_id + i  # unique seed per scenario + iteration
        df = data_gen(scenario_id, seed=seed)
        true_att = compute_true_att(df)

        # --- TWFE (always) ---
        try:
            res = twfe_estimate(df)
            res["true_att"] = true_att
            res["bias"] = res["coef"] - true_att
            res["covers_true"] = res["ci_low"] <= true_att <= res["ci_high"]
            res["scenario"] = scenario_id
            res["iteration"] = i
            results.append(res)
        except Exception as e:
            logger.warning(f"TWFE failed iter {i}: {e}")

        # --- DML-Chang (non-staggered only) ---
        if not is_staggered:
            try:
                res = dml_chang_estimate(df)
                res["true_att"] = true_att
                res["bias"] = res["coef"] - true_att
                res["covers_true"] = res["ci_low"] <= true_att <= res["ci_high"]
                res["scenario"] = scenario_id
                res["iteration"] = i
                results.append(res)
            except Exception as e:
                logger.warning(f"DML-Chang failed iter {i}: {e}")

        # --- DML-Multi (staggered only) ---
        if is_staggered:
            try:
                res = dml_multi_estimate(df)
                res["true_att"] = true_att
                res["bias"] = res["coef"] - true_att
                res["covers_true"] = res["ci_low"] <= true_att <= res["ci_high"]
                res["scenario"] = scenario_id
                res["iteration"] = i
                results.append(res)
            except Exception as e:
                logger.warning(f"DML-Multi failed iter {i}: {e}")

        elapsed_iter = time.time() - t_iter
        if verbose and (i + 1) % 10 == 0:
            avg_time = (time.time() - t_start) / (i + 1)
            eta = avg_time * (n_iterations - i - 1)
            logger.info(
                f"Scenario {scenario_id}: {i + 1}/{n_iterations} "
                f"({elapsed_iter:.2f}s/iter, ETA: {eta:.0f}s)"
            )

    return pd.DataFrame(results)


def summarize_mc(mc_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize Monte Carlo results by scenario and estimator.

    Args:
        mc_results: Output from run_monte_carlo (or concatenated runs).

    Returns:
        DataFrame with one row per (scenario, model), containing:
        mean_coef, mean_bias, rmse, coverage_rate, mean_se, n_iters.
    """
    group_cols = ["scenario", "model"]
    if mc_results["n_units"].nunique() > 1:
        group_cols.append("n_units")
    summary = (
        mc_results.groupby(group_cols)
        .agg(
            mean_coef=("coef", "mean"),
            mean_true_att=("true_att", "mean"),
            mean_bias=("bias", "mean"),
            median_bias=("bias", "median"),
            rmse=("bias", lambda x: np.sqrt((x**2).mean())),
            coverage_rate=("covers_true", "mean"),
            mean_se=("se", "mean"),
            n_iters=("iteration", "count"),
        )
        .reset_index()
    )
    return summary


def _run_single_iteration(
    scenario_id: int,
    iteration: int,
    ml_preset: str = "default",
    n_units_list: list[int] = None,
) -> list[dict]:
    """Run all estimators for a single MC iteration. Top-level for pickling.

    Args:
        scenario_id: Key into the SCENARIOS dictionary.
        iteration: Iteration index.
        ml_preset: LightGBM complexity preset key.
        n_units_list: If provided, generate data at max(n_units_list) and
            subsample to each size. If None, use the scenario default.

    Returns:
        List of result dicts (one per estimator per sample size).
    """
    scenario = SCENARIOS[scenario_id]
    is_staggered = scenario["staggered"]
    seed = 1000 * scenario_id + iteration
    results = []

    # Generate data (at max requested size if n_units_list is given)
    if n_units_list:
        df_full = data_gen(scenario_id, seed=seed, n_units_override=max(n_units_list))
    else:
        df_full = data_gen(scenario_id, seed=seed)
        n_units_list = [df_full["id"].nunique()]

    rng = np.random.default_rng(seed)
    all_ids = df_full["id"].unique()

    for n_u in sorted(n_units_list, reverse=True):
        # Subsample units if needed
        if n_u >= len(all_ids):
            df = df_full
        else:
            sampled_ids = rng.choice(all_ids, size=n_u, replace=False)
            df = df_full[df_full["id"].isin(sampled_ids)].reset_index(drop=True)

        true_att = compute_true_att(df)
        n_obs = len(df)
        actual_units = df["id"].nunique()
        n_treated = df.loc[df["treat"] == 1, "id"].nunique()

        meta = {
            "true_att": true_att,
            "scenario": scenario_id,
            "iteration": iteration,
            "n_obs": n_obs,
            "n_units": actual_units,
            "n_treated": n_treated,
        }

        # --- TWFE (always) ---
        try:
            res = twfe_estimate(df)
            res.update(meta)
            res["bias"] = res["coef"] - true_att
            res["covers_true"] = res["ci_low"] <= true_att <= res["ci_high"]
            results.append(res)
        except Exception as e:
            logger.warning(f"Scenario {scenario_id}, TWFE failed iter {iteration} (n={n_u}): {e}")

        # --- DML-Chang (non-staggered only) ---
        if not is_staggered:
            try:
                res = dml_chang_estimate(df, ml_preset=ml_preset)
                res.update(meta)
                res["bias"] = res["coef"] - true_att
                res["covers_true"] = res["ci_low"] <= true_att <= res["ci_high"]
                results.append(res)
            except Exception as e:
                logger.warning(f"Scenario {scenario_id}, DML-Chang failed iter {iteration} (n={n_u}): {e}")

        # --- DML-Multi (staggered only) ---
        if is_staggered:
            try:
                res = dml_multi_estimate(df, ml_preset=ml_preset)
                res.update(meta)
                res["bias"] = res["coef"] - true_att
                res["covers_true"] = res["ci_low"] <= true_att <= res["ci_high"]
                results.append(res)
            except Exception as e:
                logger.warning(f"Scenario {scenario_id}, DML-Multi failed iter {iteration} (n={n_u}): {e}")

    return results


def run_batch(
    scenario_ids: list[int] = None,
    n_iterations: int = 100,
    output_dir: str = "results",
    n_jobs: int = -1,
    verbose: bool = True,
    ml_preset: str = "default",
    n_units_list: list[int] = None,
) -> pd.DataFrame:
    """Run Monte Carlo simulations for multiple scenarios and save results.

    For each scenario, runs n_iterations replications in parallel and saves:
      - results/scenario_{id}_n{n_iterations}.csv: per-iteration results
      - results/all_results_n{n_iterations}.csv: combined results across all scenarios
      - results/summary_n{n_iterations}.csv: aggregated summary statistics

    Supports resuming from existing CSVs (skips completed iterations).

    Args:
        scenario_ids: List of scenario IDs to run. Defaults to all (1-12).
        n_iterations: Number of MC replications per scenario.
        output_dir: Directory to save CSV files.
        n_jobs: Number of parallel workers (-1 = all cores, 1 = sequential).
        verbose: If True, print progress updates.

    Returns:
        Combined DataFrame with all results.
    """
    import os
    from joblib import Parallel, delayed

    if scenario_ids is None:
        scenario_ids = list(SCENARIOS.keys())

    os.makedirs(output_dir, exist_ok=True)

    # Determine actual number of workers
    if n_jobs == -1:
        import multiprocessing
        actual_jobs = multiprocessing.cpu_count()
    else:
        actual_jobs = n_jobs
    logger.info(f"Using {actual_jobs} parallel workers")

    all_results = []

    for sid in scenario_ids:
        csv_path = os.path.join(output_dir, f"scenario_{sid}_n{n_iterations}.csv")

        # Check for existing results to resume from
        start_iter = 0
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            start_iter = int(existing["iteration"].max()) + 1
            all_results.append(existing)
            if start_iter >= n_iterations:
                logger.info(
                    f"Scenario {sid}: already has {start_iter} iterations, skipping"
                )
                continue
            logger.info(f"Scenario {sid}: resuming from iteration {start_iter}")

        remaining_iters = list(range(start_iter, n_iterations))
        t_start = time.time()

        # Process in batches for incremental saving
        batch_size = max(10, actual_jobs * 2)
        for batch_start in range(0, len(remaining_iters), batch_size):
            batch = remaining_iters[batch_start:batch_start + batch_size]

            # Run batch in parallel
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(_run_single_iteration)(sid, i, ml_preset, n_units_list) for i in batch
            )

            # Flatten list of lists
            flat_results = [r for sublist in batch_results for r in sublist]

            # Save incrementally
            batch_df = pd.DataFrame(flat_results)
            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path)
                batch_df = pd.concat([existing, batch_df], ignore_index=True)
            batch_df.to_csv(csv_path, index=False)

            if verbose:
                done = batch_start + len(batch) + start_iter
                elapsed = time.time() - t_start
                iters_done = done - start_iter
                avg_time = elapsed / iters_done if iters_done > 0 else 0
                eta = avg_time * (n_iterations - done)
                logger.info(
                    f"Scenario {sid}: {done}/{n_iterations} "
                    f"({avg_time:.2f}s/iter, ETA: {eta:.0f}s) "
                    f"[saved to {csv_path}]"
                )

        # Load final results for this scenario
        final_scenario = pd.read_csv(csv_path)
        all_results.append(final_scenario)

        elapsed = time.time() - t_start
        logger.info(f"Scenario {sid}: completed in {elapsed:.1f}s")

    # Combine all scenarios
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(output_dir, f"all_results_n{n_iterations}.csv")
    combined.to_csv(combined_path, index=False)

    # Save summary
    summary = summarize_mc(combined)
    summary_path = os.path.join(output_dir, f"summary_n{n_iterations}.csv")
    summary.to_csv(summary_path, index=False)

    logger.info(f"All results saved to {combined_path}")
    logger.info(f"Summary saved to {summary_path}")
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo simulation runner")
    parser.add_argument(
        "-s", "--scenarios", nargs="+", type=int, default=None,
        help="Scenario IDs to run (default: all 1-12)"
    )
    parser.add_argument(
        "-n", "--n-iterations", type=int, default=100,
        help="Number of MC iterations per scenario (default: 100)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="results",
        help="Output directory for CSV files (default: results/)"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=-1,
        help="Number of parallel workers (-1 = all cores, 1 = sequential)"
    )
    parser.add_argument(
        "-m", "--ml-preset", type=str, default="default",
        choices=list(ML_PRESETS.keys()),
        help="LightGBM complexity preset (default: default)"
    )
    parser.add_argument(
        "-u", "--n-units", nargs="+", type=int, default=None,
        help="Sample sizes to test (e.g., -u 500 2500 10000). "
             "Generates at max size and subsamples down."
    )
    args = parser.parse_args()

    run_batch(
        scenario_ids=args.scenarios,
        n_iterations=args.n_iterations,
        output_dir=args.output_dir,
        n_jobs=args.jobs,
        ml_preset=args.ml_preset,
        n_units_list=args.n_units,
    )