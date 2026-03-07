import pandas as pd
import numpy as np

def aggregate_results(csv_path: str, output_path: str = None):
    """
    Aggregates the Monte Carlo simulation results from the summary CSV.
    
    For non-staggered scenarios, the results are already aggregated by the
    simulation script (one row per model per scenario).
    
    For staggered scenarios, DoubleML's event-study aggregation implicitly weights
    the group-time ATTs by the number of units in each estimation sample.
    Since the simulation script outputs multiple rows for staggered scenarios
    (with varying `n_units`), this function calculates a weighted average of the
    ATTs, bias, and RMSE using `n_iters` * `n_units` as the weights.
    """
    # Load the summary CSV
    df = pd.read_csv(csv_path)
    
    # We will separate the scenarios that need aggregation from those that don't
    
    # 1. Identify which scenarios are staggered vs non-staggered
    # Looking at the data, Scenarios 3, 6, 9 are the staggered ones that output multiple rows
    staggered_scenarios = [3, 6, 9]
    
    # 2. Split the dataframe
    df_non_staggered = df[~df['scenario'].isin(staggered_scenarios)].copy()
    df_staggered = df[df['scenario'].isin(staggered_scenarios)].copy()
    
    # 3. Aggregate staggered scenarios using a weighted average
    # We use n_iters as a base weight, and conceptually, DoubleML's aggregate() 
    # uses sample size weights, so we weight by n_iters * n_units
    df_staggered['weight'] = df_staggered['n_iters']
    
    def weighted_average(group):
        d = {}
        total_weight = group['weight'].sum()
        
        # Weighted numerical columns
        for col in ['mean_coef', 'mean_true_att', 'mean_bias', 'rmse', 'coverage_rate', 'mean_se']:
            if col in group.columns and not group[col].isna().all():
                d[col] = (group[col] * group['weight']).sum() / total_weight
            else:
                d[col] = np.nan
        
        d['n_iters'] = group['n_iters'].sum()
        
        # We drop n_units and median_bias from the final aggregated result 
        # as they don't aggregate cleanly across varying sample sizes
        return pd.Series(d)

    
    df_staggered_agg = df_staggered.groupby(['scenario', 'model']).apply(weighted_average).reset_index()
    
    # Drop n_units and median_bias from non_staggered to match columns
    df_non_staggered_clean = df_non_staggered.drop(columns=['n_units', 'median_bias'], errors='ignore')
    
    # 4. Combine them back together
    df_final = pd.concat([df_non_staggered_clean, df_staggered_agg], ignore_index=True)
    
    # Sort for clean output
    df_final = df_final.sort_values(['scenario', 'model'])
    
    # Format the table for markdown/latex usage
    print("=== Aggregated Simulation Results ===")
    print(df_final.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    if output_path:
        df_final.to_csv(output_path, index=False)
        print(f"\nSaved aggregated results to {output_path}")
        
    return df_final

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate simulation results.")
    parser.add_argument("input_csv", help="Path to the summary CSV file")
    parser.add_argument("--output", "-o", default=None, help="Path to save the aggregated CSV")
    
    args = parser.parse_args()
    aggregate_results(args.input_csv, args.output)

