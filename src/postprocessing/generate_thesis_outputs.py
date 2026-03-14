import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set global text sizes for plots to ensure readability in LaTeX
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

def generate_markdown_table(summary_df, output_path):
    """
    Generates a Markdown table summarizing the performance of TWFE vs DML across all 12 scenarios.
    """
    # Define mapping to describe the scenarios
    scenario_desc = {
        1: ("Simple", "No", "Dynamic"),
        2: ("Simple", "No", "Dynamic"),
        3: ("Simple", "Yes", "Dynamic"),
        4: ("Mid", "No", "Dynamic"),
        5: ("Mid", "No", "Dynamic"),
        6: ("Mid", "Yes", "Dynamic"),
        7: ("Complex", "No", "Dynamic"),
        8: ("Complex", "No", "Dynamic"),
        9: ("Complex", "Yes", "Dynamic"),
        10: ("Simple", "No", "Constant"),
        11: ("Mid", "No", "Constant"),
        12: ("Complex", "No", "Constant"),
    }
    
    # Sort the summary by scenario and model to ensure consistent row ordering
    summary_df = summary_df.sort_values(by=["scenario", "model"])
    
    md_code = "## Monte Carlo Simulation Results (100 Iterations, n=2500)\n\n"
    md_code += "| Scenario | Model | Complexity | Staggered | Dynamics | Bias | RMSE | Coverage |\n"
    md_code += "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |\n"
    
    current_scenario = None
    
    for idx, row in summary_df.iterrows():
        scen = int(row['scenario'])
        model = row['model']
        
        # Format the numbers
        bias = f"{row['mean_bias']:.4f}"
        rmse = f"{row['rmse']:.4f}"
        
        # Coverage rate formatting
        cov_raw = row['coverage_rate']
        if pd.isna(cov_raw):
            cov = "N/A"
        else:
            cov = f"{cov_raw:.2f}"
            
        if scen != current_scenario:
            comp, stagg, dyn = scenario_desc[scen]
            scenario_cell = f"**{scen}**"
            comp_cell = comp
            stagg_cell = stagg
            dyn_cell = dyn
        else:
            scenario_cell = ""
            comp_cell = ""
            stagg_cell = ""
            dyn_cell = ""
            
        # Add the row
        md_code += f"| {scenario_cell} | {model} | {comp_cell} | {stagg_cell} | {dyn_cell} | {bias} | {rmse} | {cov} |\n"
        
        current_scenario = scen
        
    with open(output_path, 'w') as f:
        f.write(md_code)
    print(f"Saved Markdown table to: {output_path}")

def generate_boxplots(raw_df, output_path):
    """
    Generates a single 4x3 grid canvas containing boxplots of the Estimation Error
    (Estimated ATT - True ATT) for all 12 scenarios.
    """
    # Calculate Estimation Error
    raw_df['estimation_error'] = raw_df['coef'] - raw_df['true_att']
    
    scenarios = raw_df['scenario'].unique()
    scenarios.sort()
    
    # Create a 4x3 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharex=False, sharey=True)
    axes = axes.flatten()
    
    models = raw_df['model'].unique()
    
    for i, scen in enumerate(scenarios):
        ax = axes[i]
        scen_df = raw_df[raw_df['scenario'] == scen]
        
        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        for model in models:
            model_df = scen_df[scen_df['model'] == model]
            if not model_df.empty:
                data_to_plot.append(model_df['estimation_error'].values)
                labels.append(model)
            
        # Create boxplot
        capprops = dict(color='black')
        whiskerprops = dict(color='black')
        flierprops = dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5)
        medianprops = dict(color='red', linewidth=2)
        
        ax.boxplot(data_to_plot, tick_labels=labels, showfliers=True, 
                   capprops=capprops, whiskerprops=whiskerprops, 
                   flierprops=flierprops, medianprops=medianprops)
        
        # Add horizontal line at zero error
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=0)
        
        ax.set_title(f'Scenario {scen}')
        if i % 3 == 0:
            ax.set_ylabel('Estimation Error ($\hat{\\tau}$ - True ATT)')
            
        # Ensure no grid lines per user request
        ax.grid(False)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved master boxplot canvas to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX table and KDE plots from Monte Carlo results.")
    parser.add_argument("--dir", "-d", type=str, required=True, help="Directory containing all_results_n{iterations}.csv and summary_n{iterations}.csv")
    parser.add_argument("--iterations", "-i", type=int, default=100, help="Number of iterations run (determines filename, default 100)")
    
    args = parser.parse_args()
    
    BASE_DIR = args.dir
    RAW_DATA_PARQUET = os.path.join(BASE_DIR, f"all_results_n{args.iterations}.parquet")
    SUMMARY_CSV = os.path.join(BASE_DIR, f"summary_n{args.iterations}.csv")
    
    MARKDOWN_OUT = os.path.join(BASE_DIR, "markdown_table.md")
    
    # 1. Generate markdown table from the summary file
    if os.path.exists(SUMMARY_CSV):
        summary_df = pd.read_csv(SUMMARY_CSV)
        generate_markdown_table(summary_df, MARKDOWN_OUT)
    else:
        print(f"Summary CSV not found at {SUMMARY_CSV}. Cannot generate Markdown table.")
        
    # 2. Generate Boxplots from the raw iteration file
    if os.path.exists(RAW_DATA_PARQUET):
        raw_df = pd.read_parquet(RAW_DATA_PARQUET)
        MASTER_PLOT_OUT = os.path.join(BASE_DIR, "master_boxplot.png")
        generate_boxplots(raw_df, MASTER_PLOT_OUT)
    else:
        print(f"Raw results Parquet not found at {RAW_DATA_PARQUET}. Cannot generate plots.")
