"""Difference-in-Differences estimation using TWFE, DML, and Callaway-Sant'Anna."""

import logging
import os
import warnings

import doubleml as dml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(input_path: str, covariates: list) -> pd.DataFrame:
    """Load and preprocess Stata data file.
    
    Args:
        input_path: Path to the Stata data file.
        covariates: List of covariate column names.
    
    Returns:
        Preprocessed pandas DataFrame.
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_stata(input_path)
    df = df[~df['effyear'].isin([2005, 2009])]
    
    df = df.dropna(subset=covariates)
    
    logger.info(f"Data loaded with shape {df.shape}")
    return df

def twfe_est(df: pd.DataFrame, dep_vars: list, covariates: list) -> pd.DataFrame:
    """Estimate Two-Way Fixed Effects model.
    
    Args:
        df: Input DataFrame.
        dep_vars: List of dependent variable names.
        covariates: List of covariate column names.
    
    Returns:
        Fitted PanelOLS results DataFrame.
    """
    results_dict_twfe = {
        'dep_var': [],
        'coef': [],
        'p_value': [],
        'ci_low': [],
        'ci_high': []
    }

    for dep_var in dep_vars:
        logger.info(f"Estimating TWFE model for dependent variable: {dep_var}")
        df_twfe = df.copy()
        df_twfe['ind'] = np.where(df_twfe['cdl'] > 0, 1, 0)

        df_panel = df_twfe.set_index(['sid', 'year'])
        
        model_covariates = ['ind'] + covariates
        
        model = PanelOLS(
            dependent=df_panel[dep_var],
            exog=df_panel[model_covariates],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )

        results = model.fit(cov_type='clustered', cluster_entity=True)

        results_dict_twfe['dep_var'].append(dep_var)
        results_dict_twfe['coef'].append(results.params['ind'])
        results_dict_twfe['p_value'].append(results.pvalues['ind'])
        results_dict_twfe['ci_low'].append(results.conf_int().iloc[0]['lower'])
        results_dict_twfe['ci_high'].append(results.conf_int().iloc[0]['upper'])
    
    results_df_twfe = pd.DataFrame(results_dict_twfe)
    results_df_twfe['model'] = 'TWFE'
        
    return results_df_twfe

def dml_est(df: pd.DataFrame, dep_vars: list, covariates: list) -> tuple[pd.DataFrame, dml.did.DoubleMLDIDMulti]: #type: ignore
    """Estimate Double Machine Learning DiD model.
    
    Args:
        df: Input DataFrame.
        dep_vars: List of dependent variable names.
        covariates: List of covariate column names.
    
    Returns:
        Tuple of DataFrame with DML estimation results and fitted DML object.
    """
    np.random.seed(42)
    
    df_dml = df.copy()
    
    if df_dml[covariates].isnull().any().any():
        df_dml = df_dml.dropna()
        logger.info("Dropped rows with missing covariate values")

    results_dict_dml = {
        'dep_var': [],
        'coef': [],
        'p_value': [],
        'ci_low': [],
        'ci_high': []
    }
    
    try:
        df_dml['year'] = pd.to_datetime(df_dml['year'], format='%Y')
        df_dml['effyear'] = pd.to_datetime(df_dml['effyear'], format='%Y')
    except Exception as e:
        logger.warning(f"Date conversion failed: {e}. Proceeding with original formats.")

    for dep_var in dep_vars:
        logger.info(f"Estimating DML model for dependent variable: {dep_var}")
        
        dml_data = dml.data.DoubleMLPanelData(
            df_dml,
            y_col=dep_var,
            d_cols="effyear",
            id_col="sid",
            t_col="year",
            x_cols=covariates,
            datetime_unit="Y"
        )

        ml_g = RandomForestRegressor(n_estimators=100, max_depth=3)
        ml_m = RandomForestClassifier(n_estimators=100, max_depth=3)

        dml_did_obj = dml.did.DoubleMLDIDMulti(  # type: ignore
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            gt_combinations="standard",
            control_group="not_yet_treated",
            n_folds=2,
            n_rep=50,
            score='observational'
        )

        dml_did_fit = dml_did_obj.fit()
        dml_did_agg = dml_did_fit.aggregate('eventstudy')
        logger.info(f"DML results for {dml_did_agg}")

        # Extract overall aggregated effect from the aggregation object
        overall_summary = dml_did_agg.overall_summary
        results_dict_dml['dep_var'].append(dep_var)
        results_dict_dml['coef'].append(overall_summary['coef'].values[0])
        results_dict_dml['p_value'].append(overall_summary['P>|t|'].values[0])
        results_dict_dml['ci_low'].append(overall_summary['2.5 %'].values[0])
        results_dict_dml['ci_high'].append(overall_summary['97.5 %'].values[0])

    results_df_dml = pd.DataFrame(results_dict_dml)
    results_df_dml['model'] = 'DML'

    return results_df_dml, dml_did_agg


def plot_event_study(dml_did_agg, output_path: str) -> tuple[plt.Figure, pd.DataFrame]: # type: ignore
    """Plots the dynamic treatment effects over time.
    
    Args:
        dml_did_agg: Aggregated DoubleMLDIDMulti object.
        output_path: Directory path for output files.
    
    Returns:
        Tuple of matplotlib figure and aggregation summary DataFrame.
    """
    logger.info("Generating Event Study Plot...")
    
    agg_summary = dml_did_agg.aggregated_summary
    agg_summary_reset = agg_summary.reset_index()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    x_pos = np.arange(len(agg_summary_reset))
    errors = [
        agg_summary_reset['coef'] - agg_summary_reset['2.5 %'],
        agg_summary_reset['97.5 %'] - agg_summary_reset['coef']
    ]

    ax.errorbar(
        x=x_pos,
        y=agg_summary_reset['coef'],
        yerr=errors,
        fmt='o',
        capsize=5,
        color='#2E86AB',
        markersize=8,
        linewidth=2.5,
        label='DML DiD'
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(agg_summary_reset['index'].values, fontsize=11)
    ax.set_title('Event Study: Dynamic Treatment Effects', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Event Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Treatment Effect', fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11)
    plt.tight_layout()

    return fig, agg_summary_reset


def compare_models(combined_results: pd.DataFrame) -> plt.Figure: # type: ignore
    """Compare model results and create visualization.
    
    Args:
        combined_results: DataFrame with combined model results.
    
    Returns:
        Matplotlib figure with comparison plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    combined_results = combined_results.copy()
    combined_results['err_low'] = combined_results['coef'] - combined_results['ci_low']
    combined_results['err_high'] = combined_results['ci_high'] - combined_results['coef']

    models = combined_results['model'].unique()
    x_pos = np.array([0, 1])
    model_colors = ['#2E86AB', '#A23B72']

    for i, model in enumerate(models):
        model_data = combined_results[combined_results['model'] == model]
        errors = [[model_data['err_low'].values[0]], [model_data['err_high'].values[0]]]

        ax.errorbar(
            x=x_pos[i],
            y=model_data['coef'].values[0],
            yerr=errors,
            fmt='o',
            capsize=8,
            color=model_colors[i],
            markersize=12,
            linewidth=2.5,
            capthick=2.5,
            elinewidth=2.5
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=13, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)

    ax.set_title(
        'Effect of Castle Doctrine on Log Homicide Rate',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    ax.set_ylabel('Coefficient Estimate', fontsize=13, fontweight='bold')

    # Minimalist styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=5)

    # Horizontal line at zero for reference
    ax.axhline(0, color='#666666', linewidth=1.2, linestyle='--', zorder=0, alpha=0.7)

    # Add grid for easier reading
    ax.yaxis.grid(True, linestyle=':', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Add some padding to y-axis
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)

    plt.tight_layout()
    return fig

def save_results(
    fig_compare: plt.Figure, # type: ignore
    fig_event_study: plt.Figure, #type: ignore
    combined_results: pd.DataFrame,
    event_study_agg: pd.DataFrame,
    output_path: str
) -> None:
    """Save figures and results to output directory.
    
    Args:
        fig_compare: Matplotlib figure with model comparison.
        fig_event_study: Matplotlib figure with event study plot.
        combined_results: DataFrame with combined model results.
        event_study_agg: DataFrame with event study aggregation.
        output_path: Directory path for output files.
    """
    # Save comparison plot
    fig_compare.savefig(
        os.path.join(output_path, 'model_comparison_staggered.pdf'),
        format='pdf',
        bbox_inches='tight'
    )
    logger.info(f"Saved comparison plot to {output_path}/model_comparison_staggered.pdf")

    # Save event study plot
    fig_event_study.savefig(
        os.path.join(output_path, 'event_study_aggregation.pdf'),
        format='pdf',
        bbox_inches='tight'
    )
    logger.info(f"Saved event study plot to {output_path}/event_study_aggregation.pdf")

    # Save comparison table
    combined_results.to_latex(
        os.path.join(output_path, 'model_comparison_results_staggered.tex'),
        index=False,
        float_format='%.3f'
    )
    logger.info(f"Saved comparison table to {output_path}/model_comparison_results_staggered.tex")

    # Save event study aggregation table
    event_study_agg.to_latex(
        os.path.join(output_path, 'event_study_aggregation.tex'),
        index=False,
        float_format='%.3f'
    )
    logger.info(f"Saved aggregation table to {output_path}/event_study_aggregation.tex")

def main() -> None:
    """Run main DiD analysis workflow."""
    logger.info("Starting analysis workflow")
    input_path = '/Users/mcargnel/Documents/mea/tesis/input/castle.dta'
    output_path = '/Users/mcargnel/Documents/mea/tesis/output'
    dep_vars = ['l_homicide']
    covariates = [
        'l_police', 'unemployrt', 'income', 'l_exp_subsidy', 'poverty',
        'blackm_15_24', 'whitem_15_24', 'blackm_25_44', 'whitem_25_44'
    ]

    df = load_data(input_path=input_path, covariates=covariates)

    logger.info("Estimating TWFE model")
    twfe_results = twfe_est(df, dep_vars, covariates)

    logger.info("Estimating DML model")
    dml_results, dml_did_obj = dml_est(df, dep_vars, covariates)

    fig_event_study, event_study_agg = plot_event_study(dml_did_obj, output_path)

    combined_results = pd.concat([twfe_results, dml_results], ignore_index=True)

    fig_compare = compare_models(combined_results)
    save_results(fig_compare, fig_event_study, combined_results, event_study_agg, output_path)

    logger.info("\n" + combined_results.to_string())
    logger.info("Analysis workflow completed successfully")


if __name__ == "__main__":
    main()