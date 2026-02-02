"""Difference-in-Differences estimation comparing TWFX and DML methods."""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLIRM  # type: ignore
from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
from linearmodels.panel import PanelOLS  # type: ignore

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TREATMENT_START_YEAR = 2005 

def load_data(input_path: str) -> tuple[pd.DataFrame, dict]:
    """Load and preprocess Stata data file.
    
    Args:
        input_path: Path to the Stata data file.
    
    Returns:
        Tuple containing:
            - Preprocessed pandas DataFrame.
            - Dictionary of dependent variable names.
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_stata(input_path)
    df['frack_post'] = df['fracked'] * df['treatment']
    df.dropna(inplace=True)
    logger.info(f"Data loaded with shape {df.shape}")

    dep_vars = {
        'Actions': 'lnactionnonoil',
        'Facilities': 'lnone_non_oil',
        'Formal': 'lnstate_formal_nonoil'
    }

    return df, dep_vars

def twfx_model(df: pd.DataFrame, dep_vars: dict) -> pd.DataFrame:
    """Estimate Two-Way Fixed Effects model as in Gonzales.
    
    Args:
        df: Input DataFrame.
        dep_vars: Dictionary of dependent variable names.
    
    Returns:
        Fitted PanelOLS results DataFrame.
    """
    logger.info("Starting TWFX model estimation")
    results_dict_gonzales = {
        'dep_var': [],
        'coef': [],
        'ci_low': [],
        'ci_high': []
    }

    df_gonzales = df.set_index(['zipcode','year']).copy()

    for var in list(dep_vars.values()):
        logger.info(f"Estimating TWFX model for dependent variable: {var}")
        model1 = PanelOLS(
            dependent=df_gonzales[var],
            exog=df_gonzales[[
                'frack_post', 'fracked', 'treatment', 'lnestab', 'lnemp'
            ]],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )

        results = model1.fit(cov_type='clustered', cluster_entity=True)

        results_dict_gonzales['dep_var'].append(var)
        results_dict_gonzales['coef'].append(results.params['frack_post'])
        results_dict_gonzales['ci_low'].append(results.conf_int().iloc[0]['lower'])
        results_dict_gonzales['ci_high'].append(results.conf_int().iloc[0]['upper'])

    results_df_gonzales = pd.DataFrame(results_dict_gonzales)
    results_df_gonzales['model'] = 'Classic DiD'
    logger.info("TWFX model estimation completed")

    return results_df_gonzales

def dml_did_model(df: pd.DataFrame, dep_vars: dict) -> pd.DataFrame:
    """Implement Chang (2020) Double Machine Learning for DiD.
    
    Strategy:
        1. Reshape data to Zipcode level.
        2. Calculate Delta Y (Post_Avg - Pre_Avg).
        3. Use DoubleMLIRM with score='ATTE' to regress Delta Y on Treatment
           and Baseline Covariates.
    
    Args:
        df: Input DataFrame.
        dep_vars: Dictionary of dependent variable names.
    
    Returns:
        DataFrame with DML estimation results.
    """
    logger.info("Starting DML model estimation")
    results_dict_dml = {
        'dep_var': [],
        'coef': [],
        'ci_low': [],
        'ci_high': []
    }

    # Data Transformation for Chang (2020)
    # Define Pre and Post periods
    logger.info(f"Splitting data at treatment year: {TREATMENT_START_YEAR}")
    df_pre = df[df['year'] < TREATMENT_START_YEAR].copy()
    df_post = df[df['year'] >= TREATMENT_START_YEAR].copy()

    # Aggregate Controls (X)
    # MUST use Pre-treatment values to avoid bad controls
    # Take the mean of covariates over the pre-period for each zipcode
    X_cols = ['lnestab', 'lnemp']
    df_X = df_pre.groupby('zipcode')[X_cols].mean()

    # Get Treatment Indicator (D) - 'fracked' is time-invariant per zip
    df_D = df.groupby('zipcode')['fracked'].first()

    np.random.seed(42)

    for var in list(dep_vars.values()):
        logger.info(f"Estimating DML model for dependent variable: {var}")
        # Calculate Outcome (Y) - Change over time (Delta Y)
        y_pre = df_pre.groupby('zipcode')[var].mean()
        y_post = df_post.groupby('zipcode')[var].mean()

        # This is the "Repeated Outcomes" transformation: Y = Y(1) - Y(0)
        delta_y = y_post - y_pre

        # Combine into a single dataframe for DoubleML
        # Inner join ensures we only keep zipcodes in both pre and post
        ml_df = pd.DataFrame(
            {'delta_y': delta_y}
        ).join([df_D, df_X]).dropna()

        # Define DoubleML Data
        dml_data = DoubleMLData(
            data=ml_df,
            y_col='delta_y',    # The change in outcome
            d_cols='fracked',   # The treatment group indicator
            x_cols=X_cols       # Baseline covariates
        )

        # Define Learners
        ml_g = LGBMRegressor(
            n_estimators=50,
            learning_rate=0.0001,
            verbose=-1,
            random_state=42
        )
        ml_m = LGBMClassifier(
            n_estimators=50,
            learning_rate=0.0001,
            verbose=-1,
            random_state=42
        )

        # Initialize DoubleMLIRM (Interactive Regression Model)
        # score='ATTE' = Average Treatment Effect on the Treated
        dml_irm = DoubleMLIRM(
            dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            score='ATTE',
            n_folds=5,
            n_rep=50
        )

        # Fit model
        dml_irm.fit()
        logger.info(f"DML model fitted for {var}")

        results_dict_dml['dep_var'].append(var)
        results_dict_dml['coef'].append(dml_irm.coef[0])
        results_dict_dml['ci_low'].append(dml_irm.confint().iloc[0, 0])
        results_dict_dml['ci_high'].append(dml_irm.confint().iloc[0, 1])

    results_df_dml = pd.DataFrame(results_dict_dml)
    results_df_dml['model'] = 'DML (Chang 2020)'
    logger.info("DML model estimation completed")

    return results_df_dml

def compare_models(
    results_df_gonzales: pd.DataFrame,
    results_df_dml: pd.DataFrame
) -> tuple:
    """Compare model results and create visualization.
    
    Args:
        results_df_gonzales: TWFX model results.
        results_df_dml: DML model results.
    
    Returns:
        Tuple of (figure, combined_results DataFrame).
    """
    logger.info("Comparing models and creating visualization")
    combined_results = pd.concat([results_df_gonzales, results_df_dml])
    combined_results['dep_var'] = combined_results['dep_var'].replace({
        'lnactionnonoil': 'Actions',
        'lnone_non_oil': 'Facilities',
        'lnstate_formal_nonoil': 'Formal'
    })

    combined_results[
        ['dep_var', 'model', 'ci_low', 'coef', 'ci_high']
    ].sort_values('dep_var')

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    combined_results['err_low'] = combined_results['coef'] - combined_results['ci_low']
    combined_results['err_high'] = combined_results['ci_high'] - combined_results['coef']

    groups = combined_results['dep_var'].unique()
    models = combined_results['model'].unique()

    x_pos = np.arange(len(groups))

    total_width = 0.4
    dodge_width = total_width / len(models)
    model_colors = ['#2E86AB', '#A23B72']

    for i, model in enumerate(models):
        model_data = combined_results[combined_results['model'] == model]

        current_x = []
        for var in model_data['dep_var']:
            idx = np.where(groups == var)[0][0]
            current_x.append(x_pos[idx])
        current_x = np.array(current_x)

        shift = (i - (len(models) - 1) / 2) * dodge_width
        errors = [model_data['err_low'], model_data['err_high']]

        ax.errorbar(x=current_x + shift, 
            y=model_data['coef'],
            yerr=errors,
            fmt='o',
            capsize=5,
            linestyle='None',
            label=model,
            color=model_colors[i],
            markersize=8,
            linewidth=2.5
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=11)

    ax.set_title(
        'Model Comparison: Classic DiD vs Chang (2020) DML',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('Dependent Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient Estimate', fontsize=12, fontweight='bold')
    ax.legend(
        title='Model',
        fontsize=11,
        title_fontsize=11,
        framealpha=0.95,
        shadow=False
    )

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', which='major', labelsize=10)

    return fig, combined_results

def save_results(
    fig_compare: plt.Figure, #type: ignore
    combined_results: pd.DataFrame,
    output_path: str
) -> None:
    """Save figure and results to output directory.
    
    Args:
        fig_compare: Matplotlib figure to save.
        combined_results: DataFrame with combined model results.
        output_path: Directory path for output files.
    """
    logger.info(f"Saving results to {output_path}")
    fig_compare.savefig(
        os.path.join(output_path, 'model_comparison.pdf'),
        format='pdf',
        bbox_inches='tight'
    )

    combined_results.to_csv(
        os.path.join(output_path, 'model_comparison_results.csv'),
        index=False
    )
    combined_results.to_latex(
        os.path.join(output_path, 'model_comparison_results.tex'),
        index=False,
        float_format='%.4f'
    )
    logger.info("All results saved successfully")

def main():
    """Run main analysis workflow."""
    logger.info("Starting analysis workflow")
    input_path = '/Users/mcargnel/Documents/mea/tesis/input/zc_level.dta'
    output_path = '/Users/mcargnel/Documents/mea/tesis/output'

    if not os.path.exists(output_path):
        logger.info(f"Creating output directory: {output_path}")
        os.makedirs(output_path)

    df, dep_vars = load_data(input_path)

    results_df_gonzales = twfx_model(df, dep_vars)
    results_df_dml = dml_did_model(df, dep_vars)

    fig_compare, combined_results = compare_models(
        results_df_gonzales,
        results_df_dml
    )
    save_results(fig_compare, combined_results, output_path)
    
    logger.info("\n" + combined_results.to_string())
    logger.info("Analysis workflow completed successfully")


if __name__ == "__main__":
    main()