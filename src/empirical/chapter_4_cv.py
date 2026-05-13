"""Difference-in-Differences estimation comparing TWFX and DML methods."""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLIRM
from lightgbm import LGBMClassifier, LGBMRegressor
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TREATMENT_START_YEAR = 2005


def _lgbm_g_param_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 31),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        'verbose': -1,
        'random_state': 42,
    }


def _lgbm_m_param_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 31),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        'verbose': -1,
        'random_state': 42,
    }


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

def _build_ml_data(df: pd.DataFrame, var: str) -> tuple[DoubleMLData, list]:
    """Build DoubleMLData with delta_y outcome (Chang 2020 transformation)."""
    df_pre = df[df['year'] < TREATMENT_START_YEAR].copy()
    df_post = df[df['year'] >= TREATMENT_START_YEAR].copy()

    X_cols = ['lnestab', 'lnemp']
    df_X = df_pre.groupby('zipcode')[X_cols].mean()
    df_D = df.groupby('zipcode')['fracked'].first()

    y_pre = df_pre.groupby('zipcode')[var].mean()
    y_post = df_post.groupby('zipcode')[var].mean()
    delta_y = y_post - y_pre

    ml_df = pd.DataFrame({'delta_y': delta_y}).join([df_D, df_X]).dropna()
    dml_data = DoubleMLData(
        data=ml_df,
        y_col='delta_y',
        d_cols='fracked',
        x_cols=X_cols,
    )
    return dml_data, X_cols


def dml_did_model(
    df: pd.DataFrame,
    dep_vars: dict,
    n_trials: int = 50,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Chang (2020) DML-DiD with LightGBM learners tuned via Optuna."""
    logger.info("Starting DML model estimation (LGBM)")
    results_dict_dml = {'dep_var': [], 'coef': [], 'ci_low': [], 'ci_high': []}

    np.random.seed(42)

    for var in list(dep_vars.values()):
        logger.info(f"Estimating DML for {var}")
        dml_data, _ = _build_ml_data(df, var)

        ml_g = LGBMRegressor(verbose=-1, random_state=42)
        ml_m = LGBMClassifier(verbose=-1, random_state=42)

        dml_irm = DoubleMLIRM(
            dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            score='ATTE',
            n_folds=5,
            n_rep=10
        )

        ml_param_space = {
            'ml_g0': _lgbm_g_param_space,
            'ml_g1': _lgbm_g_param_space,
            'ml_m': _lgbm_m_param_space,
        }
        optuna_settings = {'n_trials': n_trials, 'show_progress_bar': False}
        logger.info(
            f"Optuna tuning for {var} ({n_trials} trials, {cv_folds}-fold CV)"
        )
        tune_res = dml_irm.tune_ml_models(
            ml_param_space=ml_param_space,
            cv=cv_folds,
            optuna_settings=optuna_settings,
            return_tune_res=True,
        )
        best = {k: tune_res[0][k].best_params for k in ('ml_g0', 'ml_g1', 'ml_m')}
        logger.info(f"Best params for {var}: {best}")

        dml_irm.fit()
        logger.info(f"DML model fitted for {var}")

        results_dict_dml['dep_var'].append(var)
        results_dict_dml['coef'].append(dml_irm.coef[0])
        results_dict_dml['ci_low'].append(dml_irm.confint().iloc[0, 0])
        results_dict_dml['ci_high'].append(dml_irm.confint().iloc[0, 1])

    results_df_dml = pd.DataFrame(results_dict_dml)
    results_df_dml['model'] = 'DML (Chang 2020)'
    logger.info("DML estimation completed")
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

    total_width = 0.6
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
        'Model Comparison: Classic DiD vs. Chang (2020) DML',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('Dependent Variable', fontsize=12)
    ax.set_ylabel('Coefficient Estimate', fontsize=12)
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
    input_path = '/home/cama5007/other/mea/input/zc_level.dta'
    output_path = '/home/cama5007/other/mea/output'

    if not os.path.exists(output_path):
        logger.info(f"Creating output directory: {output_path}")
        os.makedirs(output_path)

    df, dep_vars = load_data(input_path)

    results_df_gonzales = twfx_model(df, dep_vars)
    results_df_dml_lgbm = dml_did_model(df, dep_vars)

    fig_compare, combined_results = compare_models(
        results_df_gonzales,
        results_df_dml_lgbm,
    )
    save_results(fig_compare, combined_results, output_path)
    
    logger.info("\n" + combined_results.to_string())
    logger.info("Analysis workflow completed successfully")


if __name__ == "__main__":
    main()
