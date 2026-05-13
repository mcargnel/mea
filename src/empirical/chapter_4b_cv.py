"""DiD with Optuna-tuned DML (RandomForest learners), TWFE comparison."""

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
    """Load and preprocess Stata data file."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_stata(input_path)
    df = df[~df['effyear'].isin([2005, 2009])]
    df = df.dropna(subset=covariates)
    logger.info(f"Data loaded with shape {df.shape}")
    return df


def twfe_est(df: pd.DataFrame, dep_vars: list, covariates: list) -> pd.DataFrame:
    """Estimate Two-Way Fixed Effects model."""
    results_dict_twfe = {
        'dep_var': [], 'coef': [], 'p_value': [], 'ci_low': [], 'ci_high': []
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


def _ml_g_param_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
        'random_state': 42,
    }


def _ml_m_param_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
        'random_state': 42,
    }


def dml_est(
    df: pd.DataFrame,
    dep_vars: list,
    covariates: list,
    n_trials: int = 20,
    cv_folds: int = 5,
) -> tuple[pd.DataFrame, dml.did.DoubleMLDIDMulti]:  # type: ignore
    """Tune RF learners via Optuna, then fit DoubleMLDIDMulti."""
    np.random.seed(42)

    df_dml = df.copy()
    if df_dml[covariates].isnull().any().any():
        df_dml = df_dml.dropna()
        logger.info("Dropped rows with missing covariate values")

    results_dict_dml = {
        'dep_var': [], 'coef': [], 'p_value': [], 'ci_low': [], 'ci_high': []
    }

    try:
        df_dml['year'] = pd.to_datetime(df_dml['year'], format='%Y')
        df_dml['effyear'] = pd.to_datetime(df_dml['effyear'], format='%Y')
    except Exception as e:
        logger.warning(f"Date conversion failed: {e}.")

    dml_did_agg = None

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

        ml_g = RandomForestRegressor(random_state=42)
        ml_m = RandomForestClassifier(random_state=42)

        dml_did_obj = dml.did.DoubleMLDIDMulti(  # type: ignore
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            gt_combinations="standard",
            control_group="not_yet_treated",
            n_folds=2,
            n_rep=10,
            score='observational'
        )

        ml_param_space = {
            'ml_g0': _ml_g_param_space,
            'ml_g1': _ml_g_param_space,
        }
        optuna_settings = {'n_trials': n_trials, 'show_progress_bar': False}

        logger.info(f"Tuning hyperparameters via Optuna ({n_trials} trials, {cv_folds}-fold CV)")
        dml_did_obj.tune_ml_models(
            ml_param_space=ml_param_space,
            cv=cv_folds,
            optuna_settings=optuna_settings,
        )
        logger.info(f"Tuning complete for {dep_var}")

        dml_did_fit = dml_did_obj.fit()
        dml_did_agg = dml_did_fit.aggregate('eventstudy')
        logger.info(f"DML results for {dml_did_agg}")

        overall_summary = dml_did_agg.overall_summary
        results_dict_dml['dep_var'].append(dep_var)
        results_dict_dml['coef'].append(overall_summary['coef'].values[0])
        results_dict_dml['p_value'].append(overall_summary['P>|t|'].values[0])
        results_dict_dml['ci_low'].append(overall_summary['2.5 %'].values[0])
        results_dict_dml['ci_high'].append(overall_summary['97.5 %'].values[0])

    results_df_dml = pd.DataFrame(results_dict_dml)
    results_df_dml['model'] = 'DML'
    return results_df_dml, dml_did_agg


def plot_event_study(dml_did_agg, output_path: str) -> tuple[plt.Figure, pd.DataFrame]:  # type: ignore
    """Plot dynamic treatment effects (minimal style)."""
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
        x=x_pos, y=agg_summary_reset['coef'], yerr=errors,
        fmt='o', capsize=5, color='#2E86AB', markersize=8, linewidth=2.5,
        label='DML DiD (tuned)'
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agg_summary_reset['index'].values, fontsize=11)
    ax.set_title(
        'Event Study: Dynamic Treatment Effects (Tuned)',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_xlabel('Event Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.95)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    return fig, agg_summary_reset


def compare_models(combined_results: pd.DataFrame) -> plt.Figure:  # type: ignore
    """Compare model results (minimal style, matches chapter_4_cv)."""
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
            x=x_pos[i], y=model_data['coef'].values[0], yerr=errors,
            fmt='o', capsize=5, color=model_colors[i], markersize=8,
            linewidth=2.5, label=model
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(
        'Effect of Castle Doctrine on Log Homicide Rate',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_ylabel('Coefficient Estimate', fontsize=12, fontweight='normal')

    ax.axhline(0, color='#666666', linewidth=1.0, linestyle=':', zorder=0)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    return fig


def save_results(
    fig_compare: plt.Figure,  # type: ignore
    fig_event_study: plt.Figure,  # type: ignore
    combined_results: pd.DataFrame,
    event_study_agg: pd.DataFrame,
    output_path: str
) -> None:
    """Save figures and results."""
    fig_compare.savefig(
        os.path.join(output_path, 'model_comparison_staggered_cv.pdf'),
        format='pdf', bbox_inches='tight'
    )
    logger.info(f"Saved comparison plot to {output_path}/model_comparison_staggered_cv.pdf")

    fig_event_study.savefig(
        os.path.join(output_path, 'event_study_aggregation_cv.pdf'),
        format='pdf', bbox_inches='tight'
    )
    logger.info(f"Saved event study plot to {output_path}/event_study_aggregation_cv.pdf")

    combined_results.to_csv(
        os.path.join(output_path, 'model_comparison_results_staggered_cv.csv'),
        index=False
    )
    combined_results.to_latex(
        os.path.join(output_path, 'model_comparison_results_staggered_cv.tex'),
        index=False, float_format='%.3f'
    )
    event_study_agg.to_csv(
        os.path.join(output_path, 'event_study_aggregation_cv.csv'),
        index=False
    )
    event_study_agg.to_latex(
        os.path.join(output_path, 'event_study_aggregation_cv.tex'),
        index=False, float_format='%.3f'
    )
    logger.info("All results saved")


def main() -> None:
    """Run main DiD analysis with hyperparameter tuning."""
    logger.info("Starting analysis workflow")
    input_path = '/home/cama5007/other/mea/input/castle.dta'
    output_path = '/home/cama5007/other/mea/output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dep_vars = ['l_homicide']
    covariates = [
        'l_police', 'unemployrt', 'income', 'l_exp_subsidy', 'poverty',
        'blackm_15_24', 'whitem_15_24', 'blackm_25_44', 'whitem_25_44'
    ]

    df = load_data(input_path=input_path, covariates=covariates)

    logger.info("Estimating TWFE model")
    twfe_results = twfe_est(df, dep_vars, covariates)

    logger.info("Estimating DML model with Optuna tuning")
    dml_results, dml_did_obj = dml_est(df, dep_vars, covariates, n_trials=50, cv_folds=2)

    fig_event_study, event_study_agg = plot_event_study(dml_did_obj, output_path)

    combined_results = pd.concat([twfe_results, dml_results], ignore_index=True)
    fig_compare = compare_models(combined_results)
    save_results(fig_compare, fig_event_study, combined_results, event_study_agg, output_path)

    logger.info("\n" + combined_results.to_string())
    logger.info("Analysis workflow completed successfully")


if __name__ == "__main__":
    main()
