"""Regenerate tuned (CV) plots from saved CSVs, without refitting."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT = "output/empirical"


def _style(ax):
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', which='major', labelsize=10)


def plot_model_comparison(csv_path: str, out_stem: str) -> None:
    df = pd.read_csv(csv_path)
    df['model'] = df['model'].replace({'DML-LGBM (tuned)': 'DML (Chang 2020)'})

    if 'err_low' not in df.columns:
        df['err_low'] = df['coef'] - df['ci_low']
        df['err_high'] = df['ci_high'] - df['coef']

    df['dep_var'] = df['dep_var'].replace({
        'lnactionnonoil': 'Actions',
        'lnone_non_oil': 'Facilities',
        'lnstate_formal_nonoil': 'Formal',
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    groups = df['dep_var'].unique()
    models = df['model'].unique()
    x_pos = np.arange(len(groups))
    total_width = 0.6
    dodge_width = total_width / len(models)
    colors = ['#2E86AB', '#A23B72']

    for i, model in enumerate(models):
        md = df[df['model'] == model]
        cx = np.array([np.where(groups == v)[0][0] for v in md['dep_var']])
        shift = (i - (len(models) - 1) / 2) * dodge_width
        ax.errorbar(
            x=cx + shift, y=md['coef'],
            yerr=[md['err_low'], md['err_high']],
            fmt='o', capsize=5, linestyle='None',
            label=model, color=colors[i], markersize=8, linewidth=2.5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_title(
        'Model Comparison: Classic DiD vs. Chang (2020) DML',
        fontsize=14, fontweight='bold', pad=15,
    )
    ax.set_xlabel('Dependent Variable', fontsize=12)
    ax.set_ylabel('Coefficient Estimate', fontsize=12)
    ax.legend(title='Model', fontsize=11, title_fontsize=11, framealpha=0.95)
    _style(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, f'{out_stem}.png'), format='png', bbox_inches='tight')
    plt.close(fig)


def plot_staggered_cv(csv_path: str, out_stem: str) -> None:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df['err_low'] = df['coef'] - df['ci_low']
    df['err_high'] = df['ci_high'] - df['coef']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    models = df['model'].unique()
    x_pos = np.array([0, 1])
    colors = ['#2E86AB', '#A23B72']

    for i, model in enumerate(models):
        md = df[df['model'] == model]
        errs = [[md['err_low'].values[0]], [md['err_high'].values[0]]]
        ax.errorbar(
            x=x_pos[i], y=md['coef'].values[0], yerr=errs,
            fmt='o', capsize=5, color=colors[i], markersize=8,
            linewidth=2.5, label=model,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    ax.set_title(
        'Effect of Castle Doctrine on Log Homicide Rate',
        fontsize=14, fontweight='bold', pad=15,
    )
    ax.set_ylabel('Coefficient Estimate', fontsize=12)
    _style(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, f'{out_stem}.png'), format='png', bbox_inches='tight')
    plt.close(fig)


def plot_event_study_cv(csv_path: str, out_stem: str) -> None:
    df = pd.read_csv(csv_path)
    x_pos = np.arange(len(df))
    errs = [df['coef'] - df['2.5 %'], df['97.5 %'] - df['coef']]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.errorbar(
        x=x_pos, y=df['coef'], yerr=errs,
        fmt='o', capsize=5, color='#2E86AB', markersize=8, linewidth=2.5,
        label='DML DiD (tuned)',
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['index'].values, fontsize=11)
    ax.set_title(
        'Event Study: Dynamic Treatment Effects (Tuned)',
        fontsize=14, fontweight='bold', pad=15,
    )
    ax.set_xlabel('Event Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.95)
    _style(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, f'{out_stem}.png'), format='png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_model_comparison(
        os.path.join(OUTPUT, 'model_comparison_results.csv'),
        'model_comparison',
    )
    plot_staggered_cv(
        os.path.join(OUTPUT, 'model_comparison_results_staggered_cv.csv'),
        'model_comparison_staggered_cv',
    )
    plot_event_study_cv(
        os.path.join(OUTPUT, 'event_study_aggregation_cv.csv'),
        'event_study_aggregation_cv',
    )
    print('Replotted: model_comparison, model_comparison_staggered_cv, event_study_aggregation_cv')
