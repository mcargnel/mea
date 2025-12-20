"""Chapter 2: Generate and visualize simulated DiD data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gen_data() -> tuple[pd.DataFrame, str]:
    """Generate synthetic difference-in-differences data.
    
    Returns:
        Tuple of (DataFrame with simulated data, treatment start year string).
    """
    np.random.seed(42)

    n_samples = 50
    baseline = 1000
    trend_slope = 20
    cycle_strength = 150
    cycle_period = 10
    noise_level = 25

    treatment_effect = 300
    treatment_start_year = '2000-01-01'

    dates = pd.date_range(start='1980-01-01', periods=n_samples, freq='AS')
    time = np.arange(n_samples)

    trend = baseline + time * trend_slope
    cycle = cycle_strength * np.sin(2 * np.pi * time / cycle_period)

    noise_control = np.random.normal(loc=0, scale=noise_level, size=n_samples)
    y_control = trend + cycle + noise_control

    intervention = (dates > pd.to_datetime(treatment_start_year)).astype(int)

    noise_treated = np.random.normal(loc=0, scale=noise_level, size=n_samples)

    y_treated = trend + cycle + noise_treated + (intervention * treatment_effect)
    y_treated_not = (
        trend * 0.5
        + cycle_strength * np.sin(2.6 * np.pi * time / cycle_period)
        + noise_treated * 1.5
        + (intervention * treatment_effect) * 0.05
        + np.random.uniform(-200, 200, n_samples)
    )

    df_yearly = pd.DataFrame({
        'date': dates,
        'y_control': y_control,
        'y_treated': y_treated,
        'y_treated_not': y_treated_not
    })
    df_yearly = df_yearly.set_index('date')
    return df_yearly, treatment_start_year


def plot_validated_data(
    df_yearly: pd.DataFrame,
    treatment_start_year: str
) -> tuple[plt.Figure, plt.Axes]:
    """Plot simulated DiD data with parallel trends.
    
    Args:
        df_yearly: DataFrame with simulated data.
        treatment_start_year: Treatment start date string.
    
    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Define professional colors
    color_control = '#2E86AB'  # Blue
    color_treated = '#A23B72'  # Purple
    color_intervention = '#F18F01'  # Orange
    
    # Plot both series with enhanced styling
    ax.plot(
        df_yearly.index,
        df_yearly['y_control'],
        marker='o',
        markersize=5,
        linewidth=2.5,
        label='Control Unit',
        color=color_control,
        alpha=0.85
    )
    ax.plot(
        df_yearly.index,
        df_yearly['y_treated'],
        marker='s',
        markersize=5,
        linewidth=2.5,
        label='Treatment Unit',
        color=color_treated,
        alpha=0.85
    )

    # Add intervention line
    intervention_date = pd.to_datetime(treatment_start_year)
    ax.axvline(
        intervention_date,
        color=color_intervention,
        linestyle='--',
        linewidth=2.5,
        alpha=0.8,
        label=f'Policy Implementation ({treatment_start_year[:4]})'
    )

    # Formatting
    ax.set_title(
        "Simulated Difference-in-Differences Design",
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax.set_ylabel("Outcome Variable", fontsize=12, fontweight='bold')

    # Legend styling
    ax.legend(
        loc='upper left',
        fontsize=11,
        framealpha=0.95,
        shadow=False
    )

    # Minimalist styling - remove grid and excess spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    return fig, ax


def plot_non_validated_data(
    df_yearly: pd.DataFrame,
    treatment_start_year: str
) -> tuple[plt.Figure, plt.Axes]:
    """Plot simulated DiD data with violated parallel trends.
    
    Args:
        df_yearly: DataFrame with simulated data.
        treatment_start_year: Treatment start date string.
    
    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Define professional colors
    color_control = '#2E86AB'  # Blue
    color_treated = '#C1121F'  # Red (to indicate violation)
    color_intervention = '#F18F01'  # Orange
    
    # Plot both series with enhanced styling
    ax.plot(
        df_yearly.index,
        df_yearly['y_control'],
        marker='o',
        markersize=5,
        linewidth=2.5,
        label='Control Unit',
        color=color_control,
        alpha=0.85
    )
    ax.plot(
        df_yearly.index,
        df_yearly['y_treated_not'],
        marker='s',
        markersize=5,
        linewidth=2.5,
        label='Treatment Unit',
        color=color_treated,
        alpha=0.85
    )

    # Add intervention line
    intervention_date = pd.to_datetime(treatment_start_year)
    ax.axvline(
        intervention_date,
        color=color_intervention,
        linestyle='--',
        linewidth=2.5,
        alpha=0.8,
        label=f'Policy Implementation ({treatment_start_year[:4]})'
    )

    # Formatting
    ax.set_title(
        "Parallel Trends Assumption Violated",
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax.set_ylabel("Outcome Variable", fontsize=12, fontweight='bold')

    # Legend styling
    ax.legend(
        loc='upper left',
        fontsize=11,
        framealpha=0.95,
        shadow=False
    )

    # Minimalist styling - remove grid and excess spines
    ax.grid(False)

    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    return fig, ax


def main() -> None:
    """Generate and save simulated DiD figures."""
    output_dir = "/Users/mcargnel/Documents/mea/tesis/output"
    df_yearly, treatment_start_year = gen_data()

    fig, ax = plot_validated_data(df_yearly, treatment_start_year)
    fig.savefig(
        f"{output_dir}/fig_2_1_simulated_data.pdf",
        format='pdf',
        bbox_inches='tight'
    )

    fig, ax = plot_non_validated_data(df_yearly, treatment_start_year)
    fig.savefig(
        f"{output_dir}/fig_2_1_simulated_data_not_hold.pdf",
        format='pdf',
        bbox_inches='tight'
    )

    print("Figure saved to output/fig_2_1_simulated_data.pdf")
    print("Figure saved to output/fig_2_1_simulated_data_not_hold.pdf")


if __name__ == "__main__":
    main()