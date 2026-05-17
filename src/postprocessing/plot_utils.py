"""Shared constants and helpers for postprocessing plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mapping from old (internal data) scenario IDs to new display numbers
# per Table 0 (Simulation scenario design).
OLD_TO_NEW: dict[int, int] = {
    10: 1,  11: 2,  12: 3,   # Constant TE
     1: 4,   4: 5,   7: 6,   # Two-period dynamic
     2: 7,   5: 8,   8: 9,   # Six-period non-staggered dynamic
     3: 10,  6: 11,  9: 12,  # Staggered dynamic
}

# Display order = ascending new scenario number.
SCENARIO_ORDER: list[int] = [10, 11, 12, 1, 4, 7, 2, 5, 8, 3, 6, 9]

MODEL_ORDER: list[str] = ["DML-Chang", "DML-Multi", "TWFE"]

SCENARIO_DESC: dict[int, str] = {
    10: "S1: Simple, const",
    11: "S2: Mid, const",
    12: "S3: Complex, const",
     1: "S4: Simple, 2-per",
     4: "S5: Mid, 2-per",
     7: "S6: Complex, 2-per",
     2: "S7: Simple, 6-per",
     5: "S8: Mid, 6-per",
     8: "S9: Complex, 6-per",
     3: "S10: Simple, stagg",
     6: "S11: Mid, stagg",
     9: "S12: Complex, stagg",
}

# (complexity, staggered, dynamics) tuples — used by master_summary.
SCENARIO_META: dict[int, tuple[str, str, str]] = {
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

COMPLEXITY_LABELS: dict[int, str] = {
    10: "Simple", 11: "Mid",  12: "Complex",
     1: "Simple",  4: "Mid",   7: "Complex",
     2: "Simple",  5: "Mid",   8: "Complex",
     3: "Simple",  6: "Mid",   9: "Complex",
}

MODEL_COLORS: dict[str, str] = {
    "TWFE": "#4878D0",
    "DML-Chang": "#EE854A",
    "DML-Multi": "#6ACC64",
}

PRESET_COLORS: dict[str, str] = {
    "light": "#4C72B0",
    "default": "#55A868",
    "heavy": "#C44E52",
    "v_heavy": "#8172B2",
}


def setup_plot_style(font_size: int = 12) -> None:
    """Apply consistent rcParams across postprocessing scripts."""
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 2,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    })


def boxplot_props(median_color: str = "red", flier_size: int = 3,
                  median_linewidth: float = 2.0) -> dict:
    """Return kwargs for matplotlib boxplot to get the project's house style."""
    return dict(
        capprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        flierprops=dict(marker="o", markerfacecolor="gray",
                        markersize=flier_size, alpha=0.5 if flier_size >= 3 else 0.3),
        medianprops=dict(color=median_color, linewidth=median_linewidth),
    )


def grouped_bar_plot(
    df: pd.DataFrame,
    presets: list[str],
    value_fn,
    *,
    ylabel: str,
    title: str,
    output_path,
    ylim: tuple[float, float] | None = None,
    hline: tuple[float, str] | None = None,
) -> None:
    """Render a grouped bar chart over (scenario, model) labels, one bar per preset.

    `value_fn(row)` extracts the value to plot from a single matched row (may
    return NaN). `df` must already be filtered to a single n_units.
    """
    labels: list[tuple[int, str]] = []
    for scen in SCENARIO_ORDER:
        scen_sub = df[df["scenario"] == scen]
        models_here = [m for m in MODEL_ORDER if m in scen_sub["model"].values]
        for model in models_here:
            labels.append((scen, model))

    n_groups = len(labels)
    n_presets = len(presets)
    bar_width = 0.8 / n_presets
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(16, 6))
    for j, preset in enumerate(presets):
        vals: list[float] = []
        for scen, model in labels:
            cell = df[(df["scenario"] == scen) & (df["model"] == model) & (df["preset"] == preset)]
            vals.append(value_fn(cell.iloc[0]) if not cell.empty else np.nan)
        offset = (j - (n_presets - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width, label=preset,
               color=PRESET_COLORS.get(preset, "#999999"),
               edgecolor="white", linewidth=0.5)

    if hline is not None:
        y, lbl = hline
        ax.axhline(y, color="black", linestyle="--", linewidth=1, alpha=0.7, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{OLD_TO_NEW[s]}\n{m}" for s, m in labels], ha="center")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(title="ML Preset", loc="lower left" if hline else "best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
