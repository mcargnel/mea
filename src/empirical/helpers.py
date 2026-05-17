"""Shared helpers for empirical scripts (fracking_did, castle_doctrine_did)."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Errorbar palette used in both empirical chapters.
MODEL_PALETTE: list[str] = ["#2E86AB", "#A23B72"]


def style_empirical_axes(ax: plt.Axes) -> None:  # type: ignore[name-defined]
    """Minimal axis styling shared across empirical figures."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", which="major", labelsize=10)


def lgbm_param_space(trial) -> dict:
    """Optuna search space for LightGBM (used for both ml_g and ml_m in chapter 4)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 31),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "verbose": -1,
        "random_state": 42,
    }


def rf_param_space(trial) -> dict:
    """Optuna search space for Random Forest (used in chapter 4b)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
        "random_state": 42,
    }
