# Double Machine Learning for Difference in Difference: Fundamentals and Applications

Machine Learning (ML) models have traditionally been associated with prediction tasks due to their flexibility, while social scientists have typically relied on simpler, often linear, regressions for assessing causality. However, a novel framework named Double Machine Learning (DML) has emerged, providing a way to leverage the predictive performance of these complex methods for robust causal estimation. This thesis examines the fundamentals and applications of double machine learning for two distinct popular Difference-in-Differences (DiD) settings.

## Empirical scripts

`src/empirical/chapter_4b.py` runs TWFE + DML-DiD with fixed RF hyperparameters; `src/empirical/chapter_4b_cv.py` does the same but Optuna-tunes the outcome regressions (`ml_g0`/`ml_g1`).
