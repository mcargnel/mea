# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic thesis: **Double Machine Learning for Difference-in-Differences** (Universidad de Buenos Aires). Combines econometric causal inference (DiD) with machine learning (DML) through Monte Carlo simulations and two empirical applications.

Written in Spanish (advisor communications) and English (thesis text).

## Language and Environment

- Python 3.11 managed via `uv`, virtual environment in `.venv/`
- LaTeX for thesis manuscript (in `book/`)
- Requires `pandoc` for LaTeX→Markdown→Word conversion
- No automated tests, linting, or CI/CD

## Running Simulations and Scripts

All Python scripts are run via `uv run` (no need to activate the venv manually):

```bash
# Monte Carlo simulations (main simulation engine)
uv run src/monte_carlo_sim.py

# With flags: -s scenarios, -n iterations, -m preset, -u sample sizes, -o output dir
uv run src/monte_carlo_sim.py -s 1 2 3 -n 300 -m light -u 500 -o results/500_light

# Batch runs (sequential, multiple configs)
bash run_batch.sh

# Aggregate results from summary CSVs
uv run src/aggregate_results.py results/500_light/summary_n300.csv -o results/aggregated.csv

# Generate thesis plots and markdown tables from results
uv run src/generate_thesis_outputs.py -d results/500_light -i 300

# Empirical applications
uv run src/chapter_4.py    # Castle doctrine (castle.dta)
uv run src/chapter_4b.py   # CDL / fracking (zc_level.dta)

# Chapter 2 visualizations
uv run src/chapter_2.py

# Convert LaTeX chapters to Markdown/Word
uv run book/convert_to_md.py
```

## Key Dependencies

`doubleml`, `linearmodels` (PanelOLS), `scikit-learn`, `lightgbm`, `numpy`, `pandas`, `matplotlib`, `joblib`

## Architecture

### Simulation Pipeline

The simulation workflow is a 3-step pipeline:

1. **`monte_carlo_sim.py`** generates data, runs estimators, and saves raw results (`.parquet`) + summary (`.csv`) to `results/<config>/`.
2. **`aggregate_results.py`** combines summary CSVs across configs, applying weighted averaging for staggered scenarios.
3. **`generate_thesis_outputs.py`** reads results and produces markdown tables + boxplot figures for the thesis.

`run_batch.sh` / `run_batch_2.sh` orchestrate step 1 across multiple configs (varying sample sizes and ML presets).

### Core Simulation Design (`monte_carlo_sim.py`)

- **12 scenarios** (dict `SCENARIOS`) vary along 3 axes: confounding complexity (simple/mid/complex), propensity score strength (0.25/0.5/1.0), and panel structure (2-period, 6-period, staggered). Scenarios 10-12 are constant-TE variants.
- **3 estimators**: TWFE (always), DML-Chang (non-staggered via `DoubleMLIRM` with `score='ATTE'`), DML-Multi (staggered via `DoubleMLDIDMulti`).
- **3 LightGBM presets** (`ML_PRESETS`): light, default, heavy — control model complexity.
- Parallelization via `joblib.Parallel` at the iteration level; each iteration is a standalone function `_run_single_iteration()` for pickling compatibility.
- Sample size variation: generates data at `max(n_units_list)` then subsamples down.

### Data Generation (`data_generation_clean.py`)

Single entry point: `mldid_staggered_did()`. Generates synthetic panel data with:
- Covariates X1-X4 (X1, X2 continuous; X3, X4 binary; X4 is pure noise)
- Confounded treatment assignment via multinomial logit on confounding index `c(X_i)`
- Potential outcomes Y(0), Y(1) with oracle columns `y0`, `y1` for true ATT computation
- Returns long-format DataFrame with columns: `id, G, X1-X4, Y, y0, y1, period, treat, delta_e, cluster`

### Empirical Applications (`chapter_4.py`, `chapter_4b.py`)

Each loads a Stata `.dta` file from `input/`, runs TWFE + DML estimators on real data, and outputs results/plots.

### Thesis (`book/`)

- `main.tex` is the master LaTeX document; chapters in `book/chapters/`.
- `convert_to_md.py` preprocesses LaTeX (strips TikZ, fixes paths), runs pandoc, post-processes markdown, then builds combined `full_thesis.md` → `full_thesis.docx`.
- Markdown outputs live alongside `.tex` files in `book/chapters/` (not a separate directory).

### Results Structure

`results/` contains subdirectories named `{n_units}_{ml_preset}/` (e.g., `500_light/`, `2500_default/`), each containing `all_results_n{iters}.parquet` and `summary_n{iters}.csv`.
