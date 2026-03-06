# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic thesis: **Double Machine Learning for Difference-in-Differences** (Universidad de Buenos Aires). Combines econometric causal inference (DiD) with machine learning (DML) through Monte Carlo simulations and two empirical applications.

## Language and Environment

- Python 3.11 managed via `uv`, virtual environment in `.venv/`
- LaTeX for thesis manuscript (in `book/`)
- Written in Spanish (advisor communications) and English (thesis text)

## Running Simulations and Scripts

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run Monte Carlo simulations (main simulation engine)
python src/monte_carlo_sim.py

# Run empirical applications
python src/chapter_4.py    # Castle doctrine / zoning case
python src/chapter_4b.py   # CDL / fracking case

# Generate Chapter 2 visualizations
python src/chapter_2.py

# Convert LaTeX chapters to Markdown/Word
python book/convert_to_md.py
```

There are no automated tests, linting, or CI/CD configured.

## Key Dependencies

`doubleml`, `linearmodels` (PanelOLS), `scikit-learn`, `lightgbm`, `numpy`, `pandas`, `matplotlib`, `joblib`

## Architecture

### Source Code (`src/`)

- **`monte_carlo_sim.py`** — Core simulation engine. Compares four estimators (TWFE, DML-Chang, DML-Staggered, Callaway-Sant'Anna) across 9 scenarios varying confounding complexity, propensity score strength, and staggered adoption. Uses parallel processing via joblib.
- **`data_generation_clean.py`** — Synthetic panel data generator (`mldid_staggered_did()`). Configurable parameters: number of units/periods, staggered treatment, confounding complexity (simple/mid/complex), propensity score strength, heterogeneous effects.
- **`chapter_2.py`** — Visualizations demonstrating DiD concepts (parallel trends, treatment effects).
- **`chapter_4.py`** / **`chapter_4b.py`** — Empirical applications using real Stata datasets from `input/` (castle.dta, zc_level.dta).

### Thesis (`book/`)

- `main.tex` is the master LaTeX document; chapters live in `book/chapters/`.
- Markdown versions for advisor sharing are in `book/markdown/`.
- `convert_to_md.py` handles LaTeX → Markdown → Word conversion via pandoc.

### Data (`input/`)

Real-world Stata `.dta` files used by the empirical application scripts. Results output to `results/`.
