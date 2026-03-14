# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic thesis: **Double Machine Learning for Difference-in-Differences** (Universidad de Buenos Aires). Combines econometric causal inference (DiD) with machine learning (DML) through Monte Carlo simulations and two empirical applications.

Written in Spanish (advisor communications) and English (thesis text).

## Language and Environment

- Python 3.11 managed via `uv`, virtual environment in `.venv/`
- No `pyproject.toml` or `requirements.txt` at root; dependencies installed directly in `.venv/`
- LaTeX for thesis manuscript (in `book/`)
- Requires `pandoc` for LaTeX→Markdown→Word conversion
- No automated tests, linting, or CI/CD

## Running Simulations and Scripts

All Python scripts are run via `uv run` (no need to activate the venv manually):

```bash
# Monte Carlo simulations (main simulation engine)
uv run src/simulation/monte_carlo_sim.py
uv run src/simulation/monte_carlo_sim.py -s 1 2 3 -n 300 -m light -u 500 -o results/500_light

# Batch runs (sequential, multiple configs across sample sizes × ML presets)
bash run_batch.sh      # 9 jobs: {500,2500,10000} × {light,default,heavy}
bash run_batch_2.sh    # Specialized: scenarios 8,12 with v_heavy preset

# Aggregate results from summary CSVs
uv run src/postprocessing/aggregate_results.py results/500_light/summary_n300.csv -o results/aggregated.csv

# Generate thesis plots and markdown tables from results
uv run src/postprocessing/generate_thesis_outputs.py -d results/500_light -i 300

# Update simulation tables in 04_simulations.md (combined RMSE + detail tables)
uv run src/postprocessing/update_sim_tables.py                # light preset (default)
uv run src/postprocessing/update_sim_tables.py --preset heavy # different preset

# Compare ML presets across sample sizes (tables + plots)
uv run src/postprocessing/compare_presets.py                              # light/default/heavy
uv run src/postprocessing/compare_presets.py -o results/preset_comparison # custom output dir
uv run src/postprocessing/compare_presets.py --presets light heavy         # subset of presets

# Patch utilities (fix true ATT values in saved results)
uv run src/patches/fix_staggered_att.py                          # Fix TWFE true_att in staggered scenarios
uv run src/patches/patch_eventstudy_att.py --dirs 500_light      # Fix DML-Multi to eventstudy-weighted ATT

# Empirical applications
uv run src/empirical/chapter_4.py    # Castle doctrine (castle.dta)
uv run src/empirical/chapter_4b.py   # CDL / fracking (zc_level.dta)

# Chapter 2 visualizations
uv run src/empirical/chapter_2.py

# Convert LaTeX chapters to Markdown/Word
uv run src/postprocessing/convert_to_md.py          # Both Markdown + DOCX
uv run src/postprocessing/convert_to_md.py --md     # Only Markdown
uv run src/postprocessing/convert_to_md.py --docx   # Only DOCX from existing Markdown
```

## Key Dependencies

`doubleml`, `linearmodels` (PanelOLS), `scikit-learn`, `lightgbm`, `numpy`, `pandas`, `matplotlib`, `joblib`

## Architecture

### Simulation Pipeline

The simulation workflow is a 3-step pipeline:

1. **`monte_carlo_sim.py`** generates data, runs estimators, and saves raw results (`.parquet`) + summary (`.csv`) to `results/<config>/`.
2. **`aggregate_results.py`** combines summary CSVs across configs, applying weighted averaging for staggered scenarios (weight = `n_iters × n_units`).
3. **`generate_thesis_outputs.py`** reads results and produces a markdown table (`markdown_table.md`) + a 4×3 master boxplot (`master_boxplot.png`).

`run_batch.sh` / `run_batch_2.sh` orchestrate step 1 across multiple configs.

### Core Simulation Design (`monte_carlo_sim.py`)

**12 scenarios** (dict `SCENARIOS`) follow a triplet pattern grouped by confounding/outcome complexity:

| Group | Scenarios | Confounding | PS Strength | Within-group structure |
|-------|-----------|-------------|-------------|----------------------|
| Simple | 1, 2, 3 | simple | 0.25 | 2-period, 6-period, 6-period staggered |
| Mid | 4, 5, 6 | mid | 0.5 | 2-period, 6-period, 6-period staggered |
| Complex | 7, 8, 9 | complex | 1.0 | 2-period, 6-period, 6-period staggered |
| Constant TE | 10, 11, 12 | simple/mid/complex | 0.25/0.5/1.0 | All 6-period non-staggered, `te_form='constant'` |

**3 estimators** (values in output `model` column):
- **`TWFE`** — Always run. Two-Way Fixed Effects via `PanelOLS`.
- **`DML-Chang`** — Non-staggered only (scenarios 1,2,4,5,7,8,10-12). Uses `DoubleMLIRM` with `score='ATTE'` on first-differenced data.
- **`DML-Multi`** — Staggered only (scenarios 3,6,9). Uses `DoubleMLDIDMulti` with eventstudy aggregation.

**4 LightGBM presets** (`ML_PRESETS`): `light` (50 trees, depth 2), `default` (200, depth 2), `heavy` (1000, depth 3), `v_heavy` (4000, depth 4).

**Seed strategy**: `seed = 1000 × scenario_id + iteration` — deterministic per scenario/iteration pair.

**Parallelization**: `joblib.Parallel` at the iteration level; each iteration is a standalone function `_run_single_iteration()` for pickling compatibility.

### Data Generation (`data_generation_clean.py`)

Single entry point: `mldid_staggered_did()`. Generates synthetic panel data with:
- Covariates X1-X4 (X1, X2 continuous; X3, X4 binary; X4 is pure noise)
- Confounded treatment assignment via multinomial logit on confounding index `c(X_i)`
- Potential outcomes Y(0), Y(1) with oracle columns `y0`, `y1` for true ATT computation
- Returns long-format DataFrame with columns: `id, G, X1-X4, Y, y0, y1, period, treat, delta_e, cluster`

Confounding complexity controls `c(X_i)`: simple = linear, mid = interaction terms, complex = nonlinear (squares, sin).

### Empirical Applications (`chapter_4.py`, `chapter_4b.py`)

Each loads a Stata `.dta` file from `input/`, runs TWFE + DML estimators on real data, and outputs results/plots.

### Thesis (`book/`)

- `main.tex` is the master LaTeX document; chapters in `book/chapters/` (`01_introduction.tex`, `02_did.tex`, `03_dml.tex`, `05_applications.tex`, `06_conclusion.tex`, `appendix.tex`). Chapter 04 (Simulations) exists only as markdown (`book/markdown/04_simulations.md`), hence the numbering gap. This file contains auto-updated tables between HTML comment markers: `<!-- TABLE:COMBINED -->` for the main RMSE table and `<!-- TABLE:500 -->`, `<!-- TABLE:2500 -->`, `<!-- TABLE:10000 -->` for per-sample-size detail tables (Bias, RMSE, Coverage) in the Appendix subsection. Run `update_sim_tables.py` to refresh them from CSV data.
- `src/postprocessing/convert_to_md.py` preprocesses LaTeX (strips TikZ, fixes paths), runs pandoc, post-processes markdown, then builds combined `book/full_thesis.md` → `book/cargnel_tesis.docx`. Per-chapter markdown goes to `book/markdown/`.
- `book/images/logos/` contains university logos; `book/images/` contains content images.
- `PTFM/` contains thesis proposal documents (PDF/DOCX).

### Results Structure

`results/` contains subdirectories named `{n_units}_{ml_preset}/` (e.g., `500_light/`, `2500_default/`), each containing:
- `all_results_n{iters}.parquet` — one row per estimator × subsample size × iteration
- `summary_n{iters}.csv` — one row per scenario × model, with aggregated stats

## Academic Writing Guidelines

- **Language**: English. The author is not a native speaker, so keep it simple and clear.
- **Voice**: Do not use "we"; this is an individual academic production.
- **Tone**: Academic but accessible; avoid jargon when simpler language works. Avoid hyperbolic or charged language (e.g., "catastrophic", "dramatic", "striking", "dominates", "excels", "failure"); prefer neutral descriptors that let the numbers speak for themselves.
- **Target audience**: Applied economists and practitioners who may not be ML experts.
- **Length**: Be thorough but not verbose; provide necessary detail without redundancy.
- **Citations**: Reference sources using `\cite{}`. Do not suggest changes to citation keys or reference management.
- **Formatting**: Do not use bullet points, numbered lists, em-dashes, or bold text in thesis prose.
- **Transitions**: Ensure smooth logical flow; preserve original intent when improving transitions.
- Do not make unsolicited changes to thesis text.
- Do not create summary markdown files unless explicitly asked.
- Do not use relative line numbers (use absolute).

## When Reviewing/Editing Content:
- Flag unclear passages and suggest improvements to flow and logic
- Point out redundancy; check consistency across sections
- Verify math is correct, clear, and notation is consistent
- Ensure citations are properly formatted
- Keep sentences at reasonable length (avoid run-ons)

## Chapter Structure:
- Each chapter starts with a brief introduction outlining goals and structure.
- Each section/subsection has a clear purpose and flows logically from the previous content.
- Each section/subsection is self-contained, providing necessary context without relying on other parts.
- Transitions between sections should be smooth and not repetitive.