from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 6. MLDID Staggered DiD  (Hatamyar et al., 2023)
# ---------------------------------------------------------------------------


def mldid_staggered_did(
    n_units: int = 2500,
    n_periods: int = 4,
    staggered: bool = True,
    random_assignment: bool = True,
    noise_covariates: bool = False,
    time_dependent_covar: bool = True,
    confounding_complexity: str = "simple",
    outcome_complexity: str = "simple",
    het: str = "none",
    chi: int = 1,
    taumodel: int = 1,
    ps_strength: float = 0.5,
    te_form: str = "dynamic",
    seed: Optional[int] = None,
    verbose: bool = False,
    ) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    T = n_periods
    n = n_units

    # =========================================================================
    # STAGE 0: Parameters
    # =========================================================================
    #
    # time_idx: shared time trend for both Y(0) and Y(1), theta_t = t
    # te_e: dynamic treatment effect by event time, te_e[e] = e
    # te: baseline scalar added to the treatment effect (always 1.0)
    # gamG: propensity score parameter, gamG[g] = ps_strength * g / T
    # te_bet_X: covariate slope in Y(1), depends on 'het' parameter
    # =========================================================================
    time_idx = np.arange(1, T + 1, dtype=float)
    te_e = np.ones(T, dtype=float) if te_form == "constant" else time_idx.copy()
    te = 1.0
    gamG = ps_strength * np.arange(T + 1) / T

    # het controls whether covariates have a direct effect on Y(1)
    # beyond the heterogeneous treatment effect tau_i
    if het == "constant":
        te_bet_X = np.ones(T)        # X enters Y(1) with constant coefficient
    elif het == "calendar":
        te_bet_X = time_idx.copy()   # X enters Y(1) with growing coefficient
    else:
        te_bet_X = np.zeros(T)       # X does not directly affect Y(1)

    if verbose:
        print(f"bett: {bett}, betu: {betu}")
        print(f"te_e: {te_e}, te_bet_X: {te_bet_X}, het: {het}, te: {te}")

    # =========================================================================
    # STAGE 1: Covariates
    # =========================================================================
    # X1, X2: continuous, enter outcome/confounding models
    # X3: binary, enters outcome/confounding models
    # X4: binary, pure noise (never used in any model)
    # X5: optional time-varying covariate (sine wave + noise)
    # =========================================================================
    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    X3 = rng.binomial(1, 0.5, n).astype(float)
    X4 = rng.binomial(1, 0.5, n).astype(float)

    X5: Optional[np.ndarray] = None
    if time_dependent_covar:
        tp = np.linspace(0, 1, T)
        X5 = np.column_stack([
            np.sin(2 * np.pi * tp[t]) + rng.standard_normal(n)
            for t in range(T)
        ])

    noise_mat: Optional[np.ndarray] = None
    if noise_covariates:
        noise_mat = np.hstack([
            rng.standard_normal((n, 50)),
            rng.binomial(1, 0.5, (n, 50)).astype(float),
        ])

    # =========================================================================
    # STAGE 2: Treatment assignment
    # =========================================================================
    # G_i in {0, 1, ..., T}: treatment cohort (0 = never-treated)
    # Random: Pr(G=g) = 1/(T+1) for all g
    # Confounded: multinomial logit, Pr(G=g|X) ∝ exp(c_model * gamG[g])
    #   c_model complexity controls how hard the propensity score is to estimate:
    #   simple: linear additive
    #   mid:    adds an interaction term
    #   complex: nonlinear with thresholds and sine
    # =========================================================================
    if confounding_complexity == "simple":
        c_model = X1 + X2 + X3
    elif confounding_complexity == "mid":
        c_model = X1 + X2 + X3 + X2 * X3
    elif confounding_complexity == "complex":
        c_model = X1**2 + X2 * X3 + (X2 > 0).astype(float) * X1 + np.sin(X1 + X2)
    else:
        raise ValueError(f"confounding_complexity must be 'simple', 'mid', or 'complex', got '{confounding_complexity}'")


    if random_assignment:
        pr = np.full((n, T + 1), 1.0 / (T + 1))
    else:
        logits = np.outer(c_model, gamG)
        logits -= logits.max(axis=1, keepdims=True)   # numerical stability
        exp_l = np.exp(logits)
        pr = exp_l / exp_l.sum(axis=1, keepdims=True)

    G = np.array([rng.choice(T + 1, p=pr[i]) for i in range(n)])

    # Non-staggered: collapse all treated units into a single cohort (period 2)
    if not staggered:
        G[G > 0] = 2

    if verbose:
        print(f"staggered: {staggered}, random_assignment: {random_assignment}")

    # =========================================================================
    # STAGE 3: Fixed effects and covariate model
    # =========================================================================
    # Xmodel: the covariate(s) entering the outcome equation
    #   chi=1 → X1 only; chi≠1 → X1 + X2 + X3
    # Fixed effects create level differences between groups:
    #   Treated: alpha_i ~ N(G_i, 1) — later cohorts have higher intercepts
    #   Untreated: alpha_i ~ N(0, 1)
    # A good DiD estimator should difference these out.
    # =========================================================================
    treated_mask = G > 0
    Gt = G[treated_mask]
    nt = int(treated_mask.sum())
    nu = n - nt

    Xmodel = X1.copy() if chi == 1 else X1 + X2 + X3 
    Xt = Xmodel[treated_mask]
    Xu = Xmodel[~treated_mask]

    Ct = rng.standard_normal(nt) + Gt.astype(float)
    Cu = rng.standard_normal(nu)

    # =========================================================================
    # STAGE 4: Potential outcomes
    # =========================================================================
    # Y0 (untreated potential outcome):
    #   outcome_complexity controls how covariates enter Y(0), with a
    #   time-varying coefficient beta_t = (t+1)/T so the effect grows over time.
    #   This creates parallel trends conditional on X but not unconditionally:
    #   simple:  Y0 = theta_t + alpha_i + eps  (no covariate effect)
    #   mid:     Y0 = theta_t + alpha_i + X1 * (t+1)/T + eps  (linear, time-varying)
    #   complex: Y0 = theta_t + alpha_i + [sin(X1) + X2^2 + X1*X3] * (t+1)/T + eps
    #
    # tau (treatment effect heterogeneity):
    #   taumodel=1 → tau_i = X1_i (linear in X1)
    #   taumodel=2 → tau_i = (X2_i + X3_i)^2 (nonlinear)
    #   This is what ML-based estimators aim to capture.
    #
    # Y1 (treated potential outcome):
    #   Y1_{it} = theta_t + alpha_i + X_i * te_bet_X[t]
    #             + (1[G_i <= t] * te_e[event_time] + te) * tau_i + eps_{it}
    #   The treatment effect has two parts:
    #     - A dynamic component 1[G_i<=t]*te_e[e] that grows with event time
    #     - A constant baseline scalar te=1
    #   Both are multiplied by the unit-specific tau_i.
    # =========================================================================
    # Compute covariate contribution to Y(0) based on outcome_complexity
    if outcome_complexity == "simple":
        # No covariate effect on Y(0)
        y0_covar_t = np.zeros(nt)
        y0_covar_u = np.zeros(nu)
    elif outcome_complexity == "mid":
        # Linear covariate effect: X1 * beta (beta=1)
        y0_covar_t = X1[treated_mask]
        y0_covar_u = X1[~treated_mask]
    elif outcome_complexity == "complex":
        # Nonlinear: sin(X1) + X2^2 + X1*X3
        y0_covar_t = (np.sin(X1[treated_mask]) + X2[treated_mask]**2
                      + X1[treated_mask] * X3[treated_mask])
        y0_covar_u = (np.sin(X1[~treated_mask]) + X2[~treated_mask]**2
                      + X1[~treated_mask] * X3[~treated_mask])
    else:
        raise ValueError(f"outcome_complexity must be 'simple', 'mid', or 'complex', got '{outcome_complexity}'")

    Y0t = np.column_stack([
        time_idx[t] + Ct + y0_covar_t * (t + 1) / T + rng.standard_normal(nt)
        for t in range(T)
    ])

    tau = X1[treated_mask] if taumodel == 1 else (X2[treated_mask] + X3[treated_mask]) ** 2

    Y1t = np.column_stack([
        time_idx[t]
        + Ct                                                          # te_bet_ind is all ones
        + Xt * te_bet_X[t]
        + ((Gt <= t + 1).astype(float)
            * te_e[np.maximum(t + 1 - Gt + 1, 1).astype(int) - 1]
            + te) * tau
        + rng.standard_normal(nt)
        for t in range(T)
    ])

    # =========================================================================
    # STAGE 5: Observed outcomes
    # =========================================================================
    # For treated units: Y = Y(1) if currently treated (G_i <= t), else Y(0)
    # For untreated units: Y = Y(0) = Y(1) (no treatment ever)
    # =========================================================================
    active = np.column_stack([(Gt <= t + 1).astype(float) for t in range(T)])
    Yt = active * Y1t + (1 - active) * Y0t

    Y0u = np.column_stack([
        time_idx[t] + Cu + y0_covar_u * (t + 1) / T + rng.standard_normal(nu)
        for t in range(T)
    ])

    # -- Build long-format DataFrame --
    # Unit ordering: treated first, then untreated (matches R's rbind)
    treated_idx = np.where(treated_mask)[0]
    untreated_idx = np.where(~treated_mask)[0]
    unit_order = np.concatenate([treated_idx, untreated_idx])
    n_total = len(unit_order)

    ids = np.repeat(np.arange(1, n_total + 1), T)
    periods = np.tile(np.arange(1, T + 1), n_total)
    clusters = np.repeat(rng.integers(1, 51, size=n_total), T)

    data: Dict[str, np.ndarray] = {
        "id": ids,
        "G": np.repeat(G[unit_order], T),
        "X1": np.repeat(X1[unit_order], T),
        "X2": np.repeat(X2[unit_order], T),
        "X3": np.repeat(X3[unit_order], T),
        "X4": np.repeat(X4[unit_order], T),
        "Y": np.concatenate([Yt.ravel(), Y0u.ravel()]),
        "y0": np.concatenate([Y0t.ravel(), Y0u.ravel()]),
        "y1": np.concatenate([Y1t.ravel(), Y0u.ravel()]),
        "period": periods,
    }

    if time_dependent_covar and X5 is not None:
        data["X5"] = X5[unit_order].ravel()

    # Group probability columns for groups 2..T
    Gprobs = np.repeat(pr[unit_order, 2:], T, axis=0)
    for g in range(2, T + 1):
        data[f"G{g}"] = Gprobs[:, g - 2]

    if noise_covariates and noise_mat is not None:
        nl = np.repeat(noise_mat[unit_order], T, axis=0)
        for nc in range(100):
            data[f"noise_{nc + 1}"] = nl[:, nc]

    df = pd.DataFrame(data)
    df["treat"] = (df["G"] > 0).astype(int)
    df["delta_e"] = (df["period"] - df["G"] + 1).clip(lower=0)
    df["cluster"] = clusters

    # Filter out G=1 (no pre-treatment period available) and sort
    df = df[df["G"] != 1].reset_index(drop=True)
    df = df.sort_values(["id", "period"]).reset_index(drop=True)

    return df