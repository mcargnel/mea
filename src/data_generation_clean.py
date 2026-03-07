from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data Generating Process for DiD Simulations
# Adapted from Hatamyar et al. (2023), with extensions for confounding
# complexity, outcome complexity, and variable propensity score strength.
# See Chapter 4 of the thesis for the full description of the DGP.
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
    """Generate simulated panel data for DiD Monte Carlo experiments.

    Generates n units observed over T periods, with treatment cohort
    assignment G_i in {0, 1, ..., T}. The DGP constructs potential outcomes:

        Y_it(0) = theta_t + alpha_i + h(X_i) * (t+1)/T + eps_it
        Y_it(1) = theta_t + alpha_i + X^model_i * beta^X_t
                   + (1[G_i <= t] * delta_e + 1) * tau_i + eps_it

    where theta_t = t, alpha_i is a unit fixed effect, h(X_i) depends on
    outcome_complexity, delta_e depends on te_form, and tau_i depends on
    taumodel. See Chapter 4 for full notation.

    Args:
        n_units: Number of units (n).
        n_periods: Number of time periods (T).
        staggered: If True, treatment cohorts span {2, ..., T}.
            If False, all treated units are assigned to cohort 2.
        random_assignment: If True, Pr(G=g) = 1/(T+1) for all g.
            If False, treatment assignment depends on covariates
            via a multinomial logit model.
        noise_covariates: If True, add 100 irrelevant noise covariates.
        time_dependent_covar: If True, generate time-varying covariate X5.
        confounding_complexity: Controls c(X_i), the confounding index.
            'simple': X1 + X2 + X3
            'mid':    X1 + X2 + X3 + X2*X3
            'complex': X1^2 + X2*X3 + 1(X2>0)*X1 + sin(X1+X2)
        outcome_complexity: Controls h(X_i), the covariate effect on Y(0).
            'simple': 0 (no covariate effect)
            'mid':    X1
            'complex': sin(X1) + X2^2 + X1*X3
        het: Controls beta^X_t, the covariate coefficient in Y(1).
            'none': beta^X_t = 0 for all t
            'constant': beta^X_t = 1 for all t
            'calendar': beta^X_t = t
        chi: Controls X^model_i dimensionality.
            1: X^model = X1 only
            other: X^model = X1 + X2 + X3
        taumodel: Controls treatment effect heterogeneity tau_i.
            1: tau_i = X1_i (linear)
            2: tau_i = (X2_i + X3_i)^2 (nonlinear)
        ps_strength: Propensity score strength lambda. Higher values
            create stronger confounding (gamma_g = lambda * g / T).
        te_form: Treatment effect dynamics.
            'dynamic': delta_e = e (grows with event time)
            'constant': delta_e = 1 for all e
        seed: Random seed for reproducibility.
        verbose: If True, print diagnostic information.

    Returns:
        Long-format DataFrame with columns: id, G, X1-X4, Y, y0, y1,
        period, treat, delta_e, cluster, and optionally X5, G2-GT,
        noise_1-noise_100.
    """
    rng = np.random.default_rng(seed)
    T = n_periods
    n = n_units

    # =========================================================================
    # STAGE 0: Parameters
    # =========================================================================
    # theta_t: shared time trend, theta_t = t (for t = 1, ..., T)
    # delta_e: dynamic treatment effect by event time
    #     te_form="dynamic": delta_e = e (grows with event time)
    #     te_form="constant": delta_e = 1 for all e
    # te: constant baseline scalar in treatment effect (always 1.0)
    # gamma_g: propensity score parameter, gamma_g = lambda * g / T
    # beta^X_t: covariate coefficient in Y(1), controlled by 'het'
    # =========================================================================
    time_idx = np.arange(1, T + 1, dtype=float)  # theta_t = t
    te_e = np.ones(T, dtype=float) if te_form == "constant" else time_idx.copy()  # delta_e
    te = 1.0  # constant baseline in treatment effect
    gamG = ps_strength * np.arange(T + 1) / T  # gamma_g = lambda * g / T

    # beta^X_t: covariate coefficient in Y(1), depends on 'het' parameter
    if het == "constant":
        te_bet_X = np.ones(T)        # beta^X_t = 1 for all t
    elif het == "calendar":
        te_bet_X = time_idx.copy()   # beta^X_t = t (grows with calendar time)
    else:
        te_bet_X = np.zeros(T)       # beta^X_t = 0 (covariates do not enter Y(1))

    if verbose:
        print(f"te_e (delta_e): {te_e}, te_bet_X (beta^X_t): {te_bet_X}")
        print(f"het: {het}, te (baseline): {te}")

    # =========================================================================
    # STAGE 1: Covariates
    # =========================================================================
    # X1 ~ N(0,1), X2 ~ N(0,1): continuous, enter confounding/outcome models
    # X3 ~ Bernoulli(0.5): binary, enters confounding/outcome models
    # X4 ~ Bernoulli(0.5): binary, pure noise (never enters any model)
    # X5 (optional): time-varying, X5_it = sin(2*pi*t_p) + eps_it
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
    # G_i in {0, 1, ..., T}: treatment cohort (G_i = 0 means never-treated)
    #
    # Random assignment:
    #   Pr(G_i = g) = 1/(T+1) for all g
    #
    # Confounded assignment (multinomial logit):
    #   Pr(G_i = g | X_i) = exp(gamma_g * c(X_i)) / sum_g' exp(gamma_g' * c(X_i))
    #   where gamma_g = lambda * g / T
    #
    # Confounding index c(X_i) depends on confounding_complexity:
    #   simple:  c(X_i) = X1 + X2 + X3
    #   mid:     c(X_i) = X1 + X2 + X3 + X2*X3
    #   complex: c(X_i) = X1^2 + X2*X3 + 1(X2>0)*X1 + sin(X1+X2)
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
        logits = np.outer(c_model, gamG)  # n x (T+1), logit_ig = c(X_i) * gamma_g
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        exp_l = np.exp(logits)
        pr = exp_l / exp_l.sum(axis=1, keepdims=True)  # softmax -> Pr(G=g|X)

    G = np.array([rng.choice(T + 1, p=pr[i]) for i in range(n)])

    # Non-staggered: collapse all treated units into a single cohort (G=2)
    if not staggered:
        G[G > 0] = 2

    if verbose:
        print(f"staggered: {staggered}, random_assignment: {random_assignment}")

    # =========================================================================
    # STAGE 3: Fixed effects and covariate model index
    # =========================================================================
    # alpha_i (fixed effects):
    #   Treated units:   alpha_i ~ N(G_i, 1)
    #   Untreated units: alpha_i ~ N(0, 1)
    #
    # X^model_i (covariate index entering Y(1)):
    #   chi = 1:  X^model_i = X1_i
    #   chi != 1: X^model_i = X1_i + X2_i + X3_i
    # =========================================================================
    treated_mask = G > 0
    Gt = G[treated_mask]
    nt = int(treated_mask.sum())
    nu = n - nt

    Xmodel = X1.copy() if chi == 1 else X1 + X2 + X3  # X^model_i
    Xt = Xmodel[treated_mask]
    Xu = Xmodel[~treated_mask]

    alpha_t = rng.standard_normal(nt) + Gt.astype(float)  # alpha_i for treated
    alpha_u = rng.standard_normal(nu)                      # alpha_i for untreated

    # =========================================================================
    # STAGE 4: Potential outcomes
    # =========================================================================
    # Y_it(0) = theta_t + alpha_i + h(X_i) * (t+1)/T + eps_it
    #
    # h(X_i) depends on outcome_complexity:
    #   simple:  h(X_i) = 0
    #   mid:     h(X_i) = X1
    #   complex: h(X_i) = sin(X1) + X2^2 + X1*X3
    #
    # tau_i (treatment effect heterogeneity):
    #   taumodel=1: tau_i = X1_i (linear)
    #   taumodel=2: tau_i = (X2_i + X3_i)^2 (nonlinear)
    #
    # Y_it(1) = theta_t + alpha_i + X^model_i * beta^X_t
    #           + (1[G_i <= t] * delta_e + 1) * tau_i + eps_it
    # =========================================================================

    # Compute h(X_i): covariate contribution to Y(0)
    if outcome_complexity == "simple":
        h_Xi_t = np.zeros(nt)   # h(X_i) = 0
        h_Xi_u = np.zeros(nu)
    elif outcome_complexity == "mid":
        h_Xi_t = X1[treated_mask]   # h(X_i) = X1
        h_Xi_u = X1[~treated_mask]
    elif outcome_complexity == "complex":
        # h(X_i) = sin(X1) + X2^2 + X1*X3
        h_Xi_t = (np.sin(X1[treated_mask]) + X2[treated_mask]**2
                  + X1[treated_mask] * X3[treated_mask])
        h_Xi_u = (np.sin(X1[~treated_mask]) + X2[~treated_mask]**2
                  + X1[~treated_mask] * X3[~treated_mask])
    else:
        raise ValueError(f"outcome_complexity must be 'simple', 'mid', or 'complex', got '{outcome_complexity}'")

    # Y_it(0) for treated units (pre-treatment periods use this)
    Y0t = np.column_stack([
        time_idx[t] + alpha_t + h_Xi_t * (t + 1) / T + rng.standard_normal(nt)
        for t in range(T)
    ])

    # tau_i: unit-specific treatment effect modifier
    tau = X1[treated_mask] if taumodel == 1 else (X2[treated_mask] + X3[treated_mask]) ** 2

    # Y_it(1) for treated units
    Y1t = np.column_stack([
        time_idx[t]                                        # theta_t
        + alpha_t                                          # alpha_i
        + Xt * te_bet_X[t]                                 # X^model_i * beta^X_t
        + ((Gt <= t + 1).astype(float)                     # 1[G_i <= t]
            * te_e[np.maximum(t + 1 - Gt + 1, 1).astype(int) - 1]  # * delta_e
            + te) * tau                                    # + 1) * tau_i
        + rng.standard_normal(nt)                          # eps_it
        for t in range(T)
    ])

    # =========================================================================
    # STAGE 5: Observed outcomes
    # =========================================================================
    # Y_it = 1[G_i <= t] * Y_it(1) + (1 - 1[G_i <= t]) * Y_it(0)
    #
    # Treated units: Y = Y(1) in post-treatment (t >= G_i), Y(0) otherwise
    # Never-treated: Y = Y(0) always
    # =========================================================================
    active = np.column_stack([(Gt <= t + 1).astype(float) for t in range(T)])
    Yt = active * Y1t + (1 - active) * Y0t

    # Y_it(0) for never-treated units (observed outcome = Y(0) in all periods)
    Y0u = np.column_stack([
        time_idx[t] + alpha_u + h_Xi_u * (t + 1) / T + rng.standard_normal(nu)
        for t in range(T)
    ])

    # =========================================================================
    # STAGE 6: Assemble long-format panel DataFrame
    # =========================================================================
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

    # Group membership probability columns for cohorts 2..T
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

    # Exclude G=1: no pre-treatment period available for DiD identification
    df = df[df["G"] != 1].reset_index(drop=True)
    df = df.sort_values(["id", "period"]).reset_index(drop=True)

    return df