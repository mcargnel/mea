# Chapter 4: Simulations

This chapter presents a Monte Carlo simulation exercise designed to illustrate when DML estimators offer meaningful advantages over classical approaches in Difference-in-Differences settings, and when a simpler Two-Way Fixed Effects (TWFE) specification is preferred. The simulation procedure builds on the framework introduced by Hatamyar et al. (2023), which itself extended the staggered DiD design of Callaway and Sant'Anna (2021). The present work adapts and extends the Hatamyar et al. (2023) DGP in several directions: it introduces multiple tiers of confounding complexity (beyond the original linear specification), adds nonlinear outcome models that create conditional parallel trends, varies the propensity score strength parameter across scenarios, and includes noise covariates to test estimator robustness.

The chapter is organized as follows. First, the data generating process (DGP) is described, covering the construction of covariates, treatment assignment, potential outcomes, and observed outcomes. Second, the simulation scenarios are defined, varying the complexity of confounding and outcome models across three tiers. Third, the estimation procedures and Monte Carlo design are outlined.

## Data Generating Process

The simulation framework generates panel data for $n$ units observed over $T$ periods. Each unit $i$ is assigned to a treatment cohort $G_i \in \{0, 1, \ldots, T\}$, where $G_i = 0$ indicates a never-treated unit and $G_i = g > 0$ indicates that unit $i$ first receives treatment at period $g$.

The core of the DGP is defined by two potential outcome equations. The untreated potential outcome for unit $i$ at period $t$ is:

$$
Y_{it}(0) = \theta_t + \alpha_i + h(X_i) \cdot \frac{t+1}{T} + \varepsilon_{it}
$$

and the treated potential outcome is:

$$
Y_{it}(1) = \theta_t + \alpha_i + X^*_i \cdot \beta^X_t + \left(\mathbb{1}(G_i \leq t) \cdot \delta_e + 1\right) \cdot \tau_i + \varepsilon_{it}
$$

where $\theta_t = t$ is a shared time trend, $\alpha_i$ is a unit-specific fixed effect, $\varepsilon_{it} \sim N(0,1)$ is an idiosyncratic error, $h(X_i)$ is a function of covariates whose complexity varies across scenarios, $X^*_i$ is a covariate index entering $Y(1)$, $\beta^X_t$ is a time-varying coefficient on covariates in the treated outcome, $\delta_e$ is a dynamic treatment effect component that depends on event time $e = t - G_i + 1$, and $\tau_i$ is a unit-specific treatment effect modifier. The observed outcome switches between these two equations based on treatment status:

$$
Y_{it} = \mathbb{1}(G_i \leq t) \cdot Y_{it}(1) + (1 - \mathbb{1}(G_i \leq t)) \cdot Y_{it}(0)
$$

In Hatamyar et al. (2023), the untreated outcome did not depend on covariates (equivalent to $h(X_i) = 0$), confounding was limited to a single linear specification, and the propensity score strength was fixed. The present design extends each of these dimensions, as described in the subsections that follow.

### Covariates

Five types of covariates are generated for each unit. Two continuous covariates $X_1$ and $X_2$ are drawn independently from a standard normal distribution. A binary covariate $X_3$ is drawn from a Bernoulli distribution with probability 0.5. These three covariates mirror the covariate structure used in Hatamyar et al. (2023). An additional binary covariate $X_4$, also drawn from a Bernoulli(0.5) distribution, is included as a pure noise variable that does not enter any model equation. This covariate, absent from the original Hatamyar et al. (2023) design, tests whether estimation procedures are robust to irrelevant covariates.

An optional time-varying covariate $X_5$, not present in Hatamyar et al. (2023), can also be generated. For each period $t$, $X_{5,it}$ follows a sine wave pattern plus noise:

$$
X_{5,it} = \sin(2\pi t_p) + \varepsilon_{it}, \quad \varepsilon_{it} \sim N(0,1)
$$

where $t_p$ represents an evenly spaced grid over $[0,1]$. This covariate introduces temporal variation in the covariate structure, which is relevant for testing estimators in settings where conditioning variables evolve over time.

### Treatment Assignment

Treatment assignment is governed by a multinomial model that determines each unit's cohort membership $G_i$. Under random assignment, each cohort (including the never-treated group) is equally likely, so $\Pr(G_i = g) = 1/(T+1)$ for all $g \in \{0, 1, \ldots, T\}$.

Under confounded assignment, the probability of belonging to each cohort depends on the covariates through a multinomial logit model:

$$
\Pr(G_i = g \mid X_i) = \frac{\exp(\gamma_g \cdot c(X_i))}{\sum_{g'=0}^{T} \exp(\gamma_{g'} \cdot c(X_i))}
$$

where $\gamma_g = \lambda \cdot g / T$ is a cohort-specific parameter that scales with the propensity score strength $\lambda$, and $c(X_i)$ is a confounding index that depends on the covariates according to a specified complexity level. Three levels of confounding complexity are considered:

| Complexity | Confounding Index $c(X_i)$ |
|:-----------|:--------------------------|
| Simple     | $X_1 + X_2 + X_3$ |
| Mid        | $X_1 + X_2 + X_3 + X_2 \cdot X_3$ |
| Complex    | $X_1^2 + X_2 \cdot X_3 + \mathbb{1}(X_2 > 0) \cdot X_1 + \sin(X_1 + X_2)$ |

The simple specification creates a linear, additive confounding structure that is straightforward for any estimator to model. The mid specification introduces an interaction term $X_2 \cdot X_3$ and the complex specification involves squared terms, indicator functions, and trigonometric transformations, creating a confounding structure that is difficult for linear models to approximate. This is one of the differences when comparing against Callaway and Sant'Anna (2021) where covariates weren't included and Hatamyar (2023) where only the simple structure was included.

The propensity score strength parameter $\lambda$ controls how strongly the covariates influence treatment assignment. Higher values of $\lambda$ create greater imbalance between treated and control groups, making the confounding bias more severe when covariates are not properly adjusted for. In Hatamyar et al. (2023), this parameter was fixed; the present design varies $\lambda$ across scenarios (from 0.25 to 1.0) to examine how increasing confounding strength affects estimator performance.

In the non-staggered case, all treated units are collapsed into a single cohort that begins treatment at period 2, simplifying the design to one with a single treatment group and a clean pre-treatment period.

### Fixed Effects

Each unit receives an individual-specific fixed effect $\alpha_i$ that captures permanent level differences between units. For treated units, the fixed effect is drawn as $\alpha_i \sim N(G_i, 1)$, so units in later treatment cohorts tend to have higher baseline levels. For untreated units, $\alpha_i \sim N(0, 1)$. A correctly specified DiD estimator should difference out these fixed effects, but their correlation with treatment timing introduces an additional identification challenge.

### Potential Outcomes

The untreated potential outcome $Y_{it}(0)$, presented at the beginning of this section, includes a covariate effect $h(X_i)$ scaled by a time-varying coefficient $(t+1)/T$ that grows over the panel. This design creates parallel trends conditional on $X$ but not unconditionally, providing a setting where covariate adjustment is necessary for valid identification. This is a difference from Hatamyar et al. (2023), where covariates did not enter the untreated outcome equation (equivalent to the simple specification below). The introduction of outcome complexity tiers, particularly the mid and complex specifications, is one of the central extensions of the present simulation design.

Three levels of outcome complexity determine the function $h(X_i)$:

| Complexity | Outcome Function $h(X_i)$ |
|:-----------|:--------------------------|
| Simple     | $0$ (no covariate effect) |
| Mid        | $X_1$ |
| Complex    | $\sin(X_1) + X_2^2 + X_1 \cdot X_3$ |

The simple specification replicates the Hatamyar et al. (2023) setting where covariates do not affect $Y(0)$, so parallel trends hold unconditionally and TWFE is correctly specified. The mid and complex specifications are extensions introduced in the present work. The mid specification adds a linear covariate effect that grows over time, requiring covariate adjustment but within the capacity of parametric methods. The complex specification introduces nonlinearities in the outcome model such that linear controls cannot fully remove the covariate effect from $Y(0)$, creating a setting where flexible ML-based estimators should outperform linear methods.

The treatment effect is heterogeneous across units. The unit-specific treatment effect modifier $\tau_i$ is generated according to one of two specifications: $\tau_i = X_{1,i}$ (linear in $X_1$), or $\tau_i = (X_{2,i} + X_{3,i})^2$ (nonlinear). Hatamyar et al. (2023) used only the linear specification; the nonlinear alternative is an extension that assesses whether ML-based estimators can capture more complex treatment effect heterogeneity patterns.

The covariates entering the treated outcome equation are collected in a model index $X^*_i$. The parameter $\chi$ controls the dimensionality of this index: when $\chi = 1$, only $X_1$ enters; otherwise, the index is $X_1 + X_2 + X_3$. The dynamic treatment effect component $\delta_e$ depends on event time, with $\delta_e = e$ under dynamic effects and $\delta_e = 1$ under constant effects. The treatment effect for a treated unit in a post-treatment period thus has two parts: a dynamic component $\delta_e$ that grows with event time, and a constant baseline of 1. Both components are multiplied by the unit-specific $\tau_i$, so the actual treatment effect $(\delta_e + 1) \cdot \tau_i$ varies both across units and over time.

### Observed Outcomes

For treated units ($G_i > 0$), the observed outcome equals $Y_{it}(1)$ in post-treatment periods ($t \geq G_i$) and $Y_{it}(0)$ in pre-treatment periods ($t < G_i$). For never-treated units ($G_i = 0$), the observed outcome is always $Y_{it}(0)$.

The final dataset is assembled in long (panel) format with unit and period identifiers, treatment cohort indicators, observed and counterfactual outcomes, covariate values, and group membership probabilities. Units with $G_i = 1$ are excluded from the analysis because they have no pre-treatment period available, which is required for the DiD identification strategy.

## Simulation Setup

Many different combinations are possible with the data generating process described above. This exercise focuses on the following twelve scenarios, organized in three complexity tiers with a systematic variation in the number of periods and the treatment adoption structure:

| Scenario | Staggered | Periods | Complexity | Treatment Effect |
|--------|---------|-------|----------|----------------|
| 1        | No        | 2       | Simple     | Dynamic          |
| 2        | No        | 6       | Simple     | Dynamic          |
| 3        | Yes       | 6       | Simple     | Dynamic          |
| 4        | No        | 2       | Mid        | Dynamic          |
| 5        | No        | 6       | Mid        | Dynamic          |
| 6        | Yes       | 6       | Mid        | Dynamic          |
| 7        | No        | 2       | Complex    | Dynamic          |
| 8        | No        | 6       | Complex    | Dynamic          |
| 9        | Yes       | 6       | Complex    | Dynamic          |
| 10       | No        | 6       | Simple     | Constant         |
| 11       | No        | 6       | Mid        | Constant         |
| 12       | No        | 6       | Complex    | Constant         |

Scenarios 1 through 9 form a $3 \times 3$ grid that crosses three complexity tiers (simple, mid, complex) with three panel structures (2-period non-staggered, 6-period non-staggered, 6-period staggered). As complexity increases, so does the propensity score strength $\lambda$, making the confounding progressively harder to address. Scenarios 10 through 12 mirror scenarios 2, 5, and 8 but replace the dynamic treatment effect ($\delta_e = e$) with a constant one ($\delta_e = 1$), isolating the role of treatment effect dynamics.

For the non-staggered scenarios, TWFE is compared against the DML estimator of Chang (2020). In the simple tier, both estimators should perform similarly, since the confounding is linear, weak, and covariates do not enter $Y(0)$. As complexity increases, TWFE is expected to suffer from its inability to capture the nonlinear relationships in both the confounding and outcome models. The constant treatment effect scenarios (10 through 12) serve as an additional diagnostic: because the Chang (2020) estimator was designed for two-period settings where the treatment effect does not vary over time, it should perform well in these scenarios even with multiple periods, provided the complexity remains manageable.


For the staggered scenarios (3, 6, 9), TWFE is compared against the DML estimator of Callaway and Sant'Anna (2021). Standard TWFE is expected to produce biased estimates due to the negative weighting problem discussed in Chapter 2, with this bias growing as the complexity of confounding and the outcome model increases.

The target parameter is the Average Treatment Effect on the Treated (ATT), computed from the simulated counterfactual outcomes as $ATT = E[Y(1) - Y(0) \mid G > 0, \, t \geq G]$. To ensure robust results, each scenario is evaluated through 2,000 Monte Carlo replications with sample sizes of $n \in \{500, 2500, 10000\}$ units. Larger samples should benefit the DML estimators, which rely on machine learning models that improve with more training data.

Since each scenario involves different levels of complexity, the hyperparameters of the machine learning models were adapted accordingly. Given the large number of simulations and scenario variations, computational cost is a relevant concern, so LightGBM (Ke et al., 2017) was chosen as the machine learning model for both the outcome model $\hat{g}(X)$ and the propensity score model $\hat{m}(X)$. Three hyperparameter configurations were considered, varying the number of boosting rounds, the maximum depth of each tree, and the learning rate:

| Configuration | Number of Trees | Max Depth | Learning Rate |
|--------|---------|-------|----------|
| Light        | 50        | 2       | 0.1     |
| Default        | 200        | 2       | 0.1     |
| Heavy        | 1,000       | 3       | 0.05     |

These parameters jointly determine how much flexibility the model has for estimating the nuisance functions. More trees, greater depth, and a lower learning rate allow the model to capture increasingly complex relationships, but also increase the risk of overfitting. In simpler data generating scenarios, the heavy configuration may overfit the training data, leading to noisier nuisance function estimates and potentially worse performance than a more parsimonious specification. Cross-fitting is performed with 5 folds in all cases, ensuring that nuisance function predictions for each observation are generated by models trained on different data, as described in Chapter 3.

## Results


Callaway, Brantly, and Pedro H. C. Sant'Anna. 2021. "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics* 225 (2): 200--230. <https://doi.org/10.1016/j.jeconom.2020.12.001>.

Hatamyar, R., Chang, N., and Sant'Anna, P. H. C. 2023. "Double Machine Learning for Difference-in-Differences with Multiple Time Periods." Working Paper.

Ke, Guolin, Qi Meng, Thomas Finley, et al. 2017. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Proceedings of the 31st International Conference on Neural Information Processing Systems* (Red Hook, NY, USA), NIPS'17, 3149--57.