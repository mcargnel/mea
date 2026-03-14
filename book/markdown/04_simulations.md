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

where $\gamma_g = \lambda \cdot g / T$ is a cohort-specific parameter that scales with the propensity score strength $\lambda$, and $c(X_i)$ is a confounding index that depends on the covariates according to a specified complexity level. Three levels of confounding complexity are considered. Under the simple specification, $c(X_i) = X_1 + X_2 + X_3$, creating a linear, additive confounding structure that is straightforward for any estimator to model. The mid specification adds an interaction term, $c(X_i) = X_1 + X_2 + X_3 + X_2 \cdot X_3$. The complex specification introduces squared terms, indicator functions, and trigonometric transformations, $c(X_i) = X_1^2 + X_2 \cdot X_3 + \mathbb{1}(X_2 > 0) \cdot X_1 + \sin(X_1 + X_2)$, creating a confounding structure that is difficult for linear models to approximate. This is one of the differences when comparing against Callaway and Sant'Anna (2021) where covariates weren't included and Hatamyar (2023) where only the simple structure was included.

The propensity score strength parameter $\lambda$ controls how strongly the covariates influence treatment assignment. Higher values of $\lambda$ create greater imbalance between treated and control groups, making the confounding bias more severe when covariates are not properly adjusted for. In Hatamyar et al. (2023), this parameter was fixed; the present design varies $\lambda$ across scenarios (from 0.25 to 1.0) to examine how increasing confounding strength affects estimator performance.

In the non-staggered case, all treated units are collapsed into a single cohort that begins treatment at period 2, simplifying the design to one with a single treatment group and a clean pre-treatment period.

### Fixed Effects

Each unit receives an individual-specific fixed effect $\alpha_i$ that captures permanent level differences between units. For treated units, the fixed effect is drawn as $\alpha_i \sim N(G_i, 1)$, so units in later treatment cohorts tend to have higher baseline levels. For untreated units, $\alpha_i \sim N(0, 1)$. A correctly specified DiD estimator should difference out these fixed effects, but their correlation with treatment timing introduces an additional identification challenge.

### Potential Outcomes

The untreated potential outcome $Y_{it}(0)$, presented at the beginning of this section, includes a covariate effect $h(X_i)$ scaled by a time-varying coefficient $(t+1)/T$ that grows over the panel. This design creates parallel trends conditional on $X$ but not unconditionally, providing a setting where covariate adjustment is necessary for valid identification. This is a difference from Hatamyar et al. (2023), where covariates did not enter the untreated outcome equation (equivalent to the simple specification below). The introduction of outcome complexity tiers, particularly the mid and complex specifications, is one of the central extensions of the present simulation design.

Three levels of outcome complexity determine the function $h(X_i)$. Under the simple specification, $h(X_i) = 0$, so covariates do not affect $Y(0)$, parallel trends hold unconditionally, and TWFE is correctly specified. This replicates the Hatamyar et al. (2023) setting. The mid and complex specifications are extensions introduced in the present work. The mid specification sets $h(X_i) = X_1$, adding a linear covariate effect that grows over time, requiring covariate adjustment but within the capacity of parametric methods. The complex specification sets $h(X_i) = \sin(X_1) + X_2^2 + X_1 \cdot X_3$, introducing nonlinearities in the outcome model such that linear controls cannot fully remove the covariate effect from $Y(0)$, creating a setting where flexible ML-based estimators should outperform linear methods.

The treatment effect is heterogeneous across units. The unit-specific treatment effect modifier $\tau_i$ is generated according to one of two specifications: $\tau_i = X_{1,i}$ (linear in $X_1$), or $\tau_i = (X_{2,i} + X_{3,i})^2$ (nonlinear). Hatamyar et al. (2023) used only the linear specification; the nonlinear alternative is an extension that assesses whether ML-based estimators can capture more complex treatment effect heterogeneity patterns.

The covariates entering the treated outcome equation are collected in a model index $X^*_i$. The parameter $\chi$ controls the dimensionality of this index: when $\chi = 1$, only $X_1$ enters; otherwise, the index is $X_1 + X_2 + X_3$. The dynamic treatment effect component $\delta_e$ depends on event time, with $\delta_e = e$ under dynamic effects and $\delta_e = 1$ under constant effects. The treatment effect for a treated unit in a post-treatment period thus has two parts: a dynamic component $\delta_e$ that grows with event time, and a constant baseline of 1. Both components are multiplied by the unit-specific $\tau_i$, so the actual treatment effect $(\delta_e + 1) \cdot \tau_i$ varies both across units and over time.

### Observed Outcomes

For treated units ($G_i > 0$), the observed outcome equals $Y_{it}(1)$ in post-treatment periods ($t \geq G_i$) and $Y_{it}(0)$ in pre-treatment periods ($t < G_i$). For never-treated units ($G_i = 0$), the observed outcome is always $Y_{it}(0)$.

The final dataset is assembled in long (panel) format with unit and period identifiers, treatment cohort indicators, observed and counterfactual outcomes, covariate values, and group membership probabilities. Units with $G_i = 1$ are excluded from the analysis because they have no pre-treatment period available, which is required for the DiD identification strategy.

## Simulation Setup

Many different combinations are possible with the data generating process described above. This exercise focuses on twelve scenarios, organized in three complexity tiers with a systematic variation in the number of periods and the treatment adoption structure.

Scenarios 1 through 9 form a $3 \times 3$ grid that crosses three complexity tiers (simple, mid, complex) with three panel structures (2-period non-staggered, 6-period non-staggered, 6-period staggered), all with dynamic treatment effects. Within each tier, the first scenario uses 2 periods without staggered adoption (Scenarios 1, 4, 7), the second uses 6 periods without staggered adoption (Scenarios 2, 5, 8), and the third uses 6 periods with staggered adoption (Scenarios 3, 6, 9). As complexity increases, so does the propensity score strength $\lambda$ (from 0.25 in the simple tier to 1.0 in the complex tier), making the confounding progressively harder to address. Scenarios 10 through 12 are 6-period non-staggered designs at the simple, mid, and complex complexity levels, respectively, but replace the dynamic treatment effect ($\delta_e = e$) with a constant one ($\delta_e = 1$), isolating the role of treatment effect dynamics.

For the non-staggered scenarios, TWFE is compared against the DML estimator of Chang (2020). In the simple tier, both estimators should perform similarly, since the confounding is linear, weak, and covariates do not enter $Y(0)$. As complexity increases, TWFE is expected to suffer from its inability to capture the nonlinear relationships in both the confounding and outcome models. The constant treatment effect scenarios (10 through 12) serve as an additional diagnostic: because the Chang (2020) estimator was designed for two-period settings where the treatment effect does not vary over time, it should perform well in these scenarios even with multiple periods, provided the complexity remains manageable.

For the staggered scenarios (3, 6, 9), TWFE is compared against the DML estimator of Callaway and Sant'Anna (2021). Standard TWFE is expected to produce biased estimates due to the negative weighting problem discussed in Chapter 2, with this bias growing as the complexity of confounding and the outcome model increases.

The target parameter is the Average Treatment Effect on the Treated (ATT), computed from the simulated counterfactual outcomes as $ATT = E[Y(1) - Y(0) \mid G > 0, \, t \geq G]$. To ensure robust results, each scenario is evaluated through 2,000 Monte Carlo replications with sample sizes of $n \in \{500, 2500, 10000\}$ for 2 and 6 periods. Larger samples should benefit the DML estimators, which rely on machine learning models that improve with more training data.

Since each scenario involves different levels of complexity, the hyperparameters of the machine learning models were adapted accordingly. Given the large number of simulations and scenario variations, computational cost is a relevant concern, so LightGBM (Ke et al., 2017) was chosen as the machine learning model for both the outcome model $\hat{g}(X)$ and the propensity score model $\hat{m}(X)$. Three hyperparameter configurations were considered. The light configuration uses 50 trees with maximum depth 2 and a learning rate of 0.1. The default configuration increases the number of trees to 200 while keeping the same depth and learning rate. The heavy configuration uses 1,000 trees with maximum depth 3 and a lower learning rate of 0.05. These parameters jointly determine how much flexibility the model has for estimating the nuisance functions. More trees, greater depth, and a lower learning rate allow the model to capture increasingly complex relationships, but also increase the risk of overfitting. In simpler data generating scenarios, the heavy configuration may overfit the training data, leading to noisier nuisance function estimates and potentially worse performance than a more parsimonious specification. Cross-fitting is performed with 5 folds in all cases, ensuring that nuisance function predictions for each observation are generated by models trained on different data, as described in Chapter 3.

## Results

The table below reports the root mean squared error (RMSE) for each scenario-estimator combination across all three sample sizes ($n = 500$, $n = 2{,}500$, and $n = 10{,}000$), using the light LightGBM configuration (50 trees, maximum depth 2). This parsimonious specification was selected as the baseline to minimize the risk of overfitting, which can inflate variance in smaller samples and obscure the comparison between estimators. RMSE combines bias and variance into a single accuracy measure: lower values indicate that the estimator's point estimates are, on average, closer to the true ATT. The True ATT column provides the oracle target parameter for each scenario, computed from the simulated counterfactual outcomes, giving a reference point for interpreting the magnitude of the errors. Rows are grouped into four blocks: two-period scenarios (1, 4, 7), six-period non-staggered scenarios (2, 5, 8), constant treatment effect scenarios (10, 11, 12), and staggered scenarios (3, 6, 9). Detailed tables with bias and confidence interval coverage are provided in the Appendix.

<!-- TABLE:COMBINED -->
| Scenario | Model     | True ATT | RMSE (500) | RMSE (2,500) | RMSE (10,000) |
| :------- | :-------- | -------: | ---------: | -----------: | ------------: |
| 1        | DML-Chang |    0.118 |      0.167 |        0.066 |         0.032 |
|          | TWFE      |          |      0.136 |        0.061 |         0.030 |
| 4        | DML-Chang |    0.106 |      0.191 |        0.076 |         0.036 |
|          | TWFE      |          |      0.219 |        0.184 |         0.174 |
| 7        | DML-Chang |   -0.922 |      0.355 |        0.120 |         0.059 |
|          | TWFE      |          |      0.240 |        0.181 |         0.170 |
| 2        | DML-Chang |    0.079 |      0.307 |        0.109 |         0.064 |
|          | TWFE      |          |      0.148 |        0.064 |         0.032 |
| 5        | DML-Chang |    0.124 |      0.344 |        0.125 |         0.079 |
|          | TWFE      |          |      0.213 |        0.155 |         0.144 |
| 8        | DML-Chang |   -0.572 |      0.501 |        0.327 |         0.299 |
|          | TWFE      |          |      0.230 |        0.145 |         0.122 |
| 10       | DML-Chang |    0.040 |      0.296 |        0.091 |         0.043 |
|          | TWFE      |          |      0.147 |        0.064 |         0.033 |
| 11       | DML-Chang |    0.051 |      0.333 |        0.107 |         0.051 |
|          | TWFE      |          |      0.210 |        0.157 |         0.142 |
| 12       | DML-Chang |   -0.625 |      0.546 |        0.371 |         0.350 |
|          | TWFE      |          |      0.233 |        0.145 |         0.121 |
| 3        | DML-Multi |   -0.050 |      0.133 |        0.053 |         0.026 |
|          | TWFE      |          |      0.330 |        0.200 |         0.166 |
| 6        | DML-Multi |   -0.125 |      0.143 |        0.059 |         0.028 |
|          | TWFE      |          |      0.373 |        0.293 |         0.274 |
| 9        | DML-Multi |   -1.195 |      0.193 |        0.080 |         0.036 |
|          | TWFE      |          |      0.596 |        0.555 |         0.550 |
<!-- /TABLE:COMBINED -->

The following figures display the distribution of estimation errors ($\hat{\tau} - \text{True ATT}$) across all 2,000 iterations for each scenario and estimator. A box centered on zero indicates an unbiased estimator, while wider boxes reflect higher variance. The three figures correspond to $n = 500$, $n = 2{,}500$, and $n = 10{,}000$, respectively.

![Estimation error distributions across 12 scenarios, $n = 500$, light LightGBM configuration, 2,000 iterations.](../results/500_light/master_boxplot.png)

![Estimation error distributions across 12 scenarios, $n = 2{,}500$, light LightGBM configuration, 2,000 iterations.](../results/2500_light/master_boxplot.png)

![Estimation error distributions across 12 scenarios, $n = 10{,}000$, light LightGBM configuration, 2,000 iterations.](../results/10000_light/master_boxplot.png)

### Two-Period Scenarios (1, 4, 7)

The two-period scenarios provide the cleanest comparison, since DML-Chang was designed for this setting: the first-differencing step discards no information and the target parameter aligns exactly with the overall ATT.

In Scenario 1 (simple confounding), both estimators perform well, with TWFE holding a slight RMSE edge due to its parametric efficiency (0.136 vs. 0.167 at $n = 500$). This is expected: with weak linear confounding and no covariate effect in $Y(0)$, TWFE is correctly specified. As confounding increases, however, DML-Chang pulls ahead. In Scenarios 4 and 7, TWFE carries an irreducible bias of approximately 0.17, while DML-Chang remains nearly unbiased. At $n = 500$, DML-Chang's higher variance still makes its RMSE worse in Scenario 7 (0.355 vs. 0.240). By $n = 10{,}000$, however, the variance has shrunk and DML-Chang achieves an RMSE of 0.059 against TWFE's 0.170. Across all three scenarios, DML-Chang's RMSE drops sharply as the sample grows, reflecting improved nuisance function estimates. TWFE's RMSE in the mid and complex cases, by contrast, is bounded by its bias floor.

### Six-Period Non-Staggered Scenarios (2, 5, 8)

These scenarios introduce a structural challenge for DML-Chang. The estimator operates on first-differenced data, using only the last pre-treatment and first post-treatment period. With dynamic treatment effects ($\delta_e = e$), the ATT at the first post-treatment period is smaller than the average across all post-treatment periods, creating an estimand mismatch.

In Scenario 2 (simple confounding), TWFE outperforms DML-Chang: its bias is zero and its RMSE is roughly half of DML-Chang's at every sample size. DML-Chang exhibits a persistent negative bias of approximately $-0.04$ to $-0.05$, consistent with the estimand mismatch. Scenario 5 (mid confounding) presents a trade-off: both estimators are biased, but TWFE's bias (approximately 0.14) is larger. DML-Chang overtakes TWFE in RMSE only at $n = 10{,}000$ (0.079 vs. 0.144); neither estimator is fully satisfactory.

Scenario 8 (complex confounding) is the most notable result in this block. DML-Chang's RMSE exceeds TWFE's at every sample size (0.501 vs. 0.230 at $n = 500$; 0.299 vs. 0.122 at $n = 10{,}000$). Here, the bias is driven not only by the estimand mismatch but also by the ML models' difficulty in capturing complex nonlinear nuisance functions from the limited first-differenced data. This demonstrates that DML is not a universal improvement: when the estimator's structural assumptions are violated, ML-based nuisance estimation can amplify rather than reduce error.

### Constant Treatment Effect Scenarios (10, 11, 12)

Scenarios 10--12 mirror the six-period scenarios (2, 5, 8) but with a constant treatment effect ($\delta_e = 1$), eliminating the estimand mismatch since the ATT at the first post-treatment period now equals the average ATT.

Scenario 10 (simple) confirms this: both estimators are unbiased and TWFE maintains a slight RMSE advantage. Scenario 11 (mid) reveals the benefit of removing the mismatch. DML-Chang's bias drops from $-0.06$ in Scenario 5 to 0.01, and it achieves an RMSE of 0.051 against TWFE's 0.142 at $n = 10{,}000$. However, Scenario 12 (complex) replicates the pattern of Scenario 8: DML-Chang's RMSE remains approximately 0.35 even at $n = 10{,}000$. This confirms that the dominant source of error in complex settings is the ML models' difficulty in estimating nuisance functions from first-differenced data, not the estimand mismatch.

### Staggered Scenarios (3, 6, 9)

The staggered scenarios produce the clearest results. DML-Multi achieves near-zero bias across all complexity levels and sample sizes, with RMSE declining from 0.133--0.193 at $n = 500$ to 0.026--0.036 at $n = 10{,}000$.

TWFE, by contrast, exhibits substantial bias. Its RMSE barely declines with sample size because the bias floor dominates: from 0.330 to 0.166 in Scenario 3, and from 0.373 to 0.274 in Scenario 6. In Scenario 9, TWFE's RMSE of 0.550 at $n = 10{,}000$ represents nearly half the magnitude of the true ATT ($-1.195$), while DML-Multi's RMSE is just 0.036. This result is consistent with the negative weighting problem discussed in Chapter 2. TWFE implicitly uses negative weights on certain group-time effects when treatment adoption is staggered, producing bias even when parallel trends hold unconditionally. DML-Multi avoids this by estimating group-time specific ATTs with proper aggregation weights.

### Discussion

Five main findings emerge from the simulation exercise. First, DML-Multi outperforms TWFE in staggered settings across all scenarios, maintaining near-zero bias while TWFE remains substantially biased. This carries the most direct implication for applied researchers.

Second, DML-Chang performs well in two-period settings when confounding is non-trivial. In Scenarios 4 and 7, it achieves near-zero bias while TWFE's bias of approximately 0.17 does not diminish with sample size. The advantage grows with sample size as DML-Chang's variance decreases while TWFE's bias floor does not.

Third, DML-Chang underperforms in multi-period non-staggered settings due to two sources of bias: an estimand mismatch from first-differencing with dynamic treatment effects, and limited nuisance function estimation under complex confounding. The constant-TE scenarios (10--12) isolate these mechanisms. Switching to constant effects eliminates the mismatch bias in mid-complexity (Scenario 11 vs. 5) but not in complex settings (Scenario 12 vs. 8). This confirms that the dominant problem is the ML models' difficulty in capturing nonlinear relationships from the reduced first-differenced data.

Fourth, TWFE remains competitive when correctly specified. In Scenarios 1, 2, and 10, TWFE achieves the lowest RMSE at every sample size. The efficiency advantage of a correctly specified parametric model should not be dismissed.

Fifth, sample size has asymmetric effects on the two approaches. For DML, larger samples improve nuisance function estimation, reducing RMSE substantially. For TWFE, larger samples shrink variance but cannot reduce bias. The relative advantage of DML over TWFE therefore grows with sample size wherever TWFE is misspecified.

A final note on hyperparameter sensitivity: additional simulations using the heavy LightGBM configuration (1,000 trees, depth 3) at $n = 500$ produced substantially higher RMSE for DML estimators with little improvement in bias. Overfitting the nuisance functions in small samples inflates variance without reducing the bias component, reinforcing the importance of parsimonious ML specifications when sample sizes are modest.

## Appendix

The following tables report bias, RMSE, and confidence interval coverage rate for each scenario and estimator, using the light LightGBM configuration. Coverage is the fraction of Monte Carlo replications in which the 95% confidence interval contains the true ATT. Rows are grouped in the same order as the main results table: two-period scenarios (1, 4, 7), six-period non-staggered (2, 5, 8), constant treatment effect (10, 11, 12), and staggered (3, 6, 9).

### Detailed Results: $n = 500$

<!-- TABLE:500 -->
| Scenario | Model     |   Bias |  RMSE | Coverage |
| :------- | :-------- | -----: | ----: | -------: |
| 1        | DML-Chang |  0.001 | 0.167 |    0.985 |
|          | TWFE      | -0.001 | 0.136 |    0.999 |
| 4        | DML-Chang | -0.006 | 0.191 |    0.975 |
|          | TWFE      |  0.167 | 0.219 |    0.983 |
| 7        | DML-Chang | -0.035 | 0.355 |    0.956 |
|          | TWFE      |  0.163 | 0.240 |    0.976 |
| 2        | DML-Chang | -0.053 | 0.307 |    0.974 |
|          | TWFE      | -0.005 | 0.148 |    1.000 |
| 5        | DML-Chang | -0.068 | 0.344 |    0.973 |
|          | TWFE      |  0.139 | 0.213 |    0.991 |
| 8        | DML-Chang |  0.270 | 0.501 |    0.881 |
|          | TWFE      |  0.115 | 0.230 |    0.991 |
| 10       | DML-Chang |  0.004 | 0.296 |    0.979 |
|          | TWFE      |  0.000 | 0.147 |    0.991 |
| 11       | DML-Chang |  0.008 | 0.333 |    0.976 |
|          | TWFE      |  0.143 | 0.210 |    0.935 |
| 12       | DML-Chang |  0.322 | 0.546 |    0.856 |
|          | TWFE      |  0.118 | 0.233 |    0.948 |
| 3        | DML-Multi |  0.005 | 0.133 |    1.000 |
|          | TWFE      |  0.155 | 0.330 |    0.626 |
| 6        | DML-Multi |  0.012 | 0.143 |    1.000 |
|          | TWFE      |  0.280 | 0.373 |    0.432 |
| 9        | DML-Multi |  0.021 | 0.193 |    0.999 |
|          | TWFE      |  0.552 | 0.596 |    0.105 |
<!-- /TABLE:500 -->

### Detailed Results: $n = 2{,}500$

<!-- TABLE:2500 -->
| Scenario | Model     |   Bias |  RMSE | Coverage |
| :------- | :-------- | -----: | ----: | -------: |
| 1        | DML-Chang |  0.000 | 0.066 |    0.983 |
|          | TWFE      |  0.001 | 0.061 |    0.999 |
| 4        | DML-Chang |  0.002 | 0.076 |    0.960 |
|          | TWFE      |  0.172 | 0.184 |    0.683 |
| 7        | DML-Chang |  0.012 | 0.120 |    0.924 |
|          | TWFE      |  0.164 | 0.181 |    0.814 |
| 2        | DML-Chang | -0.037 | 0.109 |    0.948 |
|          | TWFE      |  0.002 | 0.064 |    1.000 |
| 5        | DML-Chang | -0.060 | 0.125 |    0.926 |
|          | TWFE      |  0.139 | 0.155 |    0.907 |
| 8        | DML-Chang |  0.291 | 0.327 |    0.442 |
|          | TWFE      |  0.111 | 0.145 |    0.946 |
| 10       | DML-Chang |  0.003 | 0.091 |    0.981 |
|          | TWFE      |  0.002 | 0.064 |    0.989 |
| 11       | DML-Chang |  0.011 | 0.107 |    0.959 |
|          | TWFE      |  0.140 | 0.157 |    0.674 |
| 12       | DML-Chang |  0.342 | 0.371 |    0.324 |
|          | TWFE      |  0.114 | 0.145 |    0.839 |
| 3        | DML-Multi |  0.002 | 0.053 |    1.000 |
|          | TWFE      |  0.154 | 0.200 |    0.430 |
| 6        | DML-Multi |  0.006 | 0.059 |    1.000 |
|          | TWFE      |  0.272 | 0.293 |    0.072 |
| 9        | DML-Multi | -0.011 | 0.080 |    0.998 |
|          | TWFE      |  0.546 | 0.555 |    0.000 |
<!-- /TABLE:2500 -->

### Detailed Results: $n = 10{,}000$

<!-- TABLE:10000 -->
| Scenario | Model     |   Bias |  RMSE | Coverage |
| :------- | :-------- | -----: | ----: | -------: |
| 1        | DML-Chang |  0.001 | 0.032 |    0.986 |
|          | TWFE      |  0.000 | 0.030 |    1.000 |
| 4        | DML-Chang |  0.002 | 0.036 |    0.955 |
|          | TWFE      |  0.171 | 0.174 |    0.014 |
| 7        | DML-Chang |  0.020 | 0.059 |    0.873 |
|          | TWFE      |  0.165 | 0.170 |    0.105 |
| 2        | DML-Chang | -0.041 | 0.064 |    0.857 |
|          | TWFE      |  0.000 | 0.032 |    1.000 |
| 5        | DML-Chang | -0.059 | 0.079 |    0.769 |
|          | TWFE      |  0.139 | 0.144 |    0.257 |
| 8        | DML-Chang |  0.291 | 0.299 |    0.009 |
|          | TWFE      |  0.114 | 0.122 |    0.646 |
| 10       | DML-Chang |  0.000 | 0.043 |    0.974 |
|          | TWFE      |  0.000 | 0.033 |    0.986 |
| 11       | DML-Chang |  0.011 | 0.051 |    0.948 |
|          | TWFE      |  0.138 | 0.142 |    0.072 |
| 12       | DML-Chang |  0.343 | 0.350 |    0.001 |
|          | TWFE      |  0.112 | 0.121 |    0.429 |
| 3        | DML-Multi |  0.002 | 0.026 |    1.000 |
|          | TWFE      |  0.155 | 0.166 |    0.074 |
| 6        | DML-Multi |  0.007 | 0.028 |    1.000 |
|          | TWFE      |  0.269 | 0.274 |    0.000 |
| 9        | DML-Multi | -0.006 | 0.036 |    0.999 |
|          | TWFE      |  0.548 | 0.550 |    0.000 |
<!-- /TABLE:10000 -->

## References

Callaway, Brantly, and Pedro H. C. Sant'Anna. 2021. "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics* 225 (2): 200--230. [https://doi.org/10.1016/j.jeconom.2020.12.001](https://doi.org/10.1016/j.jeconom.2020.12.001).

Hatamyar, R., Chang, N., and Sant'Anna, P. H. C. 2023. "Double Machine Learning for Difference-in-Differences with Multiple Time Periods." Working Paper.

Ke, Guolin, Qi Meng, Thomas Finley, et al. 2017. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Proceedings of the 31st International Conference on Neural Information Processing Systems* (Red Hook, NY, USA), NIPS'17, 3149--57.
