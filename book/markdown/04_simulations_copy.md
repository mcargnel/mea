# Chapter 4: Simulations

This chapter presents a Monte Carlo simulation exercise designed to illustrate when DML estimators offer meaningful advantages over classical approaches in Difference-in-Differences settings, and when a simpler Two-Way Fixed Effects (TWFE) specification is preferred. The simulation procedure builds on the framework introduced by Hatamyar et al. (2023), which itself extended the staggered DiD design of Callaway and Sant’Anna (2021). The present work adapts and extends the Hatamyar et al. (2023) DGP in several directions: it introduces multiple tiers of confounding complexity (beyond the original linear specification), adds nonlinear outcome models that create conditional parallel trends, varies the propensity score strength parameter across scenarios, and includes noise covariates to test estimator robustness.

The chapter is organized as follows. First, the data generating process (DGP) is described, covering the construction of covariates, treatment assignment, potential outcomes, and observed outcomes. Second, the simulation scenarios are defined, varying the complexity of confounding and outcome models across three tiers. Third, the estimation procedures and Monte Carlo design are outlined.

## Data Generating Process

The simulation framework generates panel data for units observed over periods. Each unit is assigned to a treatment cohort , where indicates a never-treated unit and indicates that unit first receives treatment at period .

The core of the DGP is defined by two potential outcome equations. The untreated potential outcome for unit at period is:

and the treated potential outcome is:

where is a shared time trend, is a unit-specific fixed effect, is an idiosyncratic error, is a function of covariates whose complexity varies across scenarios, is a covariate index entering , is a time-varying coefficient on covariates in the treated outcome, is a dynamic treatment effect component that depends on event time , and is a unit-specific treatment effect modifier. The observed outcome switches between these two equations based on treatment status:

In Hatamyar et al. (2023), the untreated outcome did not depend on covariates (equivalent to ), confounding was limited to a single linear specification, and the propensity score strength was fixed. The present design extends each of these dimensions, as described in the subsections that follow.

### Covariates

Five types of covariates are generated for each unit. Two continuous covariates and are drawn independently from a standard normal distribution. A binary covariate is drawn from a Bernoulli distribution with probability 0.5. These three covariates mirror the covariate structure used in Hatamyar et al. (2023). An additional binary covariate , also drawn from a Bernoulli (0.5) distribution, is included as a pure noise variable that does not enter any model equation.

An optional time-varying covariate , not present in Hatamyar et al. (2023), can also be generated. For each period , follows a sine wave pattern plus noise:

where represents an evenly spaced grid over . This covariate introduces temporal variation in the covariate structure, which is relevant for testing estimators in settings where conditioning variables evolve over time.

### Treatment Assignment

Treatment assignment is governed by a multinomial model that determines each unit’s cohort membership . Under random assignment, each cohort (including the never-treated group) is equally likely, so for all .

Under confounded assignment, the probability of belonging to each cohort depends on the covariates through a multinomial logit model:

where is a cohort-specific parameter that scales with the propensity score strength , and is a confounding index that depends on the covariates according to a specified complexity level. Three levels of confounding complexity are considered. Under the simple specification, , creating a linear, additive confounding structure that is straightforward for any estimator to model. The mid specification adds an interaction term, . The complex specification introduces squared terms, indicator functions, and trigonometric transformations, , creating a confounding structure that is difficult for linear models to approximate. This is one of the differences when comparing against Callaway and Sant’Anna (2021) where covariates weren’t included and Hatamyar et al. (2023) where only the simple structure was included.

The propensity score strength parameter controls how strongly the covariates influence treatment assignment. Higher values of create greater imbalance between treated and control groups, making the confounding bias more severe when covariates are not properly adjusted for. In Hatamyar et al. (2023), this parameter was fixed; the present design varies across scenarios (from 0.25 to 1.0) to examine how increasing confounding strength affects estimator performance.

In the non-staggered case, all treated units are collapsed into a single cohort that begins treatment at period 2, simplifying the design to one with a single treatment group and a clean pre-treatment period.

### Fixed Effects

Each unit receives an individual-specific fixed effect that captures permanent level differences between units. For treated units, the fixed effect is drawn as , so units in later treatment cohorts tend to have higher baseline levels. For untreated units, . A correctly specified DiD estimator should difference out these fixed effects, but their correlation with treatment timing introduces an additional identification challenge.

### Potential Outcomes

The untreated potential outcome , presented at the beginning of this section, includes a covariate effect scaled by a time-varying coefficient that grows over the panel. This design creates parallel trends conditional on but not unconditionally, providing a setting where covariate adjustment is necessary for valid identification. This is a difference from Hatamyar et al. (2023), where covariates did not enter the untreated outcome equation. The introduction of outcome complexity tiers, particularly the mid and complex specifications, is one of the central extensions of the present simulation design.

Three levels of outcome complexity determine the function . Under the simple specification, , so covariates do not affect , parallel trends hold unconditionally, and TWFE is correctly specified. This replicates the Hatamyar et al. (2023) setting. The mid and complex specifications are extensions introduced in the present work. The mid specification sets , adding a linear covariate effect that grows over time, requiring covariate adjustment but within the capacity of parametric methods. The complex specification sets , introducing nonlinearities in the outcome model such that linear controls cannot fully remove the covariate effect from , creating a setting where flexible ML-based estimators should outperform linear methods.

The treatment effect is heterogeneous across units. The unit-specific treatment effect modifier is generated according to one of two specifications: (linear in ), or (nonlinear). Hatamyar et al. (2023) used only the linear specification; the nonlinear alternative is an extension that assesses whether ML-based estimators can capture more complex treatment effect heterogeneity patterns.

The covariates entering the treated outcome equation are collected in a model index . The parameter controls the dimensionality of this index: when , only enters; otherwise, the index is . The dynamic treatment effect component depends on event time, with under dynamic effects and under constant effects. The treatment effect for a treated unit in a post-treatment period thus has two parts: a dynamic component that grows with event time, and a constant baseline of 1. Both components are multiplied by the unit-specific , so the actual treatment effect varies both across units and over time.

### Observed Outcomes

For treated units (), the observed outcome equals in post-treatment periods () and in pre-treatment periods (). For never-treated units (), the observed outcome is always .

The final dataset is assembled in long (panel) format with unit and period identifiers, treatment cohort indicators, observed and counterfactual outcomes, covariate values, and group membership probabilities. Units with are excluded from the analysis because they have no pre-treatment period available, which is required for the DiD identification strategy.

## Simulation Setup

Many different combinations are possible with the data generating process described above. This exercise focuses on twelve scenarios, organized in three complexity tiers with a systematic variation in the number of periods and the treatment adoption structure.

Scenarios 1 through 9 form a grid that crosses three complexity tiers (simple, mid, complex) with three panel structures (2-period non-staggered, 6-period non-staggered, 6-period staggered), all with dynamic treatment effects. Within each tier, the first scenario uses 2 periods without staggered adoption (Scenarios 1, 4, 7), the second uses 6 periods without staggered adoption (Scenarios 2, 5, 8), and the third uses 6 periods with staggered adoption (Scenarios 3, 6, 9). As complexity increases, so does the propensity score strength (from 0.25 in the simple tier to 1.0 in the complex tier), making the confounding progressively harder to address. Scenarios 10 through 12 are 6-period non-staggered designs at the simple, mid, and complex complexity levels, respectively, but replace the dynamic treatment effect () with a constant one (), isolating the role of treatment effect dynamics. Because the scenarios are numbered sequentially within each complexity tier (1 to 3 for simple, 4 to 6 for mid, 7 to 9 for complex) but the results section groups them by panel structure, the mapping between scenario number and design choices is not immediately obvious. Table 0 collects all the relevant dimensions in one place so the reader can quickly locate each scenario's confounding specification, outcome model, propensity score strength, panel length, adoption pattern, and treatment effect type.

Table 0: Simulation scenario design.

| **new scenario**| **Scenario** | **Tier** | **Confounding (g)** | **Outcome (h)** | **PS strength (α)** | **Periods** | **Staggered** | **Treatment effect** | **DML estimator** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|1| 10 | Simple | Linear | 0 | 0.25 | 6 | No | Constant | Chang |
|2| 11 | Mid | Interaction | Linear | 0.50 | 6 | No | Constant | Chang |
|3| 12 | Complex | Nonlinear | Nonlinear | 1.00 | 6 | No | Constant | Chang |
|4| 1 | Simple | Linear | 0 | 0.25 | 2 | No | Dynamic | Chang |
|5| 4 | Mid | Interaction | Linear | 0.50 | 2 | No | Dynamic | Chang |
|6| 7 | Complex | Nonlinear | Nonlinear | 1.00 | 2 | No | Dynamic | Chang |
|7| 2 | Simple | Linear | 0 | 0.25 | 6 | No | Dynamic | Chang |
|8| 5 | Mid | Interaction | Linear | 0.50 | 6 | No | Dynamic | Chang |
|9| 8 | Complex | Nonlinear | Nonlinear | 1.00 | 6 | No | Dynamic | Chang |
|10| 3 | Simple | Linear | 0 | 0.25 | 6 | Yes | Dynamic | Multi |
|11| 6 | Mid | Interaction | Linear | 0.50 | 6 | Yes | Dynamic | Multi |
|12| 9 | Complex | Nonlinear | Nonlinear | 1.00 | 6 | Yes | Dynamic | Multi |


For the non-staggered scenarios, TWFE is compared against the DML estimator of Chang (2020). In the simple tier, both estimators should perform similarly, since the confounding is linear, weak, and covariates do not enter . As complexity increases, TWFE is expected to suffer from its inability to capture the nonlinear relationships in both the confounding and outcome models. The constant treatment effect scenarios (10 through 12) serve as an additional diagnostic: because the Chang (2020) estimator was designed for two-period settings where the treatment effect does not vary over time, it should perform well in these scenarios even with multiple periods, provided the complexity remains manageable.

For the staggered scenarios (3, 6, 9), TWFE is compared against the DML estimator with the event study aggregation from Callaway and Sant’Anna (2021). Standard TWFE is expected to produce biased estimates due to the negative weighting problem discussed in Chapter 2, with this bias growing as the complexity of confounding and the outcome model increases.

The target parameter is the Average Treatment Effect on the Treated (ATT), computed from the simulated counterfactual outcomes as . To ensure robust results, each scenario is evaluated through 2,000 Monte Carlo replications with sample sizes of for 2 and 6 periods. Larger samples should benefit the DML estimators, which rely on machine learning models that improve with more training data.

Since each scenario involves different levels of complexity, the hyperparameters of the machine learning models were adapted accordingly. Given the large number of simulations and scenario variations, computational cost is a relevant concern, so LightGBM (Ke et al., 2017) was chosen as the machine learning model for both the outcome model and the propensity score model . Three hyperparameter configurations were considered: the light configuration uses 50 trees with maximum depth 2 and a learning rate of 0.1, the default configuration increases the number of trees to 200 while keeping the same depth and learning rate and the heavy configuration uses 1,000 trees with maximum depth 3 and a lower learning rate of 0.05. These parameters jointly determine how much flexibility the model has for estimating the nuisance functions. More trees, greater depth, and a lower learning rate allow the model to capture increasingly complex relationships, but also increase the risk of overfitting. In simpler data generating scenarios, the heavy configuration may overfit the training data, leading to noisier nuisance function estimates and potentially worse performance than a more parsimonious specification. Cross-fitting is performed with 5 folds in all cases, ensuring that nuisance function predictions for each observation are generated by models trained on different data, as described in Chapter 3.

## Results

Table 1 reports the root mean squared error (RMSE) for each scenario-estimator combination across all three sample sizes (, , and ), using the ‘light’ LightGBM configuration (50 trees, maximum depth 2)<sup>[\[1\]](#footnote-1)</sup>. This parsimonious specification was selected as the baseline to minimize the risk of overfitting, which can inflate variance in smaller samples and obscure the comparison between estimators. RMSE combines bias and variance into a single accuracy measure: lower values indicate that the estimator’s point estimates are, on average, closer to the true ATT. The True ATT column provides the target parameter for each scenario, computed from the simulated counterfactual outcomes, giving a reference point for interpreting the magnitude of the errors. Rows are grouped into four blocks: two-period scenarios (1, 4, 7), six-period non-staggered scenarios (2, 5, 8), constant treatment effect scenarios (10, 11, 12), and staggered scenarios (3, 6, 9).

Table 1: Simulation results for all scenarios and units, using ‘light’ ML set up.

| **Scenario** | **Model** | **True ATT** | **RMSE (500)** | **RMSE (2,500)** | **RMSE (10,000)** |
| --- | --- | --- | --- | --- | --- |
| 1   | DML-Chang | 0.118 | 0.167 | 0.066 | 0.032 |
| TWFE | 0.136 | 0.061 | 0.030 |
| 4   | DML-Chang | 0.106 | 0.191 | 0.076 | 0.036 |
| TWFE | 0.219 | 0.184 | 0.174 |
| 7   | DML-Chang | \-0.922 | 0.355 | 0.120 | 0.059 |
| TWFE | 0.240 | 0.181 | 0.170 |
| 2   | DML-Chang | 0.079 | 0.307 | 0.109 | 0.064 |
| TWFE | 0.148 | 0.064 | 0.032 |
| 5   | DML-Chang | 0.124 | 0.344 | 0.125 | 0.079 |
| TWFE | 0.213 | 0.155 | 0.144 |
| 8   | DML-Chang | \-0.572 | 0.501 | 0.327 | 0.299 |
| TWFE | 0.230 | 0.145 | 0.122 |
| 10  | DML-Chang | 0.040 | 0.296 | 0.091 | 0.043 |
| TWFE | 0.147 | 0.064 | 0.033 |
| 11  | DML-Chang | 0.051 | 0.333 | 0.107 | 0.051 |
| TWFE | 0.210 | 0.157 | 0.142 |
| 12  | DML-Chang | \-0.625 | 0.546 | 0.371 | 0.350 |
| TWFE | 0.233 | 0.145 | 0.121 |
| 3   | DML-Multi | \-0.050 | 0.133 | 0.053 | 0.026 |
| TWFE | 0.330 | 0.200 | 0.166 |
| 6   | DML-Multi | \-0.125 | 0.143 | 0.059 | 0.028 |
| TWFE | 0.373 | 0.293 | 0.274 |
| 9   | DML-Multi | \-1.195 | 0.193 | 0.080 | 0.036 |
| TWFE | 0.596 | 0.555 | 0.550 |

The following sub sections will deep dive into the scenarios. In each case, figures with the distribution of estimation errors () across all 2,000 iterations for each group of scenarios will be use to provide a more complete view of the results. In the three cases, a box centered on zero indicates an unbiased estimator, while wider boxes reflect higher variance.

### Two-Period Scenarios (1, 4, 7)

The two-period scenarios (Figure 1) provide the cleanest comparison, since DML-Chang was designed for this setting: the first-differencing step discards no information, and the target parameter aligns exactly with the overall ATT.

Figure 1: Distribution of estimation errors across simulation scenarios 1, 4 & 7 comparing TWFE and DML estimators on simulated data with 500 units. The dashed line at zero indicates unbiased estimation.

In Scenario 1 (simple confounding), both estimators perform well, with TWFE holding a slight RMSE edge due to its parametric efficiency (0.136 vs. 0.167 at ). This is expected: with weak linear confounding and no covariate effect in , TWFE is correctly specified. As confounding increases, however, DML-Chang pulls ahead. In Scenarios 4 and 7, TWFE carries an irreducible bias of approximately 0.17, while DML-Chang remains nearly unbiased. At , DML-Chang’s higher variance still makes its RMSE worse in Scenario 7 (0.355 vs. 0.240). By , however, the variance has shrunk and DML-Chang achieves an RMSE of 0.059 against TWFE’s 0.170. Across all three scenarios, DML-Chang’s RMSE drops sharply as the sample grows, reflecting improved nuisance function estimates. TWFE’s RMSE in the mid and complex cases, by contrast, is bounded by its bias floor.

### Six-Period Non-Staggered Scenarios (2, 5, 8)

These scenarios introduce a structural challenge for DML-Chang. The estimator operates on first-differenced data, using only the last pre-treatment and first post-treatment period. With dynamic treatment effects (), the ATT at the first post-treatment period is smaller than the average across all post-treatment periods, creating an estimand mismatch. Results are shown in Figure 2.

Figure 2: Distribution of estimation errors across simulation scenarios 2, 5 & 8 comparing TWFE and DML estimators on simulated data with 500 units. The dashed line at zero indicates unbiased estimation.

In Scenario 2 (simple confounding), TWFE outperforms DML-Chang: its bias is zero and its RMSE is roughly half of DML-Chang’s at every sample size. DML-Chang exhibits a persistent negative bias of approximately to , consistent with the estimand mismatch. Scenario 5 (mid confounding) presents a trade-off: both estimators are biased, but TWFE’s bias (approximately 0.14) is larger. DML-Chang overtakes TWFE in RMSE only at (0.079 vs. 0.144); neither estimator is fully satisfactory as both are consistently biased.

Scenario 8 (complex confounding) is the most notable result in this block. DML-Chang’s RMSE exceeds TWFE’s at every sample size (0.501 vs. 0.230 at ; 0.299 vs. 0.122 at ). Here, the bias is driven not only by the estimand mismatch but also by the ML models’ difficulty in capturing complex nonlinear nuisance functions from the limited first-differenced data. This demonstrates that DML is not a universal improvement: when the estimator’s structural assumptions are violated, ML-based nuisance estimation can amplify rather than reduce error. Once again, even if TWFE has an edge on RMSE, both fail to recover the true effect consistently.

### Constant Treatment Effect Scenarios (10, 11, 12)

Scenarios 10, 11 and 12 (Figure 3) mirror the six-period scenarios (2, 5, 8) but with a constant treatment effect (), eliminating the estimand mismatch since the ATT at the first post-treatment period now equals the average ATT.

Figure 3: Distribution of estimation errors across simulation scenarios 10,11 & 12 comparing TWFE and DML estimators on simulated data with 500 units. The dashed line at zero indicates unbiased estimation.

Scenario 10 (simple) confirms this: both estimators are unbiased and TWFE maintains a slight RMSE advantage. Scenario 11 (mid) reveals the benefit of removing the mismatch. DML-Chang’s bias drops from in Scenario 5 to 0.01, and it achieves an RMSE of 0.051 against TWFE’s 0.142 at with unbiased results for the three sample sizes, while TWFE is biased. However, Scenario 12 (complex) replicates the pattern of Scenario 8: DML-Chang’s RMSE remains approximately 0.35 even at and both estimators are biased. This confirms that the dominant source of error in complex settings is the ML models’ difficulty in estimating nuisance functions from first-differenced data, not the estimand mismatch.

### Staggered Scenarios (3, 6, 9)

The staggered scenarios (Figure 4) produce the clearest results. DML-Multi achieves near-zero bias across all complexity levels and sample sizes, with RMSE declining from 0.133–0.193 at to 0.026–0.036 at .

Figure 4: Distribution of estimation errors across simulation scenarios 3, 6 & 9 comparing TWFE and DML estimators on simulated data with 500 units. The dashed line at zero indicates unbiased estimation.

TWFE, by contrast, exhibits substantial bias. Its RMSE barely declines with sample size because the bias floor dominates: from 0.330 to 0.166 in Scenario 3, and from 0.373 to 0.274 in Scenario 6. In Scenario 9, TWFE’s RMSE of 0.550 at represents nearly half the magnitude of the true ATT (), while DML-Multi’s RMSE is just 0.036. This result is consistent with the negative weighting problem discussed in Chapter 2. TWFE implicitly uses negative weights on certain group-time effects when treatment adoption is staggered, producing bias even when parallel trends hold unconditionally. DML-Multi avoids this by estimating group-time specific ATTs with proper aggregation weights.

## Appendix

Table 2: Simulation results for all scenarios and units, using ‘default' ML set up.

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Scenario | Model | True ATT | RMSE (500) | RMSE (2,500) | RMSE (10,000) |
| 1   | DML-Chang | 0.118 | 0.262 | 0.076 | 0.032 |
| TWFE | 0.136 | 0.061 | 0.030 |
| 4   | DML-Chang | 0.106 | 0.365 | 0.098 | 0.041 |
| TWFE | 0.219 | 0.184 | 0.174 |
| 7   | DML-Chang | \-0.922 | 0.661 | 0.180 | 0.077 |
| TWFE | 0.240 | 0.181 | 0.170 |
| 2   | DML-Chang | 0.079 | 0.593 | 0.141 | 0.067 |
| TWFE | 0.148 | 0.064 | 0.032 |
| 5   | DML-Chang | 0.124 | 0.619 | 0.162 | 0.084 |
| TWFE | 0.213 | 0.155 | 0.144 |
| 8   | DML-Chang | \-0.572 | 0.741 | 0.349 | 0.301 |
| TWFE | 0.230 | 0.145 | 0.122 |
| 10  | DML-Chang | 0.040 | 0.568 | 0.123 | 0.047 |
| TWFE | 0.147 | 0.064 | 0.033 |
| 11  | DML-Chang | 0.051 | 0.636 | 0.151 | 0.058 |
| TWFE | 0.210 | 0.157 | 0.142 |
| 12  | DML-Chang | \-0.625 | 0.789 | 0.391 | 0.352 |
| TWFE | 0.233 | 0.145 | 0.121 |
| 3   | DML-Multi | \-0.050 | 0.154 | 0.058 | 0.027 |
| TWFE | 0.330 | 0.200 | 0.166 |
| 6   | DML-Multi | \-0.125 | 0.172 | 0.067 | 0.029 |
| TWFE | 0.373 | 0.293 | 0.274 |
| 9   | DML-Multi | \-1.195 | 0.242 | 0.103 | 0.044 |
| TWFE | 0.596 | 0.555 | 0.550 |

Table 3: Simulation results for all scenarios and units, using ‘heavy' ML set up.

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Scenario | Model | True ATT | RMSE (500) | RMSE (2,500) | RMSE (10,000) |
| 1   | DML-Chang | 0.118 | 0.876 | 0.138 | 0.040 |
| TWFE | 0.136 | 0.061 | 0.030 |
| 4   | DML-Chang | 0.106 | 1.001 | 0.203 | 0.059 |
| TWFE | 0.219 | 0.184 | 0.174 |
| 7   | DML-Chang | \-0.922 | 1.372 | 0.315 | 0.103 |
| TWFE | 0.240 | 0.181 | 0.170 |
| 2   | DML-Chang | 0.079 | 1.153 | 0.308 | 0.087 |
| TWFE | 0.148 | 0.064 | 0.032 |
| 5   | DML-Chang | 0.124 | 1.142 | 0.319 | 0.102 |
| TWFE | 0.213 | 0.155 | 0.144 |
| 8   | DML-Chang | \-0.572 | 1.166 | 0.444 | 0.308 |
| TWFE | 0.230 | 0.145 | 0.122 |
| 10  | DML-Chang | 0.040 | 1.148 | 0.300 | 0.071 |
| TWFE | 0.147 | 0.064 | 0.033 |
| 11  | DML-Chang | 0.051 | 1.207 | 0.301 | 0.086 |
| TWFE | 0.210 | 0.157 | 0.142 |
| 12  | DML-Chang | \-0.625 | 1.189 | 0.465 | 0.356 |
| TWFE | 0.233 | 0.145 | 0.121 |
| 3   | DML-Multi | \-0.050 | 0.208 | 0.087 | 0.032 |
| TWFE | 0.330 | 0.200 | 0.166 |
| 6   | DML-Multi | \-0.125 | 0.219 | 0.094 | 0.036 |
| TWFE | 0.373 | 0.293 | 0.274 |
| 9   | DML-Multi | \-1.195 | 0.285 | 0.134 | 0.058 |
| TWFE | 0.596 | 0.555 | 0.550 |

## References

Callaway, B., & Santa’Anna, P. H. C. (2021). Difference-in-Differences with multiple time periods. _Journal of Econometrics_, _225_(2), 200–230. https://doi.org/10.1016/j.jeconom.2020.12.001

Chang, N.-C. (2020). Double/debiased machine learning for difference-in-differences models. _The Econometrics Journal_, _23_(2), 177–191. https://doi.org/10.1093/ectj/utaa001

Hatamyar, J., Kreif, N., Rocha, R., & Huber, M. (2023). _Machine Learning for Staggered Difference-in-Differences and Dynamic Treatment Effect Heterogeneity_ (arXiv:2310.11962). arXiv. https://doi.org/10.48550/arXiv.2310.11962

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: a highly efficient gradient boosting decision tree. _Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17_, 3149–3157.

1.  Tables with ‘default’ and ‘heavy’ configurations are provided in the Appendix [↑](#footnote-ref-1)