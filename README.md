# Double Machine Learning for Difference in Difference: Fundamentals and Applications

Machine Learning (ML) models have traditionally been associated with prediction tasks due to their flexibility, while social scientists have typically relied on simpler, often linear, regressions for assessing causality. However, a novel framework named Double Machine Learning (DML) has emerged, providing a way to leverage the predictive performance of these complex methods for robust causal estimation. This thesis examines the fundamentals and applications of double machine learning for two distinct popular Difference-in-Differences (DiD) settings.


## To do
- [X] Translate .tex files to .md to word for sharing with my advisor. Then the updated workflow will be markdown -> word until final version that will be in latex.
- [X] Find conferences and seminars to present it. Share the list with advisor
- [ ] Work on a new chapter about simulations, before the applications

## Simulations
|Scenario|Periods|Staggered|Confounding $c_i$|PS strength ($\gamma_{max}$)|Outcome $Y(0)$ complexity|
|---|---|---|---|---|---|
|1|2|no|$X_1+X_2+X_3$|weak (0.25)|$\theta_t + \alpha_i + \varepsilon$|
|2|6|no|$X_1+X_2+X_3$|weak (0.25)|$\theta_t + \alpha_i + \varepsilon$|
|3|6|yes|$X_1+X_2+X_3$|weak (0.25)|$\theta_t + \alpha_i + \varepsilon$|
|4|2|no|$X_1+X_2+X_3 + X_2 X_3$|moderate (0.5)|$\theta_t + \alpha_i + X_1 \beta + \varepsilon$|
|5|6|no|$X_1+X_2+X_3 + X_2 X_3$|moderate (0.5)|$\theta_t + \alpha_i + X_1 \beta + \varepsilon$|
|6|6|yes|$X_1+X_2+X_3 + X_2 X_3$|moderate (0.5)|$\theta_t + \alpha_i + X_1 \beta + \varepsilon$|
|7|2|no|$X_1^2 + X_2 X_3 + \mathbb{1}[X_2>0] X_1 + \sin(X_1+X_2)$|strong (1.0)|$\theta_t + \alpha_i + \sin(X_1) + X_2^2 + X_1 X_3 + \varepsilon$|
|8|6|no|$X_1^2 + X_2 X_3 + \mathbb{1}[X_2>0] X_1 + \sin(X_1+X_2)$|strong (1.0)|$\theta_t + \alpha_i + \sin(X_1) + X_2^2 + X_1 X_3 + \varepsilon$|
|9|6|yes|$X_1^2 + X_2 X_3 + \mathbb{1}[X_2>0] X_1 + \sin(X_1+X_2)$|strong (1.0)|$\theta_t + \alpha_i + \sin(X_1) + X_2^2 + X_1 X_3 + \varepsilon$|


