# Chapter 1: Introduction

As discussed in (Shmueli 2010), when working in applied statistics there is a clear distinction between predicting and explaining. Prediction is usually associated with achieving the best performance on a selected goodness-of-fit metric, while explaining is focused on understanding the effects or relationships between variables. This distinction makes it clear why flexible, and often considered black-box, Machine Learning (ML) algorithms are widely used when the goal is to achieve the best prediction performance. However, when the goal is to interpret results or to draw causal relationships, other, usually simpler approaches like linear regressions are preferred. This distinction is not new, as it was highlighted by (Breiman 2001), who clearly differentiated between statistics and machine learning when the latter field started to gain popularity.

The focus on prediction and not explanation generated criticism among economists and other social scientists who needed tools for understanding and studying causal relationships, which is not possible with black-box models. However, the new models were appealing, so in recent years, many authors have explored how to bridge this gap. Early work, such as (Varian 2014), highlighted how ML techniques, especially tree-based models, could complement traditional econometric methods in settings with non-linearities and complex interactions. Similarly, (Mullainathan and Spiess 2017) explored the practical applications of machine learning in econometrics, particularly emphasizing prediction, but also cautioning against drawing causal conclusions about the effects of independent variables without careful consideration.

Although the prevailing view cautioned against using machine learning for explaining, these algorithms gained popularity among researchers over the years. As shown in (Desai 2023), which reviews how ML algorithms are being integrated into economic analysis, this popularity has grown significantly. An attempt to bridge the gap between black-box, prediction-focused machine learning methods and simpler, more interpretable methods was the emergence of interpretable machine learning. There are now several techniques that aim to combine the predictive power of complex, black box machine learning models with methods to interpret them. A comprehensive overview of these techniques can be found in (Molnar 2025). However, it is important to note that many of these methods focus on describing the behavior of the model itself, rather than uncovering the underlying data generating process (DGP) as is common in classical statistics.

While interpretable ML addresses the transparency problem, it does not directly solve the fundamental challenge economists face: estimating causal effects with formal statistical guarantees. What is needed is a framework that uses machine learning not to explain its own predictions, but rather to flexibly control for confounding variables while maintaining the ability to perform valid statistical inference on causal parameters. This work focuses its attention on a particularly influential framework that achieves exactly this: Double Machine Learning (DML) from (Chernozhukov et al. 2018).

The DML framework provides a general, robust method that formally combines the predictive power of machine learning with the theoretical rigor of causal inference to estimate a specific parameter of interest. Notably, the framework is not restricted to a single econometric setting. It can be adapted to multiple familiar contexts for economists, including Instrumental Variables (IV) estimation, treatment effect models, and, central to this thesis, Difference-in-Differences (DiD) designs.

This document aims to contribute to the growing literature on how to apply these methods in practice, focusing specifically on the application of DML within the DiD framework. By providing both conceptual foundations and practical guidance, this thesis seeks to equip practitioners with the understanding and tools needed to leverage this powerful technique in their own research.

The rest of this document is structured as follows: First, the classic Difference-in-Differences framework and its econometric tools are introduced. Second, the Double Machine Learning (DML) framework is presented along with its specific application for DiD setups. Then, two real-world applications of this algorithm are provided to demonstrate its practical utility, and finally, a conclusion is presented.

Breiman, Leo. 2001. "[Statistical Modeling: The Two Cultures (with comments and a rejoinder by the author)]{.nocase}." *Statistical Science* 16 (3): 199--231. <https://doi.org/10.1214/ss/1009213726>.

Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, et al. 2018. "Double/Debiased Machine Learning for Treatment and Structural Parameters." *The Econometrics Journal* 21 (1): C1--68. <https://doi.org/10.1111/ectj.12097>.

Desai, Ajit. 2023. *Machine Learning for Economics Research: When What and How?* Papers. SSRN. <https://doi.org/10.2139/ssrn.4404772>.

Molnar, Christoph. 2025. *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 3rd ed. <https://christophm.github.io/interpretable-ml-book>.

Mullainathan, Sendhil, and Jann Spiess. 2017. "Machine Learning: An Applied Econometric Approach." *Journal of Economic Perspectives* 31 (2): 87--106. <https://doi.org/10.1257/jep.31.2.87>.

Shmueli, Galit. 2010. "[To Explain or to Predict?]{.nocase}" *Statistical Science* 25 (3): 289--310. <https://doi.org/10.1214/10-STS330>.

Varian, Hal R. 2014. "Big Data: New Tricks for Econometrics." *Journal of Economic Perspectives* 28 (2): 3--28. <https://doi.org/10.1257/jep.28.2.3>.
