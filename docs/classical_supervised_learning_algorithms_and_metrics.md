# Classical Supervised Learning for Tabular Data

## 1. Purpose and scope

Supervised learning aims to learn a mapping from a set of predictors, or features, to an observed target.

A dataset can be represented conceptually as:

$$
\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}
$$

where $\mathbf{x}_i$ contains the predictors for observation $i$, and $y_i$ is the target.

The nature of $y$ determines the main supervised-learning task:

- **binary classification**, when the target has two categories;
- **multiclass classification**, when the target has more than two mutually exclusive categories;
- **regression**, when the target is numerical and continuous.

This document provides a foundation for classical supervised learning in tabular data. It focuses on the logic behind common algorithms, their assumptions and behavior, the metrics used to evaluate them, and the interpretation of model outputs.

The objective is not to identify one universally best algorithm. Different model families impose different assumptions about the relationship between predictors and the target. A rigorous workflow therefore compares reasonable alternatives under the same data partitions and evaluation protocol.

---

## 2. Prediction is different from explanation

A predictive model is optimized primarily to make useful predictions on observations that were not used to fit the model.

This goal differs from estimating a causal relationship or obtaining a statistically interpretable coefficient.

For example, a variable can improve prediction because it is associated with the target even if it is not a causal determinant of the outcome.

Likewise, a model can predict accurately while remaining difficult to interpret mechanistically.

Three questions should therefore remain separate:

1. **Can the model predict unseen observations?**
2. **Which variables does the model use?**
3. **Why does the underlying phenomenon occur?**

Supervised machine learning addresses the first question directly. It can help investigate the second. It does not, by itself, answer the third.

---

# Part I. General supervised-learning concepts

## 3. Features, targets, estimators, and predictions

Let:

$$
X \in \mathbb{R}^{n \times p}
$$

represent a feature matrix with $n$ observations and $p$ predictors.

For classification:

$$
y_i \in \{1,2,\ldots,K\}
$$

where $K$ is the number of classes.

For regression:

$$
y_i \in \mathbb{R}
$$

A supervised algorithm estimates a function:

$$
\hat{f}: X \rightarrow y
$$

using a training dataset.

The fitted model then generates predictions:

$$
\hat{y}_i = \hat{f}(\mathbf{x}_i)
$$

for observations not used to fit the model.

The difference between the observed and predicted values is the prediction error.

---

## 4. Training, validation, and testing

Model evaluation requires data that were not used to learn the corresponding model decisions.

The conceptual roles are:

| Subset | Main purpose |
|---|---|
| TRAIN | Fit model parameters |
| VALIDATION | Choose models, thresholds, hyperparameters, or other decisions |
| TEST | Estimate final predictive performance |

Cross-validation can replace a fixed validation set for many workflows.

The TEST set should remain untouched until the principal modeling decisions have been completed.

A high TEST score is meaningful only if TEST did not influence preprocessing, feature selection, hyperparameter optimization, threshold selection, or model choice.

---

## 5. Parameters and hyperparameters

A **parameter** is learned directly from the training observations.

Examples include:

- coefficients in linear or logistic regression;
- split points in decision trees;
- support-vector coefficients in SVMs.

A **hyperparameter** controls how the learning algorithm behaves.

Examples include:

- `C` in logistic regression or SVM;
- `n_neighbors` in KNN;
- `max_depth` in a tree;
- `n_estimators` in a random forest;
- learning rate in gradient boosting.

Hyperparameters are not estimated simply by calling `fit()` once. They are selected using a validation strategy, commonly cross-validation.

---

## 6. Bias and variance

Prediction error can be understood partly through the bias-variance trade-off.

### High bias

A model can be too restricted to represent the underlying relationship.

Typical signs include:

- poor performance on TRAIN;
- similarly poor performance on validation or TEST;
- systematic residual structure.

This is often called **underfitting**.

### High variance

A model can adapt excessively to peculiarities of TRAIN.

Typical signs include:

- very strong TRAIN performance;
- substantially weaker validation or TEST performance;
- unstable performance across resamples or seeds.

This is commonly called **overfitting**.

Model complexity, regularization, sample size, dimensionality, noise, and data partitioning all influence this trade-off.

---

# Part II. Classical algorithms for classification and regression

## 7. Why compare multiple algorithm families?

Different model families view the feature space differently.

A linear model assumes that the prediction can be described through an additive linear function of the predictors after any transformations used.

A nearest-neighbor model assumes that observations close in feature space tend to have similar targets.

A tree partitions the feature space into regions.

An ensemble combines many individual models.

A kernel model can represent nonlinear decision boundaries without explicitly generating every nonlinear feature.

Consequently, comparing several model families can reveal whether predictive structure is primarily:

- linear;
- local;
- threshold-based;
- interaction-driven;
- nonlinear;
- or dependent on more complex combinations of variables.

---

# Part III. Linear models

## 8. Linear regression

Linear regression models a continuous target as:

$$
\hat{y}
=
\beta_0
+
\beta_1x_1
+
\beta_2x_2
+\cdots+
\beta_px_p
$$

The coefficients are commonly estimated by minimizing the residual sum of squares.

### Strengths

- simple;
- fast;
- highly interpretable;
- useful baseline;
- coefficients have a clear directional interpretation when predictors are suitably encoded;
- works well when relationships are approximately linear and additive.

### Limitations

- nonlinear relationships require engineered terms or transformations;
- sensitive to influential observations;
- correlated predictors can make coefficients unstable;
- extrapolation can be poor;
- coefficient interpretation depends strongly on preprocessing and feature coding.

Linear regression should usually be included as a baseline in regression tasks.

---

## 9. Ridge regression

Ridge regression adds an $L_2$ penalty:

$$
\lambda \sum_j \beta_j^2
$$

Large coefficients are penalized, producing shrinkage.

Ridge is particularly useful when:

- predictors are correlated;
- there are many features;
- coefficient stabilization is desirable.

It generally retains all features but reduces the magnitude of their coefficients.

---

## 10. Lasso regression

Lasso uses an $L_1$ penalty:

$$
\lambda \sum_j |\beta_j|
$$

The $L_1$ penalty can reduce some coefficients exactly to zero.

Therefore, Lasso combines prediction with a form of embedded feature selection.

Its selected features can be unstable when strongly correlated predictors provide similar information.

---

## 11. Elastic Net

Elastic Net combines $L_1$ and $L_2$ penalties.

It is useful when the analyst wants:

- regularization;
- some sparsity;
- greater stability than pure Lasso when predictors are correlated.

The mixing parameter determines the relative contribution of the two penalties.

---

## 12. Logistic regression

Despite its name, logistic regression is a classification algorithm.

For binary classification, it models the log-odds of the positive class as a linear function:

$$
\log\left(
\frac{P(y=1 \mid X)}
{1-P(y=1 \mid X)}
\right)
=
\beta_0 + \sum_j \beta_jx_j
$$

The result can be transformed into a probability between 0 and 1.

### Strengths

- strong and interpretable baseline;
- probabilistic output;
- computationally efficient;
- compatible with regularization;
- suitable for high-dimensional problems when regularized.

### Limitations

- the decision boundary is linear in the modeled feature space;
- nonlinear effects require feature engineering;
- coefficients can be unstable under severe collinearity;
- probability calibration can still require verification.

### Regularized logistic regression

Common penalties include:

- $L_2$;
- $L_1$;
- Elastic Net.

The hyperparameter `C` in scikit-learn is inversely related to regularization strength:

- smaller `C` → stronger regularization;
- larger `C` → weaker regularization.

---

## 13. Multiclass logistic regression

Logistic regression can be extended to more than two classes.

A common formulation estimates a score or probability for each class:

$$
P(y=k \mid X)
$$

The predicted class is usually the class with the highest predicted probability.

Multiclass coefficients can be analyzed class by class, which is useful because a feature may help distinguish one class while contributing little to another.

---

# Part IV. Instance-based methods

## 14. K-nearest neighbors

KNN predicts an observation using nearby training observations.

For classification, the predicted class is commonly derived from the neighboring labels.

For regression, the prediction is commonly the mean or weighted mean of neighboring target values.

### Important hyperparameters

- number of neighbors $k$;
- distance metric;
- weighting strategy.

### Strengths

- conceptually simple;
- nonlinear;
- few structural assumptions;
- useful baseline for local structure.

### Limitations

- strongly affected by feature scale;
- performance deteriorates in high-dimensional spaces;
- prediction can be computationally expensive;
- sensitive to irrelevant predictors and local density.

Scaling is normally essential before KNN.

---

# Part V. Support vector methods

## 15. Support vector classifier

A support vector machine seeks a decision boundary with a large margin between classes.

With a linear kernel, the decision boundary is linear.

Kernel functions allow nonlinear boundaries.

Common kernels include:

- linear;
- radial basis function, RBF;
- polynomial.

### Important hyperparameters

- `C`;
- kernel;
- `gamma` for RBF and related kernels.

### Strengths

- effective in moderately high-dimensional spaces;
- flexible through kernels;
- mathematically well-founded;
- often competitive on small or medium tabular datasets.

### Limitations

- scaling is important;
- tuning can be sensitive;
- training can become expensive for large datasets;
- probabilities are not the native SVM output and may require additional calibration procedures.

---

## 16. Support vector regression

SVR extends the support-vector principle to regression.

It attempts to fit a function while tolerating errors within an $\epsilon$-insensitive region.

Important hyperparameters commonly include:

- `C`;
- `epsilon`;
- kernel;
- `gamma`.

SVR can perform well on moderate datasets with nonlinear structure but usually requires careful scaling and hyperparameter selection.

---

# Part VI. Probabilistic classifiers

## 17. Naive Bayes

Naive Bayes classifiers apply Bayes' theorem under simplifying assumptions about feature dependencies.

Common variants include:

- Gaussian Naive Bayes;
- Multinomial Naive Bayes;
- Bernoulli Naive Bayes.

### Strengths

- very fast;
- works surprisingly well in some high-dimensional settings;
- naturally probabilistic;
- useful baseline.

### Limitations

The conditional-independence assumption is often unrealistic for biological, clinical, and other correlated tabular variables.

Naive Bayes can nevertheless be informative as a simple benchmark.

---

# Part VII. Discriminant analysis

## 18. Linear discriminant analysis

LDA models class separation under assumptions about class-conditional distributions and covariance structure.

It can simultaneously act as:

- a classifier;
- a supervised dimensionality-reduction method.

It is attractive when classes are approximately linearly separable and covariance estimates are reliable.

---

## 19. Quadratic discriminant analysis

QDA allows each class to have its own covariance matrix.

This creates nonlinear decision boundaries but requires estimating more parameters.

It can therefore become unstable when sample size is small relative to dimensionality.

---

# Part VIII. Decision trees

## 20. Decision tree classifier

A classification tree recursively partitions feature space into regions that become increasingly class-pure.

Common splitting criteria include measures derived from:

- Gini impurity;
- entropy.

Important hyperparameters include:

- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `max_features`.

### Strengths

- nonlinear;
- handles interactions naturally;
- does not generally require scaling;
- conceptually interpretable.

### Limitations

- individual trees are unstable;
- deep trees strongly overfit;
- small changes in data can produce different structures.

Trees are often more useful as components of ensembles than as standalone high-performance predictors.

---

## 21. Decision tree regressor

Regression trees partition predictor space into regions and predict a numerical value within each terminal leaf.

They naturally capture:

- thresholds;
- nonlinear relationships;
- interactions.

However, individual regression trees are also highly variable and can overfit substantially.

---

# Part IX. Bagging and randomized tree ensembles

## 22. Random forest

Random forests combine many randomized decision trees.

Each tree is typically trained from a bootstrap sample, and only a subset of features is considered at each split.

Predictions are aggregated across trees.

For classification:

- class votes or probabilities are aggregated.

For regression:

- predictions are averaged.

### Strengths

- strong general-purpose baseline;
- captures nonlinearity and interactions;
- limited preprocessing requirements;
- relatively robust;
- handles mixed predictive structures well;
- generally does not require scaling.

### Limitations

- less interpretable than a single tree;
- large forests can be computationally expensive;
- impurity-based feature importance can be misleading;
- probabilities are not guaranteed to be calibrated.

---

## 23. Extra Trees

Extremely Randomized Trees introduce additional randomization when constructing trees.

Compared with a conventional random forest, split thresholds are usually more randomized.

This can:

- reduce variance;
- increase diversity among trees;
- sometimes improve computational efficiency.

Random Forest and Extra Trees are useful complementary ensemble baselines.

---

# Part X. Boosting

## 24. AdaBoost

AdaBoost sequentially focuses attention on observations that previous weak learners handled poorly.

The final prediction combines multiple weak models.

AdaBoost is historically important and can still be useful, but modern gradient-boosting methods are often more flexible for tabular prediction.

---

## 25. Gradient boosting

Gradient boosting builds models sequentially.

Each new learner attempts to reduce the errors of the current ensemble.

Important hyperparameters commonly include:

- number of estimators;
- learning rate;
- tree depth or number of leaves;
- minimum samples per leaf;
- regularization-related parameters.

A lower learning rate commonly requires more boosting iterations.

---

## 26. Histogram-based gradient boosting

Histogram-based methods discretize continuous variables internally into bins to accelerate training and improve scalability.

Scikit-learn provides histogram-based gradient boosting estimators for classification and regression.

These methods can be strong tabular baselines, particularly for medium and large datasets.

---

## 27. XGBoost

XGBoost is a widely used gradient-boosted tree implementation with extensive regularization and optimization options.

It supports:

- binary classification;
- multiclass classification;
- regression;
- sample weighting;
- early stopping;
- custom objectives and evaluation metrics.

Important controls include:

- number of trees;
- learning rate;
- maximum depth;
- row and feature subsampling;
- $L_1$ and $L_2$ regularization.

---

## 28. LightGBM

LightGBM is another high-performance gradient-boosting implementation.

It is designed for efficient training, including large tabular datasets.

Its tree-growth strategy can produce highly expressive models but may require careful regularization on small datasets.

---

## 29. CatBoost

CatBoost is a gradient-boosting framework designed in part to handle categorical predictors effectively.

It can reduce the need for manual one-hot encoding in suitable workflows and includes strategies intended to limit target leakage during categorical-statistic construction.

When using native categorical support, raw categorical information should be preserved rather than unnecessarily converted into arbitrary numerical codes.

---

# Part XI. Classification outputs

## 30. Predicted classes and predicted probabilities

A classifier can produce more than one type of output.

### Predicted class

```text
Alzheimer
No Alzheimer
```

or:

```text
healthy_sleep
insomnia
sleep_apnea
circadian_disruption
```

### Predicted probability

For binary classification:

$$
P(y=1 \mid X)
$$

For multiclass classification:

$$
P(y=k \mid X)
$$

for every class $k$.

Probabilities contain substantially more information than the final predicted label.

---

## 31. Decision thresholds

For binary classification, a probability can be converted to a class through a threshold.

For example:

```text
if P(positive) >= 0.50:
    predict positive
else:
    predict negative
```

The value `0.50` is not a universal law.

Changing the threshold changes:

- sensitivity;
- specificity;
- precision;
- false-positive rate;
- false-negative rate.

Threshold selection should therefore reflect the scientific or operational cost of different errors.

The threshold must not be selected by optimizing performance on the final TEST set.

---

# Part XII. Binary classification metrics

## 32. Confusion matrix

A binary confusion matrix contains:

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actual positive | TP | FN |
| Actual negative | FP | TN |

From these four counts, many classification metrics are derived.

---

## 33. Accuracy

$$
\text{Accuracy}
=
\frac{TP+TN}
{TP+TN+FP+FN}
$$

Accuracy measures the proportion of correct predictions.

### Problem

Accuracy can be misleading under class imbalance.

If 95% of observations belong to one class, predicting that class for every observation gives 95% accuracy while completely failing on the minority class.

---

## 34. Balanced accuracy

For binary classification, balanced accuracy is the average recall across classes.

It gives equal importance to each class rather than each observation.

This makes it useful when class frequencies differ substantially.

---

## 35. Sensitivity or recall

$$
\text{Recall}
=
\frac{TP}{TP+FN}
$$

Recall asks:

> Of the truly positive observations, how many did the model detect?

High recall is important when missing positives is costly.

---

## 36. Specificity

$$
\text{Specificity}
=
\frac{TN}{TN+FP}
$$

Specificity asks:

> Of the truly negative observations, how many did the model correctly reject?

---

## 37. Precision

$$
\text{Precision}
=
\frac{TP}{TP+FP}
$$

Precision asks:

> Among observations predicted positive, how many are actually positive?

Precision depends on class prevalence and therefore should be interpreted in the context of the evaluated population.

---

## 38. F1 score

$$
F_1
=
2
\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

F1 balances precision and recall.

It ignores true negatives, so it should not automatically replace other metrics.

---

## 39. $F_\beta$

$F_\beta$ generalizes F1.

A value $\beta>1$ gives greater importance to recall.

A value $\beta<1$ gives greater importance to precision.

This is useful when the costs of false positives and false negatives are asymmetric.

---

## 40. Matthews correlation coefficient

MCC incorporates all four cells of the confusion matrix.

For binary classification:

$$
MCC
=
\frac{TP\times TN-FP\times FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

It ranges approximately from:

```text
-1   complete disagreement
 0   no better than random association
+1   perfect classification
```

MCC is particularly useful for imbalanced binary classification because it considers all parts of the confusion matrix.

---

## 41. ROC curve

The ROC curve evaluates the classifier across many thresholds.

It plots:

$$
TPR
$$

against:

$$
FPR
$$

where:

$$
FPR = 1-\text{Specificity}
$$

---

## 42. ROC-AUC

ROC-AUC summarizes ranking performance across thresholds.

A useful interpretation is:

> What is the probability that a randomly chosen positive observation receives a higher score than a randomly chosen negative observation?

Typical reference values are:

```text
0.5  random ranking
1.0  perfect ranking
```

AUC does not describe calibration and does not identify the optimal operational threshold.

---

## 43. Precision-recall curve

A precision-recall curve shows how precision and recall trade off across thresholds.

It is particularly informative when the positive class is rare.

---

## 44. Average precision and PR-AUC

Average precision summarizes a precision-recall relationship.

Unlike ROC-AUC, the baseline depends on positive-class prevalence.

For rare-event detection, precision-recall analysis is often more informative than accuracy alone.

---

## 45. Log loss

Log loss evaluates predicted probabilities rather than only predicted classes.

For binary classification:

$$
-\frac{1}{n}
\sum_i
[
y_i \log(p_i)
+
(1-y_i)\log(1-p_i)
]
$$

Confident incorrect predictions are penalized heavily.

Lower values are better.

---

## 46. Brier score

For binary classification:

$$
\text{Brier}
=
\frac{1}{n}
\sum_i
(p_i-y_i)^2
$$

The Brier score evaluates probabilistic predictions.

It reflects both discrimination and calibration-related properties.

Lower is better.

---

# Part XIII. Calibration

## 47. Discrimination versus calibration

A model can rank cases correctly while producing poor probabilities.

For example, predictions of:

```text
0.60
0.70
0.80
0.90
```

may rank four observations correctly while being systematically overconfident or underconfident.

### Discrimination

Can the model separate classes?

Typical metrics:

- ROC-AUC;
- PR-AUC;
- MCC at a threshold.

### Calibration

Do predicted probabilities correspond to observed frequencies?

For example, among cases assigned probability near 0.8, approximately 80% should be positive in a well-calibrated system.

Useful tools include:

- calibration curves;
- Brier score;
- calibration intercept and slope;
- Platt scaling;
- isotonic calibration.

Calibration should be estimated using data not used to fit the underlying predictor.

---

# Part XIV. Multiclass classification metrics

## 48. Multiclass confusion matrix

For $K$ classes, the confusion matrix is $K \times K$.

Rows commonly represent actual classes and columns predicted classes.

The matrix helps identify specific confusions, for example:

```text
insomnia -> circadian_disruption
sleep_apnea -> healthy_sleep
```

A single aggregate metric cannot reveal these patterns.

---

## 49. Per-class metrics

Each class can be treated as:

```text
class k
vs
all other classes
```

to calculate:

- precision;
- recall;
- F1;
- specificity where relevant.

Per-class metrics are particularly important when some classes are more difficult or less common.

---

## 50. Macro averaging

Macro averaging computes a metric independently for each class and then averages the class values.

Each class receives equal weight.

Macro metrics are therefore useful when performance on minority classes matters.

---

## 51. Weighted averaging

Weighted averaging weights each class by its number of observations.

Large classes contribute more strongly.

Weighted averages can therefore appear good even when a small class performs poorly.

---

## 52. Micro averaging

Micro averaging aggregates decisions across all classes before computing the metric.

For single-label multiclass classification, micro-averaged precision, recall, and F1 have relationships closely tied to overall correctness.

---

## 53. Multiclass MCC

MCC has multiclass generalizations.

It can provide a single summary of prediction-label association while considering the complete confusion structure.

---

## 54. Multiclass ROC-AUC

Multiclass ROC-AUC requires an explicit strategy, commonly:

- one-vs-rest;
- one-vs-one.

The averaging method should always be reported.

A statement such as:

```text
AUC = 0.91
```

is incomplete for multiclass prediction unless the calculation strategy is specified.

---

## 55. Multiclass log loss

Multiclass log loss evaluates the complete predicted probability distribution.

For each observation, assigning low probability to the true class is penalized.

This is useful when probabilistic multiclass predictions are important.

---

## 56. Top-k accuracy

Top-k accuracy asks whether the true class appears among the model's $k$ highest-scoring classes.

This can be useful in recommendation or prioritization systems.

For small-class-count scientific classification problems, ordinary class-level metrics are usually more directly interpretable.

---

# Part XV. Regression metrics

## 57. Residuals

For observation $i$:

$$
e_i = y_i-\hat{y}_i
$$

Residual analysis is fundamental.

A regression model should not be evaluated only through one summary number.

Useful plots include:

- observed versus predicted;
- residual versus predicted;
- residual distribution;
- absolute error versus predicted;
- residuals versus important predictors.

---

## 58. Mean absolute error

$$
MAE
=
\frac{1}{n}
\sum_i
|y_i-\hat{y}_i|
$$

MAE is measured in the same units as the target.

For crop yield:

```text
MAE = 0.45 t/ha
```

has a direct interpretation.

MAE is less sensitive to a few very large errors than RMSE.

---

## 59. Mean squared error

$$
MSE
=
\frac{1}{n}
\sum_i
(y_i-\hat{y}_i)^2
$$

Squaring errors gives large deviations greater influence.

Its units are squared target units, making direct interpretation less intuitive.

---

## 60. Root mean squared error

$$
RMSE
=
\sqrt{MSE}
$$

RMSE returns to the original target units.

Because large errors are squared before averaging, RMSE penalizes large mistakes more strongly than MAE.

Comparing MAE and RMSE can provide insight into whether a subset of large errors dominates performance.

---

## 61. Median absolute error

Median absolute error is robust to extreme residuals.

It answers approximately:

> What is the typical absolute prediction error for the median observation?

It can complement MAE and RMSE when error distributions contain outliers.

---

## 62. Coefficient of determination

$$
R^2
=
1
-
\frac{
\sum_i(y_i-\hat{y}_i)^2
}{
\sum_i(y_i-\bar{y})^2
}
$$

Interpretation:

```text
R² = 1     perfect predictions
R² = 0     comparable to predicting the evaluation-set mean under the standard definition
R² < 0     worse than that reference
```

A negative $R^2$ is possible and should not be artificially clipped.

High $R^2$ does not guarantee small errors in scientifically meaningful units.

---

## 63. Explained variance

Explained variance measures how much variability in the target errors is accounted for.

It is related to but not identical to $R^2$.

For most practical regression benchmarking, MAE, RMSE, and $R^2$ form a more immediately interpretable core.

---

## 64. Mean absolute percentage error

MAPE expresses errors relative to the observed target:

$$
MAPE
=
\frac{100}{n}
\sum_i
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
$$

### Important limitation

MAPE becomes unstable or undefined when true targets approach zero.

It can also weight errors asymmetrically.

Therefore, it should not be used automatically simply because percentages appear intuitive.

---

## 65. Alternative relative-error metrics

Depending on the application, alternatives can include:

- symmetric MAPE;
- normalized RMSE;
- mean absolute scaled error;
- domain-specific percentage or tolerance metrics.

Every normalization introduces a reference scale that must be reported clearly.

---

## 66. Rank-based evaluation

Some applications care more about prioritization than exact numerical prediction.

Spearman correlation between $y$ and $\hat{y}$ can quantify ranking agreement.

However:

```text
high Spearman correlation
```

does not imply:

```text
small prediction error
```

A model can rank observations correctly while being numerically biased.

Ranking metrics should therefore complement rather than replace absolute-error metrics unless ranking is the explicit task.

---

# Part XVI. Selecting metrics for a task

## 67. There is no universally best metric

The metric should correspond to the decision being made.

### Binary classification

Possible core set:

```text
MCC
balanced accuracy
ROC-AUC
average precision
precision
recall
specificity
F1
log loss or Brier score when probabilities matter
```

### Multiclass classification

Possible core set:

```text
macro F1
balanced accuracy
multiclass MCC
per-class precision / recall / F1
confusion matrix
multiclass log loss when probabilities matter
```

### Regression

Possible core set:

```text
MAE
RMSE
R²
residual diagnostics
```

Additional metrics should be selected according to the scientific problem.

---

## 68. Primary and secondary metrics

A study should ideally define a **primary metric** before model comparison.

For example:

```text
Primary metric:
MCC

Secondary:
balanced accuracy
ROC-AUC
average precision
F1
```

or:

```text
Primary metric:
MAE

Secondary:
RMSE
R²
```

This reduces the temptation to select the model based on whichever metric happens to look best after evaluation.

---

# Part XVII. Interpretation beyond a single score

## 69. Point estimate versus uncertainty

A reported metric such as:

```text
MCC = 0.74
```

is only a point estimate.

Model performance varies because of:

- the observations included in each split;
- random initialization;
- bootstrap sampling;
- feature subsampling;
- model stochasticity.

Whenever possible, report uncertainty or variability using:

- cross-validation distributions;
- repeated cross-validation;
- bootstrap confidence intervals;
- repeated seeds where scientifically appropriate.

---

## 70. Performance distributions

Comparing:

```text
Model A MCC = 0.82
Model B MCC = 0.81
```

does not establish that Model A is meaningfully superior.

Across repeated evaluations one model may be:

- more stable;
- less variable;
- more robust to difficult splits;
- slightly weaker on average but substantially safer.

Model selection should therefore consider distributions rather than only the best observed score.

---

## 71. Baselines

Every predictive task should include meaningful baselines.

### Classification

Examples include:

- majority-class prediction;
- stratified random prediction;
- regularized logistic regression.

### Regression

Examples include:

- predicting the training mean;
- predicting the training median;
- regularized linear regression.

A complex model is useful only if it improves meaningfully over an appropriate reference.

---

# Part XVIII. Interpretability

## 72. Model-specific interpretation

Different algorithms provide different native quantities.

### Linear models

Coefficients.

### Logistic regression

Coefficients or odds-ratio-related interpretations after accounting for feature scaling and coding.

### Trees

Tree structure and split rules.

### Tree ensembles

Impurity-based importance is available but should be interpreted cautiously.

### KNN

Neighbor structure.

### SVM

Linear coefficients for linear kernels; nonlinear kernels are harder to interpret directly.

---

## 73. Permutation importance

Permutation importance measures the reduction in predictive performance after a feature is randomly permuted.

Conceptually:

```text
good predictions
    ↓
shuffle one feature
    ↓
measure performance decrease
```

Large performance loss suggests the fitted model relied on that feature.

### Caveat

Correlated predictors can share information, causing individual permutation importance values to underestimate their joint relevance.

---

## 74. Partial dependence and related tools

Partial dependence evaluates how model predictions change as a feature varies while averaging over other observations.

Related approaches include:

- individual conditional expectation;
- accumulated local effects.

These methods describe the fitted model, not necessarily the causal effect of a variable.

---

## 75. SHAP and local explanations

SHAP-based methods can attribute a prediction to feature contributions under a specific mathematical framework.

They are useful for:

- global summaries;
- local case explanations;
- interaction exploration.

However, SHAP values:

- explain model behavior;
- depend on the fitted model;
- depend on the background/reference distribution;
- do not establish biological or causal mechanisms.

---

# Part XIX. Practical algorithm comparison

## 76. A reasonable classical classification benchmark

A compact comparison can include:

```text
Logistic Regression
KNN
SVM
Random Forest
Extra Trees
Gradient Boosting
```

Optionally:

```text
XGBoost
LightGBM
CatBoost
```

according to dependencies and teaching scope.

This provides:

- linear;
- local;
- margin-based;
- bagging;
- boosting

families without testing dozens of nearly redundant estimators.

---

## 77. A reasonable classical regression benchmark

A compact comparison can include:

```text
Linear Regression
Ridge / Elastic Net
KNN Regressor
SVR
Random Forest Regressor
Extra Trees Regressor
Gradient Boosting Regressor
```

with optional external boosting frameworks.

---

## 78. Scaling requirements differ

Models commonly sensitive to scale include:

```text
KNN
SVM
linear / logistic models with regularization
many optimization-based models
```

Tree-based models generally do not require feature standardization for split construction.

This does not mean preprocessing should differ casually between benchmarked models.

Each model should receive a valid preparation strategy, and comparisons should be reproducible.

---

## 79. Categorical predictors

Classical algorithms differ in how they accept categorical predictors.

Many scikit-learn estimators require numerical matrices.

Common approaches include:

- one-hot encoding;
- ordinal encoding when truly ordinal;
- carefully validated alternative encodings.

Some boosting frameworks support categorical variables natively.

Arbitrary integer coding such as:

```text
red = 1
blue = 2
green = 3
```

creates an artificial numerical ordering if the categories are nominal.

---

# Part XX. Model selection principles

## 80. Do not choose the best TRAIN score

Training performance measures how well the model describes data it already saw.

It is not an estimate of generalization.

---

## 81. Do not choose the best TEST score

Repeatedly checking TEST and modifying the workflow converts TEST into a validation set.

The final TEST estimate then becomes optimistically biased.

---

## 82. Choose using validation information

Model choice should be based on:

- fixed validation data;
- cross-validation within TRAIN;
- nested validation strategies.

Once the modeling process is frozen, TEST is evaluated once for final performance estimation.

---

## 83. Performance is not the only criterion

When models perform similarly, relevant secondary considerations include:

- stability;
- calibration;
- inference speed;
- training cost;
- model size;
- interpretability;
- number of required features;
- missing-data tolerance;
- deployment constraints.

The most complex model is not automatically the best scientific choice.

---

# Part XXI. Reporting supervised-learning results

## 84. Classification reporting checklist

A strong report should state:

- prediction task;
- class definitions;
- sample counts;
- splitting strategy;
- preprocessing strategy;
- model family;
- hyperparameter-selection procedure;
- primary metric;
- secondary metrics;
- confusion matrix;
- class-specific performance;
- probability metrics when relevant;
- threshold-selection procedure;
- uncertainty or repeated-evaluation results;
- final TEST performance.

---

## 85. Regression reporting checklist

Report:

- target definition and units;
- sample counts;
- splitting strategy;
- preprocessing;
- model family;
- hyperparameter selection;
- MAE;
- RMSE;
- $R^2$;
- residual diagnostics;
- uncertainty or repeated-evaluation results;
- final TEST performance.

If predictions will be used for prioritization, rank-based metrics can be added explicitly.

---

# Part XXII. Key messages

1. Supervised learning estimates mappings from predictors to known targets.
2. Binary classification, multiclass classification, and regression require different evaluation logic.
3. No model family is universally best.
4. Linear models remain important baselines.
5. KNN and SVM are sensitive to feature scale.
6. Trees naturally model thresholds and interactions.
7. Random forests and Extra Trees reduce the instability of individual trees.
8. Gradient boosting is one of the strongest general-purpose approaches for tabular prediction.
9. Classification should not be summarized by accuracy alone.
10. Multiclass metrics require explicit averaging strategies.
11. Regression should not be summarized by $R^2$ alone.
12. Probability discrimination and calibration are different properties.
13. Threshold choice is a decision problem, not an intrinsic property of the classifier.
14. Performance distributions are more informative than isolated best scores.
15. TEST should estimate final performance, not guide model development.
16. Interpretation tools describe model behavior and should not be confused with causal explanation.
17. Classical supervised learning works best when embedded in a reproducible, leakage-safe evaluation framework.
