# Hyperparameter Optimization and Model Selection

## 1. Purpose

Training a machine-learning algorithm requires more than choosing a model family.

Most algorithms contain **hyperparameters** that control:

- model complexity;
- regularization;
- learning behavior;
- neighborhood size;
- tree structure;
- optimization;
- stochastic sampling;
- computational cost.

Hyperparameter optimization searches for configurations that generalize well to observations not used to fit the corresponding configuration.

This document focuses on the decisions required to optimize models correctly and select a final model without contaminating the final evaluation.

It assumes that preprocessing, feature engineering, scaling, dataset partitioning, and end-to-end data preparation have already been introduced separately.

---

# Part I. What is being optimized?

## 2. Parameters versus hyperparameters

### Model parameters

Learned directly during fitting.

Examples:

```text
linear-regression coefficients
logistic-regression coefficients
tree split points
support vectors
```

### Hyperparameters

Specified outside the fitting process.

Examples:

```text
KNN n_neighbors
logistic-regression C
SVM C and gamma
tree max_depth
random-forest n_estimators
boosting learning_rate
```

Hyperparameter optimization attempts to identify useful hyperparameter values according to a predefined evaluation criterion.

---

## 3. The objective of optimization

Hyperparameter optimization is not simply:

> Find the configuration with the largest number.

The actual objective is closer to:

> Identify a configuration that is expected to generalize well under the intended prediction scenario.

This distinction matters because the highest observed validation score can result from:

- random sampling variation;
- an unusually favorable fold;
- excessive searching;
- overfitting to the validation procedure;
- a noisy metric.

Optimization should therefore consider both performance and reliability.

---

## 4. Define the primary metric first

A search procedure requires an objective.

Examples:

### Binary classification

```text
MCC
balanced accuracy
ROC-AUC
average precision
log loss
```

### Multiclass classification

```text
macro F1
balanced accuracy
multiclass MCC
multiclass log loss
```

### Regression

```text
MAE
RMSE
R²
```

The primary metric should reflect the scientific or operational objective.

Secondary metrics remain important but should not be used opportunistically to redefine the winner after seeing results.

---

## 5. Metric direction

Some metrics are maximized:

```text
MCC
F1
ROC-AUC
R²
```

Others are minimized:

```text
MAE
RMSE
log loss
Brier score
```

Optimization software may convert minimization metrics into negative scorers.

For example, scikit-learn commonly uses names such as:

```text
neg_mean_absolute_error
neg_root_mean_squared_error
```

The sign convention must be interpreted correctly.

---

# Part II. Which data may be used?

## 6. TRAIN is where model development happens

Hyperparameter optimization belongs inside the model-development data.

Conceptually:

```text
TRAIN
  ↓
cross-validation / internal validation
  ↓
hyperparameter search
  ↓
selected configuration
```

VALIDATION may be used for additional model-selection decisions when the workflow includes a fixed validation set.

TEST should remain untouched.

---

## 7. TEST must not optimize hyperparameters

The following workflow is invalid:

```text
configuration A -> TEST score
configuration B -> TEST score
configuration C -> TEST score
choose best TEST score
```

TEST has now become part of the optimization procedure.

The reported TEST performance is no longer an unbiased estimate of final generalization.

---

## 8. Cross-validation as the optimization engine

Rather than judging a configuration from one validation split, cross-validation evaluates it across several training/validation partitions.

For each configuration:

```text
fold 1 -> score
fold 2 -> score
fold 3 -> score
...
```

The search then summarizes performance across folds.

This gives information about:

- mean performance;
- variance;
- worst-case folds;
- stability.

---

## 9. Classification cross-validation

For classification, stratified cross-validation is commonly preferred when the class label can be preserved across folds.

Examples include:

```text
StratifiedKFold
RepeatedStratifiedKFold
```

Rare classes may limit the number of feasible folds.

A class containing fewer than $k$ observations cannot be represented in every fold of a $k$-fold scheme.

---

## 10. Regression cross-validation

For regression, common strategies include:

```text
KFold
RepeatedKFold
```

Continuous targets cannot be directly stratified in the same way as categorical labels.

Temporary target quantile bins can sometimes be used for approximate stratification, but this is an evaluation design choice rather than a universal requirement.

---

## 11. Groups and dependencies override convenience

If observations belong to:

- patients;
- families;
- sites;
- compounds;
- clusters;
- homologous sequence groups;
- experimental batches;
- repeated measurements,

ordinary random CV may leak related information across folds.

Group-aware, cluster-aware, temporal, spatial, distance-aware, or homology-aware validation may be more appropriate.

Hyperparameter optimization must use the same scientifically defensible splitting logic as final evaluation.

---

# Part III. Search spaces

## 12. Search-space design is part of modeling

A search algorithm can only explore values that the researcher defines.

A poorly designed search space can:

- waste computation;
- miss useful regions;
- include invalid configurations;
- encourage severe overfitting.

Search spaces should be scientifically and computationally reasonable.

---

## 13. Categorical hyperparameters

Examples:

```text
kernel ∈ {linear, rbf}
criterion ∈ {gini, entropy}
class_weight ∈ {None, balanced}
```

These are sampled from discrete alternatives.

---

## 14. Integer hyperparameters

Examples:

```text
n_neighbors
max_depth
min_samples_leaf
n_estimators
```

Their valid range should reflect sample size and computational constraints.

Searching:

```text
n_neighbors = 1 ... 1000
```

makes little sense for a dataset containing 300 training observations.

---

## 15. Continuous hyperparameters

Examples include:

```text
learning_rate
subsample
l1_ratio
```

Continuous sampling can explore them more efficiently than coarse manually selected grids.

---

## 16. Logarithmic scales

Some hyperparameters vary meaningfully across orders of magnitude.

Examples:

```text
C
gamma
regularization strength
learning rate in some settings
```

Searching:

```text
0.0001
0.001
0.01
0.1
1
10
100
```

is often more reasonable than a linear search:

```text
1
2
3
4
5
...
```

This is commonly called a log-uniform search.

---

## 17. Conditional search spaces

Some hyperparameters are relevant only when another choice is active.

Example:

```text
kernel = linear
    gamma is irrelevant

kernel = rbf
    gamma matters
```

Optimization frameworks such as Optuna can express conditional search spaces naturally.

---

# Part IV. Common optimization strategies

## 18. Manual tuning

The analyst chooses a small number of configurations based on knowledge and previous results.

### Advantages

- simple;
- transparent;
- useful for educational demos;
- useful for quick sanity checks.

### Limitations

- subjective;
- poor coverage;
- difficult to reproduce rigorously if decisions are made interactively.

Manual tuning is useful during exploration but should not replace a formal optimization protocol in a benchmark.

---

## 19. Grid search

Grid search evaluates every combination in a predefined Cartesian grid.

Example:

```text
C = [0.1, 1, 10]
penalty = [l1, l2]
```

produces up to six combinations, subject to estimator compatibility.

### Advantages

- deterministic;
- transparent;
- easy to explain;
- appropriate for small low-dimensional search spaces.

### Limitations

The number of trials increases multiplicatively.

For:

```text
5 × 5 × 5 × 5
```

the search already contains:

```text
625 configurations
```

before cross-validation folds are considered.

Grid search is inefficient when only a few hyperparameters strongly influence performance.

---

## 20. Random search

Random search samples configurations from distributions.

Instead of exhaustively testing a grid, a computational budget is specified:

```text
100 random configurations
```

### Advantages

- efficient for large search spaces;
- allows continuous distributions;
- often finds good configurations faster than grid search;
- simple and highly parallelizable.

### Key idea

If only a few hyperparameters are important, random search explores more distinct values of those hyperparameters than a rigid grid with the same number of evaluations.

---

## 21. Bayesian optimization

Bayesian optimization builds a surrogate model of:

```text
hyperparameter configuration -> expected performance
```

and uses previous evaluations to decide which configuration to evaluate next.

The objective is to balance:

- exploration of uncertain regions;
- exploitation of promising regions.

Bayesian optimization is particularly useful when each model evaluation is expensive.

---

## 22. Gaussian-process optimization

Gaussian processes are a classical surrogate model for Bayesian optimization.

They work particularly well for:

- relatively low-dimensional search spaces;
- expensive objectives;
- smooth response surfaces.

Their computational cost can become limiting as the number of evaluated trials grows.

---

## 23. Tree-structured Parzen estimator

TPE models distributions of promising and less promising hyperparameter values and proposes new candidates accordingly.

TPE is widely used in modern hyperparameter optimization and supports:

- categorical variables;
- continuous variables;
- conditional spaces;
- pruning strategies.

Optuna commonly uses a TPE-based sampler by default for many workflows.

---

## 24. Evolutionary algorithms

Evolutionary optimization represents candidate configurations as a population.

Typical operations include:

- mutation;
- recombination;
- selection.

Evolutionary methods can explore complex or irregular spaces but can require substantial computational budgets.

They are less commonly necessary for routine classical tabular benchmarks.

---

# Part V. Resource-aware optimization

## 25. Successive halving

Successive halving starts with many candidate configurations using limited resources.

Poor candidates are removed, while promising candidates receive more resources.

The resource can represent:

- number of training samples;
- number of boosting iterations;
- another incremental training budget.

This reduces computation spent on clearly weak candidates.

---

## 26. Hyperband

Hyperband combines multiple successive-halving strategies with different resource allocations.

It is useful when model performance can be assessed meaningfully using partial training.

---

## 27. Early stopping

Many iterative models can stop when validation performance stops improving.

Examples include gradient boosting implementations.

Conceptually:

```text
iteration 1
iteration 2
...
best validation score at iteration 147
...
no improvement for patience window
STOP
```

Early stopping acts as both:

- computational optimization;
- regularization.

Its validation data must still follow leakage-safe evaluation design.

---

## 28. Pruning optimization trials

Optimization frameworks can stop unpromising trials before completion.

For expensive boosting or neural models, this can save substantial computation.

Pruning requires an intermediate metric that is meaningful during training.

---

# Part VI. Common software

## 29. scikit-learn

Scikit-learn provides:

```text
GridSearchCV
RandomizedSearchCV
HalvingGridSearchCV
HalvingRandomSearchCV
```

Its major advantage is close integration with:

- estimators;
- pipelines;
- preprocessing;
- cross-validation;
- scoring.

For classical teaching workflows, scikit-learn search utilities provide an excellent starting point.

---

## 30. Optuna

Optuna is a general hyperparameter-optimization framework.

Typical strengths include:

- TPE sampling;
- pruning;
- conditional spaces;
- persistent studies;
- visualization;
- parallel optimization;
- custom objectives.

An objective function typically:

1. samples hyperparameters;
2. builds the estimator;
3. evaluates it with cross-validation;
4. returns the optimization metric.

---

## 31. scikit-optimize

`scikit-optimize` is commonly encountered in Python workflows for sequential model-based optimization and Bayesian search.

It provides interfaces designed to work with scikit-learn-style estimators.

When choosing a library, project maintenance status and environment compatibility should always be checked.

---

## 32. Ray Tune

Ray Tune supports distributed experiment execution and hyperparameter search.

It is useful when:

- many trials are expensive;
- multiple CPUs or GPUs are available;
- distributed execution is needed.

For small classroom tabular datasets, it can be unnecessary operational complexity.

---

## 33. Native optimization interfaces

Libraries such as:

```text
XGBoost
LightGBM
CatBoost
```

include their own training controls, validation handling, and early-stopping mechanisms.

These can be integrated with broader optimization frameworks.

---

# Part VII. Preprocessing must be inside the search

## 34. Why a frozen preprocessed matrix can be dangerous

Suppose feature selection is fitted once using all TRAIN observations:

```text
TRAIN
  ↓
feature selection
  ↓
cross-validation
```

The validation folds have already influenced the selected features.

This introduces leakage between folds.

The correct structure is:

```text
CV fold
  ↓
fold-specific training subset
  ↓
fit preprocessing
fit feature selection
fit scaling
fit model
  ↓
fold-specific validation subset
```

This is one of the principal reasons to integrate preprocessing and the estimator in a scikit-learn `Pipeline` during hyperparameter optimization.

---

## 35. What can be outside the CV loop?

Operations that do not learn information from the observations can usually occur before CV.

Examples include:

- deterministic column renaming;
- deterministic unit conversion;
- deterministic feature formulas whose definition does not depend on dataset statistics.

Operations that estimate something from data normally belong inside the training fold.

Examples include:

- imputation statistics;
- scaling parameters;
- target encoding;
- feature selection;
- PCA fitted from data;
- supervised dimensionality reduction.

---

# Part VIII. Joint optimization of preprocessing and model

## 36. Preprocessing can be hyperparameterized

The search can include choices such as:

```text
scaler = StandardScaler or RobustScaler
number of selected features = 10, 20, 30
model C = ...
```

This allows the workflow to evaluate combinations.

However, the search space should remain controlled.

Searching every conceivable preprocessing option creates:

- huge computational cost;
- additional selection bias;
- harder interpretation.

Only scientifically reasonable alternatives should be optimized.

---

# Part IX. Classification-specific considerations

## 37. Class imbalance and optimization metric

Optimizing accuracy on an imbalanced dataset can favor majority-class predictions.

Possible alternatives include:

```text
balanced accuracy
MCC
macro F1
average precision
```

The optimization metric should correspond to the objective of the classifier.

---

## 38. Threshold optimization is separate

Hyperparameter optimization usually operates on model-level parameters.

Decision-threshold selection is another decision.

For example:

```text
hyperparameters selected by CV
        ↓
probabilities on validation data
        ↓
choose threshold
        ↓
freeze model + threshold
        ↓
evaluate TEST
```

The threshold must not be chosen from TEST.

---

## 39. Multiclass objectives

A multiclass model should use an objective that reflects class-level performance.

For example:

```text
macro F1
```

weights classes equally.

```text
weighted F1
```

weights large classes more heavily.

The choice can materially change which hyperparameters appear optimal.

---

# Part X. Regression-specific considerations

## 40. Optimize the metric that matters

If the practical concern is typical absolute prediction error, optimizing:

```text
MAE
```

can be more meaningful than optimizing $R^2$.

If large errors are particularly costly, RMSE may be more aligned with the objective.

---

## 41. Target transformations

If the target is transformed during model fitting, predictions should be returned to original units before final scientific interpretation.

Target transformations must be fitted and applied consistently within the training workflow.

---

# Part XI. Nested cross-validation

## 42. The model-selection problem

Suppose we use cross-validation to search 500 configurations and report the best CV score.

That score was also used to select the winning configuration.

It can therefore be optimistically biased.

---

## 43. Nested CV

Nested cross-validation separates:

- model selection;
- performance estimation.

Conceptually:

```text
OUTER fold
    ├── OUTER TRAIN
    │      ↓
    │   INNER CV
    │      ↓
    │   hyperparameter optimization
    │      ↓
    │   selected configuration
    │
    └── OUTER TEST
           ↓
       unbiased fold evaluation
```

The process repeats over outer folds.

Nested CV is particularly valuable when:

- the dataset is small;
- model comparison is extensive;
- an independent TEST set is unavailable;
- unbiased model-selection performance is important.

---

## 44. Cost of nested CV

If:

```text
5 outer folds
×
5 inner folds
×
100 configurations
```

then up to:

```text
2500 model fits
```

may be required for one algorithm family, before repetitions.

The statistical benefit must therefore be balanced against computational cost.

---

# Part XII. Selecting a model after optimization

## 45. Do not select only by rank 1

Suppose:

| Configuration | Mean CV MCC | SD |
|---|---:|---:|
| A | 0.802 | 0.061 |
| B | 0.800 | 0.021 |
| C | 0.798 | 0.019 |

Configuration A has the highest mean, but its variability is much larger.

The differences in means may not be meaningful.

Selection can consider:

- average performance;
- variability;
- worst-fold behavior;
- complexity;
- calibration;
- computational cost.

---

## 46. The one-standard-error principle

A classical selection strategy chooses the simplest model whose performance falls within approximately one standard error of the best model.

The general idea is:

> Prefer a simpler model when evidence for the more complex alternative is weak.

This can improve robustness and interpretability.

---

## 47. Parsimony

If two models perform essentially equivalently, prefer the one that is:

- simpler;
- faster;
- easier to reproduce;
- easier to deploy;
- easier to interpret.

This principle is especially valuable in small-data settings where tiny validation-score differences are noisy.

---

## 48. Stability across folds

A candidate that performs consistently across folds can be preferable to one that alternates between excellent and poor performance.

Useful summaries include:

```text
mean
median
standard deviation
interquartile range
minimum
maximum
```

Plotting the complete fold-level performance distribution is preferable to reporting only the mean.

---

# Part XIII. Multiple algorithms

## 49. Hyperparameter optimization does not make algorithms directly comparable by itself

Each model family should receive:

- a reasonable search space;
- a comparable validation protocol;
- the same primary metric;
- the same data partitions;
- leakage-safe preprocessing.

Giving one algorithm 1000 optimization trials and another three manual settings produces an unfair comparison.

---

## 50. Equal compute versus equal scientific effort

Strictly equal trial counts are not always necessary because search spaces differ.

However, computational budgets should be defensible.

Possible designs include:

```text
same maximum wall time
same approximate number of fits
algorithm-specific but predeclared search budgets
```

The chosen policy should be documented.

---

# Part XIV. Over-optimization

## 51. Validation overfitting

Repeated experimentation against the same validation set eventually adapts the research process to that set.

This can happen even when TEST remains untouched.

Examples:

```text
try feature set
check validation
modify preprocessing
check validation
change algorithm
check validation
repeat 50 times
```

The validation set has now become part of the development feedback loop.

Cross-validation, nested CV, locked TEST sets, and predeclared model-selection rules help reduce this problem.

---

## 52. Search size and winner's curse

When many configurations are evaluated, some will achieve high validation scores partly by chance.

Selecting the maximum can therefore produce an optimistic winner.

This is one reason to examine:

- performance distributions;
- stability;
- repeated evaluation;
- independent TEST performance.

---

# Part XV. Reproducibility

## 53. Random seeds

Record seeds for:

- splits;
- stochastic model initialization;
- random search;
- optimization samplers;
- subsampling.

A seed does not guarantee perfect reproducibility across every platform or library version, but it is an essential component.

---

## 54. Persist the search results

Do not save only:

```text
best_params.json
```

Also retain trial-level or candidate-level results such as:

```text
configuration
fold scores
mean score
score variance
runtime
status
seed
```

This makes the optimization auditable.

---

## 55. Save the complete selected model

After selecting a configuration, refit the full training workflow on the intended development data and serialize the resulting pipeline.

The saved object should include the preprocessing required for future observations.

---

# Part XVI. Suggested workflow

## 56. Fixed TEST workflow

A common practical structure is:

```text
DATASET
   ↓
TRAIN / TEST
   ↓
TRAIN
   ↓
cross-validation
   ↓
hyperparameter optimization
   ↓
select configuration
   ↓
refit on all TRAIN
   ↓
TEST once
```

---

## 57. TRAIN / VALIDATION / TEST workflow

Another structure is:

```text
DATASET
   ↓
TRAIN / VALIDATION / TEST
   ↓
TRAIN
   ↓
CV hyperparameter optimization
   ↓
candidate models
   ↓
VALIDATION
   ↓
final model / threshold decision
   ↓
refit according to predefined protocol
   ↓
TEST once
```

The precise refit policy should be specified before final evaluation.

---

## 58. Small-data workflow

When data are scarce:

```text
DATASET
   ↓
development data + protected TEST
   ↓
repeated / nested CV inside development data
```

or, if a protected TEST is not feasible:

```text
nested cross-validation
```

can provide more efficient use of observations.

---

# Part XVII. Reporting optimization

## 59. Minimum information to report

A reproducible optimization description should include:

- algorithm;
- search method;
- hyperparameter search space;
- sampling distributions;
- number of trials;
- random seed;
- cross-validation strategy;
- number of folds;
- grouping constraints;
- preprocessing inside CV;
- primary optimization metric;
- handling of failed trials;
- early-stopping rule if used;
- selection rule;
- final selected hyperparameters;
- final evaluation procedure.

---

# Part XVIII. Key messages

1. Hyperparameter optimization estimates useful configurations; it does not merely maximize an arbitrary score.
2. TEST must not participate in tuning.
3. The optimization metric should be defined before the search.
4. Cross-validation is commonly the internal evaluation engine.
5. Preprocessing learned from data must occur inside each CV training fold.
6. Grid search is transparent but scales poorly.
7. Random search is a strong general-purpose default for large spaces.
8. Bayesian and TPE-based approaches can allocate expensive evaluations more intelligently.
9. Successive halving, early stopping, and pruning can reduce wasted computation.
10. Nested CV separates model selection from performance estimation.
11. The highest mean validation score is not automatically the most reliable model.
12. Stability, simplicity, calibration, and computational cost can influence final model selection.
13. Optimization itself can overfit when many configurations are tested.
14. Full trial histories should be retained, not only the winner.
15. A final fitted workflow should include the preprocessing required for future data.
