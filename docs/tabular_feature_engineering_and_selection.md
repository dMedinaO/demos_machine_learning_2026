# Feature Engineering and Feature Selection for Tabular Data

## 1. Purpose of this document

After a dataset has been preprocessed and explored, the next question is often not yet **which model should be trained**, but rather:

> What information should the model receive, and in what form?

This question is addressed through **feature engineering**, **feature selection**, and, in some cases, **feature extraction or dimensionality reduction**.

These concepts are related, but they are not equivalent.

- **Feature engineering** creates or transforms variables so that useful information is represented more effectively.
- **Feature selection** retains a subset of the available variables and removes variables that are redundant, uninformative, unstable, or inappropriate for the modelling objective.
- **Feature extraction** constructs a new lower-dimensional representation from the original variables, usually by combining them mathematically.

These operations can improve interpretability, computational efficiency, statistical stability, and sometimes predictive performance. They can also make a model worse if they are performed mechanically, without considering the data-generating process or without protecting the evaluation from information leakage.

This document focuses on **tabular data** containing mixtures of numerical and categorical predictors and considers three common supervised-learning settings:

```text
binary classification
multiclass classification
regression with a numerical target
```

The objective is not to prescribe one universal recipe. Instead, the goal is to understand the available strategies, what problem each strategy solves, how they relate to exploratory data analysis, and when each approach is appropriate.

---

# Part I. Core concepts

## 2. What is a feature?

A **feature** is a variable supplied to a predictive model as input.

For a patient-level dataset, features might include:

```text
age
body_mass_index
smoker
hypertension
cholesterol
sleep_duration
```

For an agricultural dataset, features might include:

```text
soil_ph
rainfall_mm
mean_temperature_c
irrigation
fertilizer_type
pest_pressure
```

Not every column in a table should automatically be treated as a feature.

Columns may instead represent:

- identifiers;
- metadata;
- acquisition information;
- batch information;
- labels or targets;
- variables unavailable at prediction time;
- variables derived from the target itself;
- post-outcome measurements.

The first feature-engineering decision is therefore often simply to define **which columns are legitimate predictors**.

---

## 3. Feature engineering, feature selection, and feature extraction

These three operations answer different questions.

### 3.1 Feature engineering

Feature engineering asks:

> Can the available information be represented in a more useful form?

Examples include:

```text
weight + height              -> BMI
waist / hip circumference   -> waist-to-hip ratio
date of birth               -> age
raw timestamp                -> hour of day / day of week
income                       -> log(income)
city                         -> one-hot encoded city indicators
```

The number of variables may increase, decrease, or remain unchanged.

### 3.2 Feature selection

Feature selection asks:

> Which of the available features should be retained?

For example, starting from:

```text
100 candidate predictors
```

we might retain:

```text
25 informative and non-redundant predictors
```

The retained variables preserve their original meaning.

### 3.3 Feature extraction

Feature extraction asks:

> Can several original variables be combined into a smaller set of derived dimensions?

For example, Principal Component Analysis may transform:

```text
50 numerical variables
```

into:

```text
10 principal components
```

The new components summarize combinations of the original variables, but usually have less direct semantic interpretation.

This distinction is important:

```text
feature selection  -> keeps original variables
feature extraction -> creates new dimensions
```

---

## 4. Why reduce the feature space?

More features are not automatically better.

A dataset may contain variables that are:

- redundant;
- nearly constant;
- highly correlated;
- noisy;
- weakly measured;
- duplicated under different names;
- irrelevant to the target;
- informative only because of leakage;
- too sparse to estimate reliably;
- unstable across samples or folds.

Reducing the feature space can provide several benefits.

### 4.1 Improved interpretability

A model based on 12 meaningful variables is generally easier to inspect and communicate than one based on 600 weakly justified variables.

### 4.2 Lower computational cost

Fewer variables reduce memory use and can reduce training and tuning time.

### 4.3 Reduced variance

Removing noisy predictors can make fitted models more stable across different training samples.

### 4.4 Reduced multicollinearity

Strongly redundant numerical predictors can create unstable coefficient estimates in linear and generalized linear models.

### 4.5 Better sample-to-feature ratio

Feature selection can be especially important when the number of predictors is large relative to the number of observations.

### 4.6 Better generalization

A carefully reduced feature set may generalize better if irrelevant variables otherwise encourage the model to fit accidental patterns.

However, dimensionality reduction is not guaranteed to improve predictive performance. Modern regularized and tree-based models can often tolerate many predictors. Feature reduction should therefore be treated as a **hypothesis to evaluate**, not as an obligatory ritual.

---

# Part II. Relationship with exploratory data analysis

## 5. EDA should inform feature engineering

Exploratory Data Analysis provides much of the evidence used to decide whether feature engineering or feature selection may be useful.

For example, EDA may reveal:

| EDA observation | Possible downstream action |
|---|---|
| Strongly skewed positive variable | Consider log or power transformation |
| Strongly correlated numerical variables | Consider redundancy reduction |
| Binary feature with almost no positive observations | Consider removing or carefully grouping it |
| Categorical variable with many rare levels | Consider level grouping or alternative encoding |
| Nonlinear relationship with target | Consider transformation, spline, bins, or nonlinear model |
| Interaction visible between two variables | Consider interaction feature |
| Strong group-dependent distributions | Preserve feature and test its predictive stability |
| Variable dominated by outliers | Investigate data quality or robust transformation |
| Duplicate measurements of the same concept | Choose one, aggregate, or derive a composite |
| Strong batch or site effect | Investigate confounding rather than blindly selecting it |

EDA therefore helps generate **feature hypotheses**.

But EDA does not automatically justify modifying the data.

A pattern seen in one dataset may be accidental. Feature engineering decisions that use the target should later be evaluated inside a leakage-safe modelling procedure.

---

## 6. EDA and feature selection are not the same thing

EDA is primarily descriptive and diagnostic.

Feature selection is a modelling decision.

For example, an EDA may show that two variables have correlation:

```text
rho = 0.93
```

That observation indicates strong redundancy, but it does not tell us automatically which variable should be removed.

The decision may also depend on:

- measurement reliability;
- missingness;
- interpretability;
- availability at deployment;
- cost of acquisition;
- association with the target;
- stability across folds;
- domain knowledge.

Similarly, a feature may show a weak univariate association with the target but still become important through interactions with other predictors.

Therefore:

> EDA identifies structure. Feature selection decides what information to retain for modelling.

---

# Part III. The central rule: prevent information leakage

## 7. Feature engineering can leak information

One of the most important principles in machine learning is that transformations learned from data must not use information from the evaluation set.

Consider the following workflow:

```text
complete dataset
    ↓
select 20 features using association with the target
    ↓
train/test split
    ↓
train model
    ↓
evaluate
```

This is incorrect.

The feature-selection step has already used target information from observations that later appear in the test set.

The reported test performance is therefore optimistic.

The correct workflow is:

```text
complete dataset
    ↓
train/test split
    ↓
TRAINING DATA
    ↓
fit feature engineering / selection
    ↓
fit model
    ↓
apply learned transformations to TEST
    ↓
evaluate once
```

The same principle applies during cross-validation.

For each fold:

```text
training fold
    ↓
fit preprocessing
fit encoding
fit scaling
fit feature selection
fit dimensionality reduction
fit model
    ↓
validation fold
```

The validation fold must only be **transformed**, never used to decide how the transformations are fitted.

---

## 8. What should be fitted only on training data?

Examples include:

- means and standard deviations used for scaling;
- medians used for imputation;
- categorical vocabularies when unknown-level handling matters;
- target encoding;
- feature-selection scores computed against the target;
- correlation-based pruning thresholds when the selected variables depend on observed training structure;
- PCA components;
- supervised dimensionality reduction;
- recursive feature elimination;
- model-based importance thresholds;
- learned interaction structures.

The safest implementation is normally to include these operations inside a **machine-learning pipeline**.

---

## 9. Exploratory analysis and the final test set

Once a final hold-out test set has been defined, it should not be repeatedly inspected to make modelling decisions.

A useful conceptual separation is:

```text
TRAIN / DEVELOPMENT DATA
    exploration
    feature engineering
    feature selection
    tuning
    cross-validation

FINAL TEST DATA
    final evaluation only
```

In educational notebooks, the full synthetic dataset may be explored to teach concepts. In a real predictive study, however, the final test set should remain protected from target-informed analytical decisions.

---

# Part IV. Feature engineering for numerical variables

## 10. Numerical transformations

Numerical features can often be represented more effectively using transformations.

### 10.1 Log transformation

For strongly right-skewed positive variables:

```python
import numpy as np

df["log_income"] = np.log1p(df["income"])
```

This may reduce skewness and compress very large values.

Typical examples include:

- concentrations;
- counts;
- income;
- durations;
- molecular quantities spanning several orders of magnitude.

A log transformation should have a scientific or distributional rationale. It should not be applied automatically to every numerical variable.

### 10.2 Square-root transformation

Square-root transformations can be useful for non-negative count-like variables with moderate right skew.

### 10.3 Power transformations

Common alternatives include:

- Box-Cox;
- Yeo-Johnson.

Yeo-Johnson can also handle zero and negative values.

These transformations are often estimated from training data and should therefore be included inside the training pipeline.

---

## 11. Scaling

Scaling is not always considered feature engineering in the strictest sense, but it changes the numerical representation supplied to the model and is therefore closely related.

Common methods include:

### Standard scaling

```text
mean = 0
standard deviation = 1
```

Useful for:

- logistic regression;
- linear regression with regularization;
- support-vector machines;
- k-nearest neighbours;
- PCA;
- neural networks.

### Min-max scaling

Maps values to a specified range, commonly `[0, 1]`.

### Robust scaling

Uses robust statistics such as the median and interquartile range and can be useful when extreme observations are present.

Tree-based methods usually do not require scaling for predictive reasons.

---

## 12. Derived ratios and domain-informed combinations

Some of the most useful engineered features come directly from domain knowledge.

Examples include:

```text
BMI = weight / height^2
waist_to_hip_ratio = waist / hip
price_per_unit = total_price / number_of_units
growth_rate = change / elapsed_time
```

Such features can make an important relationship explicit instead of requiring a model to discover it indirectly.

However, a derived variable may be highly redundant with its source variables. The choice between keeping:

```text
weight + height + BMI
```

or only a subset should be evaluated rather than assumed.

---

## 13. Interaction features

An interaction occurs when the effect of one variable depends on another variable.

A simple numerical interaction is:

```python
df["temperature_x_irrigation"] = (
    df["mean_temperature_c"] * df["irrigation_mm"]
)
```

For linear models, explicit interactions can represent relationships that the model cannot otherwise learn.

For example:

```text
y = b0 + b1*x1 + b2*x2 + b3*(x1*x2)
```

Tree-based models can often learn interactions without explicitly creating interaction columns, although explicit domain-derived interactions may still be useful.

Creating all possible pairwise interactions is usually a bad first strategy because the feature space can grow rapidly.

---

## 14. Polynomial terms

A nonlinear numerical relationship may be represented through polynomial terms:

```text
x
x^2
x^3
```

For example, crop yield might increase with temperature up to an optimum and then decrease.

A simple quadratic representation can model this pattern:

```text
yield ~ temperature + temperature^2
```

Polynomial expansion should generally be selective. High polynomial degrees can create unstable models and extreme extrapolation behavior.

---

## 15. Binning numerical variables

A numerical variable can sometimes be converted into intervals:

```text
age
    ↓
18-29
30-44
45-59
60+
```

Potential reasons include:

- clinically meaningful thresholds;
- policy-defined categories;
- nonlinear threshold effects;
- improved interpretability.

However, arbitrary binning loses information and should not be used merely because a variable is continuous.

Data-driven bin boundaries learned from the target are particularly prone to overfitting and must be learned only from training data.

---

# Part V. Feature engineering for categorical variables

## 16. Why categorical variables require representation choices

Most machine-learning algorithms cannot directly interpret arbitrary strings such as:

```text
soil_type = clay
soil_type = sandy
soil_type = loam
```

Categorical values therefore need an appropriate numerical representation.

The correct representation depends on whether the variable is:

- nominal;
- ordinal;
- binary;
- high-cardinality;
- sparse;
- associated with a known hierarchy.

---

## 17. One-hot encoding

For a nominal variable:

```text
soil_type = clay / sandy / loam
```

one-hot encoding creates indicator variables such as:

```text
soil_type_clay
soil_type_sandy
soil_type_loam
```

This is one of the safest and most interpretable default encodings for low- or moderate-cardinality categorical variables.

Example with scikit-learn:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"` is useful because a category may occur in validation or test data that was not observed during training.

---

## 18. Ordinal encoding

Ordinal encoding is appropriate when categories possess a meaningful order.

For example:

```text
low < medium < high
```

could be mapped to:

```text
0, 1, 2
```

This should **not** be used for nominal categories where the assigned integer order is arbitrary.

Encoding:

```text
red = 0
green = 1
blue = 2
```

would falsely imply an ordered numerical structure.

---

## 19. Frequency and count encoding

A category can be replaced by its frequency in the training data.

For example:

```text
category A -> 0.42
category B -> 0.08
category C -> 0.50
```

This can be useful for high-cardinality variables when one-hot encoding would create a very large sparse matrix.

Frequency encoding does not directly use the target, but it still learns the distribution of categories from data and should generally be fitted on the training partition.

---

## 20. Target encoding

Target encoding replaces categories using information from the target.

For binary classification, a category may be represented by the estimated positive-class rate.

For regression, it may be represented by the mean target value.

Target encoding can be powerful for high-cardinality categorical variables, but it has a **very high leakage risk**.

A naive implementation such as:

```text
compute category mean using complete dataset
```

can encode the target itself into the predictors.

Correct target encoding should use:

- training data only;
- cross-fitted or out-of-fold estimates;
- smoothing or regularization for rare categories.

Target encoding is therefore not recommended as the first encoding method for beginner workflows.

---

## 21. Grouping rare categories

A categorical feature may contain many levels represented by very few observations.

For example:

```text
A: 450
B: 380
C: 140
D: 12
E: 9
F: 5
G: 4
```

Rare levels can sometimes be grouped into:

```text
other
```

This can reduce sparsity and improve stability.

The threshold for defining rare categories should be justified and learned without using the final test set.

---

# Part VI. Removing obviously unsuitable variables

## 22. Identifier columns

Unique identifiers such as:

```text
patient_id
sample_id
record_id
```

are normally excluded from predictive modelling.

An identifier may accidentally correlate with acquisition order, batch, hospital, date, or another hidden structure. Treating it as a feature can create spurious predictive performance.

---

## 23. Constant features

A constant feature contains the same value for every observation.

For example:

```text
country = Germany
```

for every row.

Such a feature contains no information for distinguishing observations and can be removed.

---

## 24. Near-zero-variance features

A feature may technically vary but be almost constant.

For example:

```text
0 -> 998 observations
1 -> 2 observations
```

Whether to remove it depends on context.

A rare event may still be scientifically important. Near-zero variance should therefore trigger inspection rather than automatic deletion.

---

## 25. Leakage features

Some variables are highly predictive because they contain information that would not be available when predictions are made.

Examples include:

```text
final pathology result
post-treatment measurement
future event count
manually assigned diagnostic code
```

A feature can be statistically excellent and methodologically invalid.

Feature validity must therefore be judged not only by association strength but also by the intended prediction scenario.

---

# Part VII. Redundancy reduction

## 26. Correlation-based redundancy reduction

Strongly correlated numerical features may encode nearly the same information.

For example:

```text
weight_kg
weight_lb
```

would be almost perfectly correlated.

A common strategy is to identify pairs with absolute correlation above a threshold such as:

```text
|r| > 0.90
```

or:

```text
|rho| > 0.90
```

and retain only one variable from each highly redundant pair.

The threshold is not universal.

When deciding which variable to retain, consider:

- interpretability;
- missingness;
- measurement reliability;
- cost;
- target association;
- expected availability during deployment;
- domain relevance.

---

## 27. Pearson versus Spearman for redundancy

**Pearson correlation** measures linear association.

**Spearman correlation** measures monotonic association based on ranks.

For redundancy analysis, Spearman can be preferable when variables are strongly monotonic but not linearly related.

Neither coefficient captures every possible nonlinear dependency.

---

## 28. Variance Inflation Factor

The **Variance Inflation Factor**, or VIF, is commonly used to diagnose multicollinearity among numerical predictors in linear-model settings.

For feature `j`:

```text
VIF_j = 1 / (1 - R_j^2)
```

where `R_j^2` is obtained by predicting feature `j` from the other predictors.

Large VIF values indicate that the feature can be explained strongly by other features.

Common heuristic interpretations include:

```text
VIF around 1    -> little multicollinearity
VIF > 5         -> potentially concerning
VIF > 10        -> often considered strong multicollinearity
```

These are heuristics, not universal laws.

VIF is most useful when coefficient stability and interpretation matter. It is less central for tree-based models.

---

## 29. Redundancy among categorical variables

Categorical predictors can also be redundant.

Useful association measures include:

- chi-square statistics;
- Cramér's V;
- contingency-table inspection;
- mutual information.

For example, two categorical variables may represent nearly the same administrative classification under different names.

---

## 30. Redundancy in mixed numerical and categorical data

Mixed datasets require different association measures for different pairs of variable types.

A useful conceptual matrix is:

| Predictor A | Predictor B | Possible association measure |
|---|---|---|
| Numerical | Numerical | Pearson / Spearman |
| Categorical | Categorical | Cramér's V / mutual information |
| Numerical | Categorical | Group comparison, correlation ratio, mutual information |

There is no single universal correlation coefficient that perfectly describes all mixed-type relationships.

---

# Part VIII. Feature selection families

## 31. Three major families

Feature-selection methods are commonly divided into:

```text
filter methods
wrapper methods
embedded methods
```

A fourth useful category is **post-hoc model-based selection**, where feature importance from a fitted model is used as evidence for retaining variables.

---

# Part IX. Filter methods

## 32. What are filter methods?

Filter methods score variables using statistical or information-based criteria without repeatedly fitting the final predictive model.

They are usually:

- fast;
- scalable;
- interpretable;
- useful as an initial reduction step.

However, they often evaluate variables individually and may miss interactions.

---

## 33. Variance filtering

A simple unsupervised filter removes constant or extremely low-variance features.

Example:

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
```

This does not use the target.

It is useful for removing structurally uninformative columns, but it should not be confused with supervised feature selection.

---

## 34. Correlation filtering

Correlation filtering removes redundant numerical predictors based on pairwise association.

It is also target-independent if only predictor-predictor correlations are used.

A typical procedure is:

```text
1. Compute predictor correlation matrix.
2. Identify highly correlated pairs.
3. Define a deterministic retention rule.
4. Remove redundant variables.
```

It should not simply remove every feature that participates in a high-correlation pair without considering which variable should be retained.

---

## 35. Univariate statistical filters for classification

For numerical predictors and a **binary categorical label**, common options include:

- Welch's t-test;
- Mann-Whitney U;
- point-biserial correlation;
- ANOVA F score;
- mutual information.

For numerical predictors and a **multiclass label**, common options include:

- one-way ANOVA;
- Welch ANOVA;
- Kruskal-Wallis;
- mutual information.

These methods test whether a predictor differs across target groups.

A small p-value is not sufficient by itself. Effect size should also be considered.

---

## 36. Categorical predictors versus categorical labels

For categorical predictor and categorical target combinations, common methods include:

- chi-square test of independence;
- Fisher's exact test for appropriate small contingency tables;
- Cramér's V;
- mutual information.

For binary predictors in a binary classification problem, additional interpretable measures include:

- odds ratio;
- risk difference;
- prevalence difference.

---

## 37. Univariate filters for regression

For a numerical target, candidate numerical predictors can be evaluated using:

- Pearson correlation;
- Spearman correlation;
- Kendall correlation;
- univariate linear-regression F statistics;
- mutual information for regression.

Categorical predictors can be compared against a continuous target using:

- Welch t-test for two groups;
- ANOVA for multiple groups when assumptions are reasonable;
- Mann-Whitney U for two-group nonparametric comparisons;
- Kruskal-Wallis for multiple groups;
- effect-size measures.

Again, a weak linear correlation does not imply that a predictor is useless. A variable may have a strong nonlinear relationship with the target.

---

## 38. Mutual information

Mutual information measures dependency between variables without assuming a purely linear relationship.

It can be used for:

```text
classification -> mutual_info_classif
regression     -> mutual_info_regression
```

Mutual information is useful when nonlinear dependencies are plausible.

However:

- estimates can be noisy for small datasets;
- rankings may vary with sample size and estimator settings;
- it does not directly indicate the direction of an association;
- it should not replace visual inspection or effect-size reasoning.

---

## 39. Multiple-testing correction

If hundreds of features are tested independently, some will appear significant by chance.

Therefore, univariate feature-screening procedures should consider multiple-testing correction.

A common option is:

```text
Benjamini-Hochberg False Discovery Rate
```

The workflow may be:

```text
compute one test per feature
    ↓
collect p-values
    ↓
apply FDR correction
    ↓
interpret adjusted significance together with effect size
```

Feature selection should not be reduced to:

```text
adjusted p-value < 0.05 -> keep
adjusted p-value >= 0.05 -> remove
```

Statistical detectability, predictive usefulness, redundancy, and domain value are different concepts.

---

# Part X. Wrapper methods

## 40. What are wrapper methods?

Wrapper methods evaluate subsets of variables by repeatedly fitting a predictive model.

Instead of asking:

> Is this individual feature associated with the target?

wrapper methods ask:

> Does this subset of features improve model performance?

They can capture interactions and model-specific usefulness, but are more computationally expensive.

---

## 41. Sequential feature selection

### Forward selection

Begin with no variables and add features iteratively.

```text
0 features
   ↓
add best feature
   ↓
add next best feature
   ↓
continue
```

### Backward selection

Begin with all variables and remove features iteratively.

```text
all features
   ↓
remove least useful
   ↓
continue
```

Sequential selection can be useful for moderate-dimensional datasets when interpretability matters.

Its result depends on:

- the estimator;
- evaluation metric;
- cross-validation procedure;
- stopping criterion.

---

## 42. Recursive Feature Elimination

Recursive Feature Elimination, or **RFE**, repeatedly fits a model and removes the least important feature or group of features.

A cross-validated version, **RFECV**, can select the number of features based on validation performance.

Typical workflow:

```text
fit model with all features
    ↓
rank features
    ↓
remove weakest feature(s)
    ↓
refit
    ↓
repeat
```

RFE is model-dependent. A subset selected by logistic regression may differ from one selected by a random forest.

---

## 43. Advantages and limitations of wrappers

Advantages:

- directly linked to predictive performance;
- can consider interactions;
- can be aligned with the final model family.

Limitations:

- computationally expensive;
- can overfit feature subsets when data are limited;
- may be unstable across folds;
- require nested or carefully structured validation when combined with tuning.

---

# Part XI. Embedded methods

## 44. What are embedded methods?

Embedded methods perform feature selection as part of model fitting.

The selection mechanism is built into the estimator or its regularization strategy.

---

## 45. L1 regularization

L1 regularization penalizes the absolute magnitude of model coefficients.

For sufficiently strong regularization, some coefficients become exactly zero.

This provides a natural feature-selection mechanism.

Common examples include:

- Lasso regression;
- L1-regularized logistic regression.

L1 selection is attractive when:

- the number of predictors is moderate or large;
- sparse solutions are desirable;
- predictors are appropriately encoded and scaled.

With strongly correlated features, L1 may select one variable somewhat arbitrarily from a correlated group.

---

## 46. Elastic Net

Elastic Net combines L1 and L2 regularization.

It can be more stable than pure L1 when groups of predictors are correlated.

It is particularly useful for high-dimensional numerical data and one-hot encoded tabular matrices.

---

## 47. Tree-based embedded importance

Decision trees and ensembles such as random forests or gradient boosting can provide feature-importance estimates.

These may be used to rank variables.

However, impurity-based importance can be biased toward:

- continuous variables;
- high-cardinality variables;
- variables offering many potential split points.

Therefore, built-in importance should not be treated as an unquestionable measure of scientific relevance.

---

# Part XII. Post-hoc importance and stability

## 48. Permutation importance

Permutation importance asks how much predictive performance decreases when one feature is randomly shuffled.

Conceptually:

```text
baseline model performance
    ↓
permute one feature
    ↓
recompute performance
    ↓
importance = performance decrease
```

It is useful because it is relatively model-agnostic.

However, strongly correlated features can mask one another. If two predictors contain almost the same information, permuting one may have little effect because the model can still use the other.

---

## 49. Stability selection

A feature may look important in one split but disappear in another.

A robust selection procedure therefore asks:

> How often is this feature selected across resampled training datasets or cross-validation folds?

For example:

| Feature | Selection frequency |
|---|---:|
| age | 0.98 |
| hypertension | 0.91 |
| biomarker_7 | 0.54 |
| feature_42 | 0.08 |

Stability can be more informative than a single ranking.

This is especially important when:

- sample size is limited;
- features are correlated;
- many features have similar predictive value.

---

## 50. Boruta and shadow-feature approaches

Boruta is a less commonly used but useful method for identifying features that outperform randomly permuted "shadow" variables, typically using tree-based models.

It is designed more as an **all-relevant feature selection** approach than a minimal-subset selector.

Potential advantages:

- can detect nonlinear effects;
- can retain multiple relevant correlated variables.

Potential limitations:

- computational cost;
- dependence on the underlying tree model;
- unstable behavior in small datasets;
- more complex interpretation than simple filters.

It is useful to know, but it should not be the default first method for every tabular problem.

---

## 51. Relief-based methods

Algorithms such as ReliefF estimate feature relevance by comparing nearby observations belonging to similar or different target classes.

They can detect interactions that univariate filters may miss.

They are less common in standard tabular workflows than regularization, mutual information, RFE, or tree-based importance, but can be useful in specialized settings.

---

## 52. Minimum Redundancy Maximum Relevance

**mRMR** attempts to select variables that are:

```text
highly relevant to the target
```

while also being:

```text
minimally redundant with one another
```

This explicitly balances two objectives that simple univariate ranking ignores.

It can be useful when many variables are individually informative but strongly redundant.

As with all supervised selectors, it must be fitted inside the training data only.

---

# Part XIII. Feature extraction and dimensionality reduction

## 53. When feature extraction is preferable to selection

Feature selection is often desirable when interpretation of original variables matters.

Feature extraction can be preferable when:

- there are many correlated numerical measurements;
- compact representation matters more than direct interpretation;
- downstream methods benefit from lower-dimensional inputs;
- the original variables describe a common latent structure.

---

## 54. Principal Component Analysis

PCA constructs orthogonal linear combinations of numerical predictors.

Its goals include:

- variance compression;
- redundancy reduction;
- visualization;
- construction of lower-dimensional model inputs.

PCA is **unsupervised**.

It does not use the target when determining components.

Therefore, the components explaining the greatest predictor variance are not guaranteed to be the components most useful for prediction.

When PCA is used for modelling, it should be fitted on the training partition and then applied to validation/test data.

---

## 55. How PCA differs from feature selection

Suppose the original variables are:

```text
x1, x2, x3, x4
```

Feature selection might retain:

```text
x1, x3
```

PCA might instead produce:

```text
PC1 = 0.5*x1 + 0.4*x2 - 0.3*x3 + 0.7*x4
PC2 = ...
```

The PCA representation is compact but less directly interpretable.

---

## 56. Categorical and mixed-data dimensionality reduction

PCA is naturally designed for numerical variables.

Related techniques exist for other data types.

### Multiple Correspondence Analysis

**MCA** is commonly used for categorical variables represented through indicator structures.

### Factor Analysis of Mixed Data

**FAMD** combines ideas from PCA and MCA for datasets containing both numerical and categorical variables.

These methods can be useful for exploratory representation and dimensionality reduction, but they are less common in basic predictive pipelines than one-hot encoding followed by an estimator.

---

## 57. Supervised projections

Methods such as Partial Least Squares incorporate information from the target while constructing latent dimensions.

Because they are supervised, they must be fitted strictly inside training data.

They can be useful when predictors are numerous and highly correlated, but should not be confused with unsupervised PCA.

---

# Part XIV. Binary classification

## 58. Feature selection with a binary label

Consider a target:

```text
No Alzheimer
Alzheimer
```

For numerical predictors, useful exploratory or filter evidence may include:

- Welch t-test;
- Mann-Whitney U;
- Hedges' g;
- Cliff's delta;
- mutual information;
- model-based importance.

For categorical predictors:

- prevalence differences;
- chi-square;
- Fisher's exact test;
- odds ratio;
- Cramér's V;
- mutual information.

A reasonable feature-selection workflow might be:

```text
remove invalid/non-predictive columns
    ↓
remove constants
    ↓
inspect redundancy
    ↓
perform training-only univariate screening
    ↓
fit regularized or wrapper selector
    ↓
assess stability across CV folds
    ↓
compare reduced vs unreduced predictive pipeline
```

The final criterion should include validation performance and stability, not only univariate p-values.

---

# Part XV. Multiclass classification

## 59. Why multiclass selection is different

With a target such as:

```text
healthy_sleep
insomnia
sleep_apnea
circadian_disruption
```

there is no single positive-versus-negative comparison.

A feature may distinguish:

- one class from all others;
- two specific classes;
- all classes globally;
- only a subset of class pairs.

This structure should be preserved during feature evaluation.

---

## 60. Numerical predictors with multiclass targets

Useful global methods include:

- ANOVA;
- Welch ANOVA;
- Kruskal-Wallis;
- eta squared;
- epsilon squared;
- mutual information.

If a global test indicates group differences, pairwise post-hoc analysis can determine **which classes differ**.

For example:

```text
sleep_latency
```

might strongly distinguish insomnia from healthy sleep but contribute little to separation between sleep apnea and circadian disruption.

This is why a single global p-value is not a complete description of multiclass relevance.

---

## 61. Categorical predictors with multiclass targets

Useful methods include:

- class-conditional prevalence tables;
- chi-square test of independence;
- Cramér's V;
- mutual information.

For interpretation, it is often useful to inspect the full contingency table rather than only one scalar score.

---

## 62. One-versus-rest feature analysis

For some applications, a multiclass problem can be examined with one-versus-rest contrasts:

```text
class A vs all others
class B vs all others
class C vs all others
```

This can reveal class-specific variables.

However, one-versus-rest screening should complement, not replace, the global multiclass analysis.

---

# Part XVI. Regression

## 63. Feature selection with a numerical target

For regression, the target is continuous:

```text
yield_t_ha
```

There are no target classes.

Feature relevance therefore concerns relationships with the **magnitude of the target**.

---

## 64. Numerical predictors in regression

Common evidence includes:

- Pearson correlation;
- Spearman correlation;
- Kendall correlation;
- univariate regression F score;
- mutual information;
- nonlinear relationship plots;
- model-based importance.

It is especially important not to select only by Pearson correlation.

A predictor may show a U-shaped or inverted-U relationship with the target and therefore have a low linear correlation despite being highly informative.

For example:

```text
yield
  ↑
  |       /\
  |      /  \
  |_____/    \____
        temperature
```

A simple correlation can miss this structure.

---

## 65. Categorical predictors in regression

For a binary categorical predictor and continuous target, useful comparisons include:

- Welch t-test;
- Mann-Whitney U;
- Hedges' g;
- Cliff's delta.

For categorical predictors with more than two levels:

- ANOVA;
- Welch ANOVA;
- Kruskal-Wallis;
- effect-size measures.

For predictive selection, one-hot encoded categories can also be evaluated using regularized regression, permutation importance, or tree-based models.

---

# Part XVII. Mixed numerical and categorical predictors

## 66. The practical mixed-data problem

Real tabular datasets often contain both:

```text
numerical predictors
categorical predictors
```

A correct feature-engineering pipeline must transform them differently.

For example:

```text
numerical
    imputation
    scaling
    optional transformation

categorical
    imputation
    one-hot encoding
```

and then combine the resulting feature matrix.

---

## 67. ColumnTransformer

In scikit-learn, `ColumnTransformer` provides a clean way to apply different operations to different feature groups.

A simplified structure is:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)
```

This can then be placed inside a complete modelling pipeline.

---

## 68. Selection before or after one-hot encoding?

Both approaches are possible, but they answer slightly different questions.

### Selection before encoding

Selects original variables such as:

```text
soil_type
exercise_status
smoker
```

This preserves variable-level interpretation.

### Selection after encoding

May select individual dummy variables such as:

```text
soil_type_clay
```

while removing:

```text
soil_type_sandy
```

This can improve predictive compactness but makes interpretation at the original-variable level less direct.

For teaching and scientific interpretation, it is often useful to distinguish:

```text
variable-level selection
```

from:

```text
encoded-column selection
```

---

## 69. Group-aware treatment of encoded categorical features

If a categorical variable generates multiple one-hot columns, treating each dummy independently may fragment the original variable.

For some analyses, it is preferable to retain or remove the entire group of encoded columns together.

This is especially relevant when reporting which original variables were selected.

---

# Part XVIII. Common versus less-common techniques

## 70. Common first-line techniques

For most tabular projects, the following methods cover a large fraction of practical needs:

| Goal | Common technique |
|---|---|
| Remove constants | Variance threshold |
| Detect numerical redundancy | Pearson / Spearman correlation |
| Diagnose linear multicollinearity | VIF |
| Encode nominal categories | One-hot encoding |
| Encode ordinal categories | Ordinal encoding |
| Binary-label numerical screening | Welch / Mann-Whitney + effect size |
| Multiclass numerical screening | ANOVA / Kruskal-Wallis + effect size |
| Regression numerical screening | Pearson / Spearman + plots |
| Flexible univariate relevance | Mutual information |
| Sparse linear selection | L1 / Elastic Net |
| Model-dependent subset selection | RFE / sequential selection |
| Model-agnostic importance | Permutation importance |
| Numerical feature extraction | PCA |

These methods are common because they are relatively understandable, widely implemented, and easy to integrate into reproducible pipelines.

---

## 71. Useful but more specialized techniques

Less-common or more specialized options include:

- Boruta;
- ReliefF;
- mRMR;
- stability selection;
- supervised latent projections;
- MCA;
- FAMD;
- group Lasso;
- hierarchical feature selection;
- interaction screening;
- spline basis construction;
- target encoding with cross-fitting.

These methods can be valuable when the data structure justifies them, but complexity should not be added simply because a method exists.

---

# Part XIX. Choosing a strategy

## 72. There is no universal best selector

A good feature-selection strategy depends on:

- sample size;
- number of predictors;
- target type;
- predictor types;
- degree of redundancy;
- expected nonlinearity;
- model family;
- interpretability requirements;
- computational budget;
- deployment constraints.

The same feature set is not necessarily optimal for every model.

---

## 73. Small number of features

If a dataset contains:

```text
20-30 well-defined predictors
```

aggressive feature selection may be unnecessary.

A sensible workflow may be:

```text
remove identifiers / leakage
inspect redundancy
encode correctly
compare regularized and tree-based models
```

Selection should not be performed merely to say that feature selection was performed.

---

## 74. Moderate-dimensional tabular data

For tens to hundreds of predictors, useful strategies include:

```text
structural filtering
    +
redundancy reduction
    +
univariate screening
    +
regularized or model-based selection
```

Selection stability should be inspected across resamples.

---

## 75. High-dimensional data

When the number of predictors approaches or exceeds the number of observations:

```text
p ≈ n
```

or:

```text
p >> n
```

feature reduction becomes more important.

Useful approaches may include:

- strong regularization;
- dimensionality reduction;
- stability selection;
- domain-informed prefiltering;
- sparse linear models;
- careful nested cross-validation.

Overfitting risk is especially high in this setting.

---

# Part XX. Model-specific considerations

## 76. Linear and logistic regression

Linear models benefit strongly from attention to:

- scaling;
- multicollinearity;
- nonlinear transformations;
- interaction terms;
- regularization.

Feature engineering can substantially change what these models are able to represent.

---

## 77. k-Nearest Neighbours

KNN is distance-based.

Therefore it is sensitive to:

- scale;
- irrelevant variables;
- high dimensionality.

Feature selection and scaling can be particularly important.

---

## 78. Support Vector Machines

SVMs are generally sensitive to scale.

Feature reduction can improve computational efficiency, especially for high-dimensional data.

Kernel models can represent nonlinear relationships without explicit polynomial feature construction, although tuning becomes important.

---

## 79. Decision trees and random forests

Tree-based methods naturally represent:

- thresholds;
- nonlinearities;
- interactions.

They do not generally require scaling.

However, they can still benefit from removing:

- leakage variables;
- duplicated features;
- extremely noisy variables;
- high-cardinality artifacts.

---

## 80. Gradient boosting

Gradient-boosted trees often perform strongly on heterogeneous tabular data and can capture complex nonlinear interactions.

Aggressive manual feature engineering may be less necessary than for linear models, but domain-derived features can still improve learning efficiency and interpretability.

---

# Part XXI. Evaluating whether feature selection actually helped

## 81. Always compare against a baseline

A reduced feature set should be compared against the unreduced baseline.

For example:

```text
Pipeline A: all valid features
Pipeline B: selected features
```

Both pipelines should be evaluated under the same cross-validation splits and metric.

This allows the question:

> Did feature selection improve the model, or did it merely make the feature matrix smaller?

---

## 82. Evaluate more than predictive performance

Useful criteria include:

- validation performance;
- variance across folds;
- number of retained variables;
- selection stability;
- computational cost;
- interpretability;
- deployment availability;
- measurement cost.

A slightly less accurate model may still be preferable if it uses dramatically fewer, cheaper, or more reliable variables.

---

## 83. Selection stability across folds

Suppose five cross-validation folds select these feature sets:

```text
Fold 1: A B C D
Fold 2: A B C E
Fold 3: A B D E
Fold 4: A B C D
Fold 5: A B C D
```

Features `A` and `B` are highly stable.

Features `C`, `D`, and `E` are less stable.

This information should be retained. Reporting only one final selected subset hides uncertainty in the selection process.

---

## 84. Feature selection is part of model uncertainty

There may be several feature subsets with nearly identical predictive performance.

This is common when predictors are correlated.

Therefore, a selected feature set should not automatically be interpreted as the unique set of biologically or scientifically important variables.

Predictive selection and causal interpretation are different goals.

---

# Part XXII. Feature importance is not causality

## 85. Predictive relevance does not imply causal relevance

A feature can be highly predictive because it is:

- causal;
- correlated with a causal variable;
- a proxy for a hidden confounder;
- associated with site or batch;
- downstream of the outcome;
- partially leaking the target.

Therefore:

```text
selected feature != causal mechanism
```

Feature engineering and feature selection are predictive tools unless the study has been explicitly designed for causal inference.

---

# Part XXIII. Recommended workflow for this repository

## 86. Conceptual sequence

For the tabular examples in this repository, the recommended high-level sequence is:

```text
raw data
    ↓
preprocessing
    ↓
EDA
    ↓
define modelling predictors
    ↓
train / validation / test strategy
    ↓
feature engineering fitted on training data
    ↓
feature selection fitted on training data
    ↓
model fitting
    ↓
validation
    ↓
final test evaluation
```

The key conceptual transition is:

```text
EDA discovers candidate structure
feature engineering represents it
feature selection tests whether it should be retained
```

---

## 87. A practical staged strategy

A useful staged workflow is:

### Stage 1. Structural exclusions

Remove or exclude:

- identifiers;
- known leakage variables;
- invalid features;
- constants.

### Stage 2. Representation

Define transformations for:

- numerical features;
- categorical features;
- missing values;
- scaling when necessary.

### Stage 3. Redundancy control

Inspect:

- numerical correlations;
- VIF when relevant;
- categorical redundancy;
- duplicate or derived variables.

### Stage 4. Target-aware screening

Depending on the task, use:

- classification statistics;
- regression associations;
- mutual information;
- effect sizes.

This stage must be training-only once evaluation begins.

### Stage 5. Model-based selection

Optionally evaluate:

- L1 / Elastic Net;
- RFE;
- sequential selection;
- permutation importance;
- tree-based ranking.

### Stage 6. Stability analysis

Repeat selection across folds or resamples.

### Stage 7. Compare complete pipelines

Compare:

```text
all valid features
vs
reduced feature set
```

using identical evaluation splits.

---

# Part XXIV. Examples for the three tabular tasks

## 88. Task 1: binary Alzheimer classification

For the Alzheimer example, EDA may identify variables such as:

```text
hypertension
diabetes_type2
family_history_alzheimer
hearing_loss
stroke_history
```

with different prevalences between classes.

A subsequent feature-selection workflow could evaluate:

- numerical features using Welch/Mann-Whitney and effect sizes;
- binary categorical features using chi-square/Fisher and Cramér's V;
- redundancy among anthropometric and physiological variables;
- regularized logistic regression;
- permutation importance from a fitted classifier;
- stability across cross-validation folds.

The EDA evidence motivates candidate features, but the final subset must be selected inside training data.

---

## 89. Task 2: multiclass sleep-profile classification

For the sleep-profile example, different variables may distinguish different classes.

For example:

```text
sleep_latency_min
sleep_efficiency_pct
stress_score_0_10
bedtime_variability_min
social_jetlag_hours
shift_worker
loud_snoring
```

A useful selection strategy could combine:

```text
Kruskal-Wallis / ANOVA
    +
effect size
    +
pairwise follow-up
    +
chi-square / Cramér's V for categorical variables
    +
multiclass model-based selection
```

It is important not to retain features only because they distinguish one easy class while contributing nothing to the remaining class boundaries unless that behavior is acceptable for the modelling objective.

---

## 90. Task 3: crop-yield regression

For the crop-yield dataset, feature selection must account for nonlinear relationships.

A workflow based only on Pearson correlation could incorrectly discard predictors such as temperature or soil moisture when their relationship with yield is non-monotonic.

A better workflow may combine:

```text
Pearson / Spearman / Kendall
    +
quantile-profile inspection
    +
mutual information
    +
nonlinear model importance
    +
redundancy analysis
```

Categorical management variables can be evaluated using group comparisons and model-based importance after appropriate encoding.

---

# Part XXV. Common mistakes

## 91. Selecting features before splitting the data

Incorrect:

```text
full dataset
    ↓
select best features
    ↓
split
```

Correct:

```text
split
    ↓
fit selector on training data
    ↓
apply selector to validation/test data
```

---

## 92. Selecting only by p-value

A statistically significant feature may have a negligible effect size.

Conversely, a useful predictive feature may fail a strict univariate significance threshold but contribute through interactions.

Use statistical tests as evidence, not as an automatic truth machine.

---

## 93. Removing all correlated variables blindly

Correlation does not make one variable invalid.

Two correlated variables may differ in:

- reliability;
- cost;
- interpretability;
- missingness;
- deployment availability.

Choose deliberately.

---

## 94. Using the label to create features outside cross-validation

Examples include:

- target encoding;
- target-guided binning;
- supervised feature ranking;
- supervised dimensionality reduction.

These must be fitted within the training data for each fold.

---

## 95. Treating feature importance as biological truth

A predictive model ranks variables according to its own representation and objective.

Importance can change with:

- model family;
- data split;
- correlated predictors;
- hyperparameters;
- metric.

Scientific interpretation requires additional evidence.

---

## 96. Performing feature selection with no baseline comparison

A feature-selection method that reduces 50 variables to 10 may look elegant but still reduce generalization performance.

Always compare against the unreduced valid feature set.

---

## 97. Overengineering a small dataset

For a dataset with 15 meaningful features, it may be unnecessary to apply:

```text
mutual information
RFE
Boruta
PCA
mRMR
Elastic Net
```

all at once.

Complexity should solve a real problem identified in the data.

---

# Part XXVI. Recommended visualizations

## 98. Feature-selection visualizations

Useful figures include:

### Ranked effect-size plot

Useful for displaying the magnitude and direction of group differences.

### Feature-importance bar plot

Useful for showing model-based rankings.

### Correlation heatmap

Useful for identifying numerical redundancy.

### Clustered correlation map

Useful when many correlated variables form groups.

### Selection-frequency plot

Useful for stability analysis across folds or resamples.

### Performance versus number of features

Useful for wrapper methods and dimensionality-reduction studies.

### PCA explained-variance curve

Useful when evaluating feature extraction.

### Regularization path

Useful for understanding how coefficients enter or leave sparse linear models.

Visualizations should support a decision, not merely decorate the analysis.

---

# Part XXVII. Reporting feature engineering and selection

## 99. What should be documented?

A reproducible analysis should report:

- original number of predictor variables;
- excluded identifier or leakage columns;
- numerical transformations;
- categorical encoding strategy;
- scaling strategy;
- redundancy criteria;
- feature-selection method;
- target-aware statistics used;
- correction for multiple comparisons when applicable;
- estimator used by wrapper or embedded methods;
- cross-validation strategy;
- selected number of variables;
- feature-selection stability when assessed;
- final retained feature names;
- performance before and after feature reduction.

The objective is that another analyst could reconstruct exactly how the final feature matrix was obtained.

---

## 100. Export useful artifacts

Useful outputs may include:

```text
feature_metadata.csv
univariate_feature_scores.csv
redundancy_matrix.csv
selected_features.csv
selection_frequency.csv
feature_importance.csv
feature_engineering_config.json
```

The exact files depend on the workflow, but feature-selection decisions should not exist only inside notebook cells.

---

# Part XXVIII. Practical decision table

## 101. Choosing techniques by task

| Problem | Predictor type | Good first methods | Additional methods |
|---|---|---|---|
| Binary classification | Numerical | Welch/Mann-Whitney, effect size, MI | L1, RFE, permutation importance |
| Binary classification | Categorical | Chi-square/Fisher, Cramér's V, MI | Regularized encoded models |
| Multiclass classification | Numerical | ANOVA/Kruskal-Wallis, effect size, MI | Pairwise follow-up, multiclass RFE |
| Multiclass classification | Categorical | Chi-square, Cramér's V, MI | Model-based importance |
| Regression | Numerical | Pearson/Spearman/Kendall, plots, MI | Elastic Net, RFE, permutation importance |
| Regression | Categorical | Welch/ANOVA or nonparametric alternatives | Encoded model-based selection |
| Any task | Numerical redundancy | Pearson/Spearman, VIF | mRMR, clustering |
| Any task | High-dimensional numerical | Regularization, PCA | Stability selection |
| Mixed predictors | Numerical + categorical | ColumnTransformer + task-specific selector | Group-aware or model-based selection |

This table provides starting points, not rigid rules.

---

# Part XXIX. A compact checklist

## 102. Before feature engineering

Ask:

- Is the dataset already structurally clean?
- Which columns are legitimate predictors?
- Which columns are identifiers or metadata?
- Is there any obvious leakage?
- What did the EDA reveal about skewness, redundancy, interactions, and nonlinearity?
- What is the target type?

## 103. During feature engineering

Ask:

- Does the transformation have a clear rationale?
- Is information being preserved or unnecessarily discarded?
- Does the method need to be fitted on training data?
- Will the engineered feature exist at prediction time?
- Is the transformation interpretable and reproducible?

## 104. During feature selection

Ask:

- Am I selecting features using only training data?
- Am I considering effect size, not only p-values?
- Could correlated features make the ranking unstable?
- Am I using a method appropriate for the target type?
- Is the selected subset stable across folds?
- Does the reduced subset actually improve the pipeline?

## 105. Before final evaluation

Ask:

- Was the complete preprocessing and selection procedure fitted without test information?
- Can the entire transformation be reproduced from saved configuration or code?
- Is the final feature set available at deployment time?
- Has the reduced pipeline been compared fairly against an unreduced baseline?

---

# Part XXX. Final perspective

## 106. The main principle

Feature engineering and feature selection should not be treated as independent preprocessing tricks.

They are part of the scientific and predictive definition of the model.

A useful conceptual sequence is:

```text
understand the variables
        ↓
understand their distributions and relationships
        ↓
represent meaningful structure
        ↓
remove invalid or redundant information
        ↓
select useful information using training data only
        ↓
evaluate whether reduction improves generalization
```

The goal is not to obtain the smallest possible feature set.

The goal is to obtain a feature representation that is:

- valid;
- reproducible;
- leakage-safe;
- sufficiently informative;
- appropriately compact;
- stable;
- interpretable when interpretation matters.

For tabular machine learning, good feature engineering is often less about inventing large numbers of new variables and more about **representing the existing information correctly, removing unjustified complexity, and validating every reduction decision under the intended prediction scenario**.
