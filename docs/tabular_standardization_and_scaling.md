# Standardization, Scaling, and Numerical Transformations for Tabular Data

## 1. Purpose of this document

Tabular datasets often contain numerical variables measured on very different scales. A clinical table may combine age in years, body mass in kilograms, blood pressure in millimetres of mercury, ratios near one, and binary indicators taking only the values 0 and 1. An agricultural table may combine rainfall in millimetres, nutrient concentrations, pH, percentages, and a continuous yield target.

These differences are not necessarily a problem for exploratory analysis, but they can strongly influence many machine-learning algorithms. **Scaling** is the family of transformations used to place numerical variables on a more comparable numerical basis.

This document explains:

- what standardization and normalization mean;
- why scaling is useful;
- which methods are common;
- how the main transformations work;
- when each method is appropriate;
- how outliers affect scaling;
- how scaling should be handled for mixed numerical and categorical data;
- how the strategy differs for binary classification, multiclass classification, and regression;
- when a numerical target may also be scaled;
- how scaling relates to EDA, feature engineering, and the later machine-learning workflow;
- how to avoid data leakage.

The accompanying notebooks are **simple demonstrations**. They intentionally do not use end-to-end machine-learning pipelines. A later notebook will combine preprocessing, feature engineering, feature selection, scaling, and modelling into a complete leakage-safe preparation workflow.

---

# Part I. Core concepts

## 2. Scaling is not the same as cleaning

Scaling does not repair a dataset.

Before scaling, the dataset should already have undergone basic preprocessing. At minimum, numerical columns should have valid numerical types, missing values should be understood or handled appropriately, impossible values should have been investigated, and identifiers should already be distinguished from predictors.

A useful conceptual order is:

```text
raw data
   ↓
preprocessing and quality control
   ↓
EDA
   ↓
feature engineering / feature selection
   ↓
scaling / standardization
   ↓
model preparation and training
```

This order is conceptual rather than absolute. In a formal machine-learning workflow, operations that learn parameters from the data must be fitted only on the training data and, during cross-validation, refitted inside each training fold.

---

## 3. Why numerical scale matters

Consider three predictors:

```text
age                  18–90
waist_hip_ratio       0.6–1.2
daily_steps        1000–18000
```

Numerically, `daily_steps` varies over a much larger range than `waist_hip_ratio`. An algorithm that uses distances, dot products, gradients, or regularization may therefore give disproportionate influence to the larger-scale variable even when this has no scientific justification.

Scaling changes the **numerical representation**, not the underlying observation.

For example, a blood pressure value can be represented in mmHg or as a standardized z-score. The patient has not changed; only the coordinate system used by the algorithm has changed.

---

## 4. Terminology

The terms *scaling*, *standardization*, and *normalization* are sometimes used inconsistently. It is useful to distinguish them.

### 4.1 Scaling

**Scaling** is the broad term. It refers to transformations that alter the numerical scale of one or more variables.

### 4.2 Standardization

Standardization commonly refers to centering a variable around zero and scaling it by its standard deviation:

\[
z = \frac{x - \mu}{\sigma}
\]

The resulting training variable has approximately:

```text
mean ≈ 0
standard deviation ≈ 1
```

This is the transformation implemented by `StandardScaler` in scikit-learn.

### 4.3 Normalization

The word *normalization* can mean different things.

It is often used informally for **Min-Max scaling**, where variables are mapped to a fixed interval such as `[0, 1]`.

In scikit-learn, however, `Normalizer` refers to a different operation: scaling **each row** to unit norm. That is a sample-wise operation, not the usual feature-wise scaling used for ordinary tabular predictors.

For this reason, it is better to name the exact transformation rather than simply say “normalize the data.”

---

# Part II. When scaling matters

## 5. Algorithms that are sensitive to scale

Scaling is particularly important when an algorithm depends on **distances**, **angles**, **gradients**, or **penalties on coefficient magnitude**.

Common examples include:

| Method | Why scale matters |
|---|---|
| k-nearest neighbours | Uses distances between observations |
| k-means and related clustering | Uses distances to centroids |
| Support Vector Machines | Margins and kernels depend on feature magnitudes |
| Logistic regression with regularization | Penalty acts on coefficient magnitudes |
| Linear regression with regularization | L1/L2 penalties depend on coefficient scale |
| Elastic Net | Combines scale-sensitive L1 and L2 penalties |
| Neural networks | Gradient-based optimization is usually easier with comparable scales |
| PCA | Variance determines component directions |
| t-SNE / UMAP on mixed-scale numerical inputs | Distances are affected by feature scale |
| many gradient-based models | Optimization may converge more reliably |

If one numerical variable ranges from 0 to 100,000 and another ranges from 0 to 1, an Euclidean distance can be dominated by the first variable.

---

## 6. Algorithms that usually do not require feature scaling

Decision-tree-based methods are generally much less sensitive to monotonic feature rescaling because they choose thresholds within individual variables.

Examples include:

```text
Decision Tree
Random Forest
Extra Trees
Gradient-boosted decision trees
XGBoost
LightGBM
CatBoost
```

Scaling these variables is usually not necessary solely for tree training.

This does **not** mean scaling is forbidden. It means scaling is not generally required for the tree-splitting mechanism itself.

A comparative study should therefore avoid assuming that every algorithm benefits from the same numerical representation.

---

## 7. Scaling and interpretability

Scaling changes the units in which coefficients are expressed.

For an unscaled linear model, a coefficient may describe a change in the target per one original unit:

```text
+1 mmHg
+1 year
+1 kg/ha
```

After standardization, the coefficient instead corresponds to a change of approximately one standard deviation in the predictor.

This can be useful for comparing coefficient magnitudes, but the interpretation is different.

The original units should therefore always remain documented.

---

# Part III. Leakage-safe scaling

## 8. The most important rule: fit on TRAIN only

A scaler learns parameters from data.

For example, `StandardScaler` learns the training mean and standard deviation. `MinMaxScaler` learns the training minimum and maximum. `RobustScaler` learns training medians and quantiles.

These quantities must not be estimated using the held-out test set.

The correct sequence is:

```text
full dataset
    ↓
train / test split
    ↓
fit scaler on TRAIN
    ↓
transform TRAIN using training parameters
transform TEST using the same training parameters
```

In code:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The incorrect sequence is:

```python
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, ...)
```

because information from the future test observations has already influenced the transformation.

---

## 9. Why test values may fall outside the training range

Suppose `MinMaxScaler` is fitted on a training variable whose observed range is:

```text
10 to 50
```

Training values are mapped to `[0, 1]`.

If the test set contains a value of 55, its transformed value will be greater than 1. This is not an error. It simply means that the new observation lies outside the range observed while fitting the scaler.

The same principle applies to standardized values. The test set is not expected to have mean exactly zero or standard deviation exactly one.

Those properties describe the transformed **training data**, because the scaling parameters were learned there.

---

# Part IV. Main scaling techniques

## 10. StandardScaler: z-score standardization

The most common standardization is:

\[
z = \frac{x - \mu_{train}}{\sigma_{train}}
\]

where the mean and standard deviation are estimated from the training partition.

### Properties

For the transformed training data:

```text
mean ≈ 0
standard deviation ≈ 1
```

The shape, ordering, skewness, and relative spacing of the distribution are not fundamentally removed. Standardization is an affine transformation.

### Appropriate situations

`StandardScaler` is a strong default for:

- logistic regression;
- linear regression with L1/L2/Elastic Net penalties;
- SVM;
- k-nearest neighbours;
- PCA;
- many neural-network inputs;
- methods where comparable feature variance is desirable.

### Limitation

The mean and standard deviation are sensitive to extreme observations. A few strong outliers can alter both parameters.

Standardization does **not** make a skewed distribution Gaussian and does **not** remove outliers.

---

## 11. MinMaxScaler: mapping to a fixed interval

Min-Max scaling maps a variable to a specified range, usually `[0, 1]`:

\[
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
\]

### Properties

Training values are mapped to the chosen interval while preserving their order and relative spacing.

### Useful situations

Min-Max scaling can be useful when:

- bounded numerical inputs are convenient;
- a neural network benefits from a controlled input range;
- a downstream procedure assumes a particular range;
- the original units differ strongly and a fixed range is desirable.

### Limitation

Minimum and maximum values are highly sensitive to outliers. A single extreme observation can compress the majority of observations into a narrow part of the interval.

Test observations can also legitimately fall outside `[0, 1]` when they exceed the training extrema.

---

## 12. RobustScaler: median and interquartile range

`RobustScaler` commonly centers a variable using the median and scales it using the interquartile range:

\[
x' = \frac{x - \operatorname{median}(x_{train})}{Q_{3,train}-Q_{1,train}}
\]

### Why it is robust

The median and quartiles are less influenced by extreme observations than the mean and standard deviation.

### Useful situations

Robust scaling is useful when:

- numerical variables contain legitimate extreme observations;
- the distribution is heavy-tailed;
- removing extreme values is not scientifically justified;
- a distance- or optimization-based model still requires comparable scales.

### Important limitation

`RobustScaler` does not remove outliers and does not guarantee a symmetric distribution. It merely estimates location and scale using robust statistics.

---

## 13. MaxAbsScaler

`MaxAbsScaler` divides each variable by its maximum absolute training value:

\[
x' = \frac{x}{\max(|x_{train}|)}
\]

Training values usually fall in approximately `[-1, 1]`.

It does not center the data around zero.

This can be useful for sparse matrices because subtracting a mean would destroy sparsity. It is therefore common in some high-dimensional sparse representations.

For ordinary dense biomedical or agricultural tables, StandardScaler or RobustScaler is often more informative.

---

## 14. Unit-norm normalization is different

`Normalizer` operates on rows rather than columns.

For an observation vector \(x\), L2 normalization calculates:

\[
x' = \frac{x}{||x||_2}
\]

Each observation then has norm one.

This is useful in settings where the **direction** of a vector is more important than its magnitude, such as some text-vector and embedding applications.

It is not a routine replacement for feature-wise standardization in ordinary heterogeneous tabular data.

For example, row-normalizing a patient record containing age, BMI, blood pressure, and binary disease indicators usually has no clear scientific interpretation.

---

# Part V. Distribution-shaping transformations

## 15. Scaling does not necessarily change distribution shape

StandardScaler, MinMaxScaler, RobustScaler, and MaxAbsScaler mainly change location and scale. They do not generally remove skewness or transform a multimodal distribution into a Gaussian distribution.

Sometimes the objective is not merely to place variables on comparable scales but to change the **shape** of a distribution.

These are related preprocessing operations, but conceptually different from simple scaling.

---

## 16. Log transformations

For non-negative variables with a long right tail, a logarithmic transformation can reduce skewness:

\[
x' = \log(1+x)
\]

`log1p` is convenient because it is defined at zero.

Possible examples include:

```text
income
read counts
concentration
transaction amounts
some duration variables
```

A logarithm should not be applied automatically. The transformed quantity should still make scientific sense.

---

## 17. PowerTransformer

Scikit-learn provides `PowerTransformer`, including:

- **Yeo-Johnson**, which can accommodate zero and negative values;
- **Box-Cox**, which requires strictly positive values.

These transformations attempt to make distributions more Gaussian-like and stabilize variance.

They can optionally standardize the transformed result.

### When useful

Power transformations may be useful for strongly skewed continuous predictors used by models that benefit from more symmetric numerical distributions.

### Caution

A more Gaussian-looking variable is not automatically a more predictive or scientifically meaningful variable.

---

## 18. QuantileTransformer

`QuantileTransformer` uses the empirical cumulative distribution of the training data and maps values to a chosen output distribution, commonly uniform or normal.

This is a much stronger transformation than simple standardization.

It can reduce the influence of extreme values and create a nearly normal marginal distribution, but it can also substantially alter distances and linear relationships.

It is therefore best treated as a deliberate modelling choice, not as a default preprocessing step.

---

# Part VI. Outliers and scaling

## 19. Scaling does not solve the outlier problem

An extreme value remains an extreme observation after StandardScaler. It simply receives an extreme z-score.

Outlier handling should start with an EDA question:

> Is this observation a data error, an unusual but valid measurement, or evidence of a different subpopulation?

Possible actions include:

- correcting a verified data-entry error;
- removing an observation only when exclusion is scientifically justified;
- winsorization or clipping under a documented rule;
- transforming a skewed variable;
- using robust statistics;
- using RobustScaler;
- retaining the observation and using a model that tolerates it.

The decision should not be based solely on the visual inconvenience of an extreme point.

---

## 20. StandardScaler versus RobustScaler with outliers

Suppose most values lie between 10 and 20, but one observation equals 200.

`StandardScaler` uses the mean and standard deviation, both of which are strongly influenced by 200.

`RobustScaler` uses the median and IQR, which are much less affected.

This does not make RobustScaler universally better. If the distribution is approximately symmetric without problematic extremes, StandardScaler provides a very natural representation.

---

# Part VII. Mixed tabular data

## 21. Numerical and categorical predictors should not be treated identically

A mixed table may contain:

```text
age                         numerical continuous
blood_pressure              numerical continuous
smoker                      binary categorical
cultivar                    categorical
hospital                    categorical
```

Scaling should be based on the **meaning of the variable**, not simply its storage dtype.

A categorical variable encoded as integer codes:

```text
red    -> 0
green  -> 1
blue   -> 2
```

is not a continuous numerical variable. Standardizing those codes would create artificial distances between categories.

Categorical encoding and numerical scaling are separate operations.

---

## 22. Binary 0/1 indicators

Binary indicators such as:

```text
smoker = 0/1
hypertension = 0/1
drought_event = 0/1
```

can often be left unchanged.

This preserves their direct interpretation.

For some scale-sensitive models, standardizing binary indicators is mathematically possible and occasionally useful, but it changes their interpretation and is not automatically required.

In the demonstration notebooks in this repository, we use the simple rule:

```text
continuous / count numerical predictors -> scale
binary categorical indicators           -> keep as 0/1
identifier                               -> never scale
classification label                     -> never scale
```

Later end-to-end modelling notebooks may compare alternative treatments where scientifically justified.

---

## 23. One-hot encoded categorical variables

One-hot encoded columns already have a bounded `0/1` representation.

For many models they can remain unchanged while continuous predictors are standardized.

However, distance-based models raise a broader issue: the relative contribution of a multi-level categorical variable represented by many one-hot columns can differ from that of a single continuous feature. This is not solved simply by blindly standardizing every dummy column.

Representation choices should be model-aware.

---

# Part VIII. Task-specific considerations

## 24. Binary classification

For a binary target such as:

```text
Alzheimer = 0 / 1
```

the **target is not standardized**.

Only eligible numerical predictors are scaled.

A simple demonstration is:

```python
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)
X_test_scaled[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)
```

The label remains unchanged.

Scaling is especially relevant for later demonstrations using logistic regression with regularization, SVM, or k-nearest neighbours.

---

## 25. Multiclass classification

Exactly the same leakage rule applies to a multiclass target such as:

```text
healthy_sleep
insomnia
sleep_apnea
circadian_disruption
```

The class label is not numerically standardized.

Even if software internally encodes classes as integers, those integers represent categories, not a continuous numerical scale.

The numerical predictors can be standardized using training-set statistics, while categorical predictors remain represented appropriately.

Scaling does not use class information. Therefore, the same transformation is applied to observations from every class.

---

## 26. Regression

For regression there are two separate decisions:

```text
scale X predictors?
scale y target?
```

These are not the same operation.

### 26.1 Scaling predictors

Numerical predictors can be standardized in the same way as in classification.

This is useful for regularized linear regression, Elastic Net, k-nearest-neighbour regression, SVM regression, neural networks, and similar models.

### 26.2 Scaling the target

The continuous target may sometimes also be standardized, especially for optimization-sensitive models or neural networks.

For example:

```python
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
)
```

Predictions must then be transformed back to the original units:

```python
y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
```

This is essential for interpretation and for metrics reported in the original units.

### 26.3 When target scaling is unnecessary

Many regression algorithms do not require target scaling. Tree-based models, ordinary linear regression, and many practical workflows can operate directly on the original target.

A target such as crop yield measured in `t/ha` has an interpretable scientific unit. Keeping that unit is often preferable unless the model has a reason to benefit from target transformation.

---

# Part IX. Relationship with EDA and feature engineering

## 27. EDA informs the scaling decision

EDA provides information about:

- scale differences;
- skewness;
- outliers;
- bounded variables;
- long tails;
- zero inflation;
- multimodality;
- class-specific distributions;
- nonlinear relationships with a continuous target.

These observations help decide whether StandardScaler, RobustScaler, a distribution-shaping transform, or no scaling is appropriate.

Scaling should therefore not be selected blindly from habit.

---

## 28. Feature engineering should generally occur before scaling

Suppose we want to calculate:

```text
BMI = weight / height²
pulse pressure = systolic - diastolic
N:P ratio = nitrogen / phosphorus
```

These features have meaningful definitions in the original units.

It is usually clearer to create them **before** standardizing their inputs.

For example, calculating pulse pressure from already-standardized systolic and diastolic pressure would no longer produce a value in mmHg and would represent a different mathematical quantity.

A common conceptual sequence is therefore:

```text
original predictors
      ↓
deterministic feature engineering
      ↓
feature selection / define retained feature space
      ↓
fit scaling parameters on training data
      ↓
scaled model matrix
```

During formal cross-validation, target-dependent feature selection and scaling must both be fitted within each training fold.

---

## 29. Feature selection and scaling can interact

Some feature-selection methods are scale invariant; others are not.

Examples that are largely unaffected by simple positive affine scaling include:

```text
Pearson correlation
Spearman correlation
Mann–Whitney U
ANOVA F statistic under equivalent affine rescaling
chi-square on categorical variables
```

Embedded methods with coefficient penalties, such as L1 logistic regression or Elastic Net, are scale sensitive. Numerical variables should generally be placed on comparable scales before interpreting regularized coefficient selection.

This is why the final end-to-end machine-learning workflow must define the order of operations explicitly.

The present notebooks are only demonstrations of the scaling methods themselves.

---

# Part X. Comparing common methods

## 30. Summary table

| Method | Center | Scale | Outlier sensitivity | Main use |
|---|---|---|---|---|
| StandardScaler | Mean | Standard deviation | High | General scale-sensitive ML |
| MinMaxScaler | Minimum | Range | High | Fixed training range |
| RobustScaler | Median | IQR | Lower | Heavy tails / legitimate extremes |
| MaxAbsScaler | No centering | Maximum absolute value | High | Sparse-compatible scaling |
| Normalizer | Row-wise | Vector norm | Different concept | Direction-based vectors |
| PowerTransformer | Distribution-dependent | Optional standardization | Moderate | Skewness / variance stabilization |
| QuantileTransformer | Quantile mapping | Output distribution | Robust marginal mapping | Strong distribution reshaping |

No method is universally best.

---

## 31. Practical decision guide

A useful starting point is:

```text
approximately symmetric continuous variables
    -> StandardScaler

strong legitimate outliers / heavy tails
    -> consider RobustScaler

need bounded training representation
    -> MinMaxScaler

sparse numerical matrix
    -> consider MaxAbsScaler

strong right skew with meaningful positive scale
    -> investigate log1p or a power transform

very strong marginal distribution reshaping desired
    -> QuantileTransformer, with caution

tree-only model
    -> scaling often unnecessary
```

This is a starting point, not a rulebook.

---

# Part XI. Examples

## 32. StandardScaler example

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_scaled = train.copy()
test_scaled = test.copy()

train_scaled[numerical_columns] = scaler.fit_transform(
    train[numerical_columns]
)

test_scaled[numerical_columns] = scaler.transform(
    test[numerical_columns]
)
```

Notice the distinction:

```text
TRAIN -> fit_transform
TEST  -> transform
```

---

## 33. RobustScaler example

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_robust = scaler.fit_transform(X_train[numerical_columns])
X_test_robust = scaler.transform(X_test[numerical_columns])
```

The scaler learns the training median and quantile range.

---

## 34. MinMaxScaler example

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

X_train_minmax = scaler.fit_transform(X_train[numerical_columns])
X_test_minmax = scaler.transform(X_test[numerical_columns])
```

Do not force test values back into `[0, 1]` simply because they exceed the training range. Such values may contain meaningful information about distribution shift.

---

## 35. Power transformation example

```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(
    method="yeo-johnson",
    standardize=True,
)

X_train_power = transformer.fit_transform(X_train[numerical_columns])
X_test_power = transformer.transform(X_test[numerical_columns])
```

This operation changes more than scale; it also changes marginal distribution shape.

---

# Part XII. What should not be scaled

## 36. Identifiers

Never scale an identifier such as:

```text
patient_id
participant_id
plot_id
```

An identifier is not a quantitative predictor merely because it contains digits.

---

## 37. Classification labels

Do not apply StandardScaler, MinMaxScaler, or similar numerical transformations to binary or multiclass class labels.

Class labels represent categories.

---

## 38. Arbitrary category codes

Suppose categories are encoded as:

```text
control = 0
mild    = 1
severe  = 2
```

If those numbers are merely labels, standardizing them does not create a meaningful continuous variable.

Only genuine ordinal variables with scientifically meaningful ordering should be treated as ordered, and even then the spacing between levels must be considered carefully.

---

# Part XIII. Common mistakes

## 39. Fitting the scaler before the split

Incorrect:

```text
full dataset -> StandardScaler.fit_transform -> split
```

Correct:

```text
split -> fit scaler on train -> transform train and test
```

---

## 40. Scaling every numeric dtype column

An integer dtype does not imply a continuous measurement.

Binary flags, category codes, identifiers, and labels can all be stored as numbers.

Column roles must be defined semantically.

---

## 41. Assuming standardization makes data normal

A z-score transformation changes mean and variance. It does not make a skewed or multimodal distribution Gaussian.

---

## 42. Removing outliers because scaled values look extreme

A z-score of 5 does not automatically imply an invalid observation.

Investigate the observation before changing or removing it.

---

## 43. Scaling the test set independently

Incorrect:

```python
train_scaler.fit_transform(X_train)
test_scaler.fit_transform(X_test)
```

This gives train and test different coordinate systems.

The test set must use parameters learned from training:

```python
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)
```

---

## 44. Forgetting inverse transformation for a scaled regression target

If the model predicts a standardized target, predictions must be returned to the original scale before interpretation.

A crop-yield prediction of `0.83` standardized units is not the final scientific prediction. The value must be converted back to `t/ha`.

---

## 45. Scaling because “machine learning requires it”

Machine learning does not have a universal scaling requirement.

The correct question is:

> Does the selected algorithm or representation depend meaningfully on numerical scale?

---

# Part XIV. Recommended educational workflow

## 46. Simple demonstration workflow

The notebooks accompanying this document intentionally use explicit objects and intermediate DataFrames:

```text
load engineered data
      ↓
split train / test
      ↓
identify continuous numerical columns
      ↓
fit one scaler on train
      ↓
transform train and test
      ↓
inspect transformed summaries
      ↓
repeat with alternative scalers
```

This makes every learned quantity visible.

We deliberately do **not** hide these operations inside a pipeline yet.

---

## 47. Later full preparation workflow

Once each operation is understood independently, the later end-to-end preparation notebook can combine:

```text
raw analytical feature set
        ↓
train / validation / test logic
        ↓
imputation if required
        ↓
encoding
        ↓
feature engineering
        ↓
feature selection
        ↓
scaling
        ↓
model-ready matrix
```

Formal cross-validation will require all learned operations to be refitted within training folds.

The educational goal is to understand the individual operations **before** automating their composition.

---

# Part XV. Checklist

## 48. Before scaling

Confirm that:

- numerical and categorical variables have been identified semantically;
- identifiers are excluded;
- the target role is known;
- missing values have been addressed appropriately;
- obvious data-quality problems have been investigated;
- feature engineering formulas have been defined in meaningful original units;
- the train/test split has been created before fitting any data-dependent scaler.

## 49. When fitting the scaler

Confirm that:

- only training observations are used to estimate scaling parameters;
- only eligible numerical predictors are supplied;
- the same fitted object transforms validation and test data;
- categorical indicators are intentionally treated, not scaled by accident;
- the selected scaler is justified by the data distribution and later algorithm.

## 50. After scaling

Check that:

- `StandardScaler` training means are close to zero;
- `StandardScaler` training standard deviations are close to one;
- Min-Max training ranges match the requested interval;
- RobustScaler training medians are close to zero when centering is enabled;
- test statistics are **not** expected to exactly match training statistics;
- classification labels are unchanged;
- regression target scaling, if used, has an inverse transformation available;
- original units and column names remain documented.

---

# 51. Take-home message

Scaling is not a cosmetic operation. It changes the coordinate system in which a model sees the data.

The central principles are:

```text
scale only variables for which scaling is meaningful
choose the transformation according to the data and algorithm
fit transformation parameters on training data only
use the same fitted transformation for future observations
keep classification labels untouched
do not confuse feature scaling with distributional correction or outlier removal
```

For the three tabular tasks in this repository, the accompanying notebooks demonstrate these principles separately for:

```text
Task 1 -> binary classification
Task 2 -> multiclass classification
Task 3 -> regression with a continuous target
```

The next stage will combine these independently understood operations into a complete, end-to-end machine-learning preparation workflow.
