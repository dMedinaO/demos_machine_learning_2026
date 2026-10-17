# Dataset Splitting and Validation Strategies for Tabular Machine Learning

## 1. Purpose of this document

A machine-learning model is useful only if it can generalize to observations that were not used to construct it. Dataset partitioning is therefore not a clerical step performed immediately before model training. It defines the **generalization question** that the experiment is trying to answer.

A split determines which observations are available for fitting, which observations may be used for model selection, and which observations remain unseen until the final evaluation. A weak split can make a poor model appear excellent. A rigorous split can expose limitations that would otherwise remain hidden.

This document explains:

- the roles of training, validation, and test data;
- when a simple train/test split is sufficient;
- when train/validation/test partitions are preferable;
- common random and stratified splitting strategies;
- how partitioning differs for binary classification, multiclass classification, and regression;
- how cross-validation relates to holdout sets;
- what to do when the dataset is small;
- how to inspect whether the resulting partitions are reasonable;
- how data leakage can occur during splitting;
- why group-aware, cluster-aware, temporal, spatial, distance-aware, homology-aware, and related strategies are sometimes necessary;
- why those advanced strategies are important even though the accompanying notebooks focus on common tabular splitting methods.

The accompanying notebooks are intentionally **simple demonstrations**. They do not build end-to-end machine-learning pipelines and they do not train predictive models. Their purpose is to create and inspect partitions.

---

# Part I. Why splitting exists

## 2. Training performance is not generalization performance

Suppose a model is fitted using every observation in a dataset and then evaluated on those same observations. The resulting score answers only the following question:

> How well can the model reproduce data it has already seen?

That is not the question we normally care about.

The more useful question is:

> How well does the fitted procedure perform on new observations drawn from the population or deployment scenario of interest?

A held-out partition approximates this second question.

The central idea is therefore:

```text
available observations
        ↓
separate observations used for learning
from observations used for evaluation
```

---

## 3. The three common data roles

### 3.1 Training set

The **training set** is used to learn model parameters and any data-dependent transformation required by the model.

Examples include:

- imputation values;
- scaling parameters;
- encoding parameters;
- feature-selection decisions;
- PCA loadings used for modelling;
- model coefficients;
- tree structures;
- neural-network weights.

In a formal workflow, these quantities are estimated from training data only.

### 3.2 Validation set

The **validation set** is used for decisions made during model development.

Typical uses include:

- choosing hyperparameters;
- selecting among algorithms;
- choosing a probability threshold;
- deciding when to stop neural-network training;
- selecting a model checkpoint;
- comparing alternative preprocessing strategies;
- choosing among candidate feature sets.

The validation set is not truly unseen if it is consulted repeatedly while developing the model.

### 3.3 Test set

The **test set** is reserved for final evaluation.

Ideally, it is not used to:

- choose features;
- choose a scaler;
- select hyperparameters;
- choose a model family;
- set a classification threshold;
- decide whether to continue modifying the workflow.

The test set is intended to approximate an independent evaluation.

A useful mental model is:

```text
TRAIN       learn
VALIDATION  decide
TEST        evaluate
```

---

# Part II. Train/test versus train/validation/test

## 4. When train/test is enough

A two-way split is often sufficient when model selection is handled entirely inside the training data.

For example:

```text
full dataset
    ↓
TRAIN 80%
TEST  20%
```

The training partition can then use cross-validation:

```text
TRAIN
  ↓
5-fold cross-validation
  ↓
hyperparameter/model selection
  ↓
refit using all TRAIN
  ↓
final evaluation on TEST
```

This strategy is attractive when data are limited because a separate validation set would remove additional observations from model fitting.

A train/test design is also appropriate for simple demonstrations where no tuning or model selection is performed.

---

## 5. When train/validation/test is useful

A dedicated validation set is useful when model development repeatedly needs a fixed held-out subset.

Examples include:

- deep-learning early stopping;
- manual model comparison;
- iterative threshold tuning;
- checkpoint selection;
- repeated inspection of validation predictions;
- workflows where cross-validation is not practical.

A common illustrative split is:

```text
TRAIN       70%
VALIDATION  15%
TEST        15%
```

Other ratios such as `60/20/20` or `80/10/10` may also be reasonable.

There is no universally correct ratio. The appropriate choice depends on:

- total sample size;
- class frequencies;
- expected model complexity;
- the amount of tuning;
- computational resources;
- the cost of obtaining additional observations;
- whether an external test dataset exists.

---

## 6. Validation data are not a substitute for a test set

Repeatedly inspecting validation performance can lead to indirect overfitting to the validation set.

If ten modelling decisions are made after looking at the same validation scores, the final workflow has effectively been optimized for that validation sample.

This is why a final independent test set remains useful.

---

# Part III. Random splitting

## 7. Random holdout splitting

The simplest strategy randomly assigns observations to partitions.

For example:

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
)
```

Random splitting assumes that observations can reasonably be treated as exchangeable samples from the same population.

This assumption can be appropriate for ordinary independent tabular observations, but it can fail badly when the dataset contains groups, repeated measurements, temporal structure, related biological sequences, spatial dependence, or other forms of non-independence.

---

## 8. Reproducibility and random seeds

Random partitions should normally be reproducible.

A fixed `random_state` makes it possible to regenerate the same assignment:

```python
RANDOM_STATE = 42
```

The seed should be recorded together with:

- partition sizes;
- sample identifiers;
- splitting method;
- stratification variable when used;
- grouping variable when used;
- software configuration when reproducibility is critical.

A seed does not make a split scientifically correct. It only makes the random procedure reproducible.

---

# Part IV. Stratified splitting for classification

## 9. Why random splitting may alter class composition

Suppose a binary dataset contains:

```text
80% negative
20% positive
```

A purely random split will usually produce approximately the same proportions, but not exactly. With small datasets or rare classes, the difference can become substantial.

A validation set could accidentally contain very few minority-class examples, making evaluation unstable.

---

## 10. Stratified splitting

**Stratification** attempts to preserve the class proportions of the complete dataset in each partition.

For binary classification:

```python
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42,
)
```

The same idea applies naturally to multiclass classification.

If the original distribution is:

```text
class A  40%
class B  30%
class C  20%
class D  10%
```

stratification aims for similar proportions in TRAIN and TEST.

---

## 11. Stratification does not balance the dataset

This distinction is important.

If a dataset is:

```text
90% class 0
10% class 1
```

then a stratified split will produce partitions that are also approximately:

```text
90% class 0
10% class 1
```

Stratification **preserves imbalance**. It does not correct it.

Class balancing, resampling, class weighting, or threshold adjustment are separate modelling decisions.

---

## 12. Stratified train/validation/test splitting

Scikit-learn does not require a special three-way function. A three-way split can be created sequentially.

For example, to create approximately `70/15/15`:

```python
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)
```

The first split protects the training fraction. The second divides the remaining observations into validation and test sets while preserving class proportions again.

---

## 13. Rare classes and stratification

Stratification requires enough observations from each class to populate the requested partitions.

If a class contains only three observations, it is impossible to create large numbers of representative folds or holdout subsets.

When classes are extremely rare, possible responses include:

- collecting more data;
- reducing the number of partitions;
- using cross-validation designed for limited data;
- merging categories only when scientifically defensible;
- reporting uncertainty rather than pretending the estimate is stable.

The software error produced by an impossible stratified split is often revealing a real experimental-design limitation.

---

# Part V. Regression splitting

## 14. A continuous target cannot be directly stratified

For regression, the target may contain hundreds or thousands of distinct numerical values.

Therefore, code such as:

```python
train_test_split(X, y, stratify=y)
```

is generally inappropriate for a continuous target.

The default approach for an ordinary independent regression dataset is commonly a **random split**.

---

## 15. What should be preserved in regression?

Although exact stratification is not defined for a continuous target, it is still useful to inspect whether partitions cover similar regions of the target distribution.

Useful summaries include:

- minimum and maximum;
- mean and median;
- standard deviation;
- quartiles;
- low and high quantiles;
- histograms or KDE curves;
- empirical cumulative distributions;
- the number of very low or very high target values.

The purpose is not to force identical partitions. Small differences are expected.

The purpose is to detect problematic accidents such as:

```text
TRAIN contains almost all high-yield observations
TEST contains almost no high-yield observations
```

unless such a shift is deliberately part of the evaluation design.

---

## 16. Approximate stratification through target bins

A sometimes-used technique is to divide the continuous target into quantile bins and stratify using those bins.

For example:

```python
bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
```

The bin labels may then be used only for the split assignment.

The original regression target remains continuous.

Conceptually:

```text
continuous target
      ↓
temporary quantile bins
      ↓
stratified split using bins
      ↓
discard bins
      ↓
model the original continuous target
```

This can make holdout target distributions more similar, especially for small datasets or strongly skewed targets.

However, it is not automatically superior to random splitting.

Potential disadvantages include:

- arbitrary dependence on the number of bins;
- insufficient observations in some bins;
- a misleading impression that target distribution is the only structure that matters;
- conflict with groups, temporal order, or spatial structure.

It should therefore be treated as an optional sampling strategy, not a universal rule.

---

## 17. Regression with natural structure

If observations are grouped by farm, patient, year, site, device, subject, or another dependency, preserving the marginal target distribution may be less important than respecting that structure.

For example, a crop-yield dataset collected across years might require:

```text
train = earlier years
test  = future year
```

rather than a random split across rows.

The correct split follows the intended deployment scenario.

---

# Part VI. Cross-validation

## 18. Why cross-validation exists

A single holdout split can be unusually easy or difficult by chance.

Cross-validation repeatedly changes which observations are used for fitting and validation so that performance can be estimated across multiple partitions.

In ordinary `K`-fold cross-validation:

```text
fold 1  VALIDATION | TRAIN TRAIN TRAIN TRAIN
fold 2  TRAIN | VALIDATION | TRAIN TRAIN TRAIN
fold 3  TRAIN TRAIN | VALIDATION | TRAIN TRAIN
...
```

Each observation serves as validation data once.

---

## 19. KFold and StratifiedKFold

For regression, ordinary independent data are often evaluated with:

```python
from sklearn.model_selection import KFold
```

For classification, a common choice is:

```python
from sklearn.model_selection import StratifiedKFold
```

which approximately preserves class proportions in each fold.

---

## 20. Repeated cross-validation

When performance estimates are sensitive to a particular fold assignment, repeated cross-validation can sample several fold configurations.

Examples include:

```text
RepeatedKFold
RepeatedStratifiedKFold
```

This produces a distribution of performance estimates rather than relying on one partition realization.

It also costs more computationally.

---

## 21. Nested cross-validation

If hyperparameters are tuned and performance must be estimated without optimistic bias, nested cross-validation separates:

```text
outer loop  → evaluation
inner loop  → model/hyperparameter selection
```

Nested CV is especially useful in small datasets where allocating a large permanent validation set would waste too much training data.

It is more computationally demanding and will be treated later in the modelling workflow rather than in the simple splitting notebooks.

---

# Part VII. Small datasets

## 22. Why small datasets are difficult to partition

Suppose a dataset contains only 100 observations.

A `70/15/15` split gives approximately:

```text
70 training observations
15 validation observations
15 test observations
```

A metric estimated from 15 test observations can be extremely unstable.

For classification, the situation becomes worse if the minority class is rare.

For regression, a tiny test set may inadequately cover the target range.

---

## 23. There is no universal definition of “small”

Whether a dataset is small depends on:

- dimensionality;
- class balance;
- target variability;
- model complexity;
- expected effect size;
- biological or experimental heterogeneity;
- the required precision of the final estimate.

A dataset with 500 observations may be generous for one problem and severely limited for another.

Therefore, avoid universal thresholds such as:

```text
n < 1000 means small
```

without context.

---

## 24. Practical strategies for limited data

When observations are scarce, useful designs include:

### Strategy A — Train/test plus cross-validation inside TRAIN

```text
full data
   ↓
TRAIN 80%
TEST  20%

TRAIN
   ↓
cross-validation for model selection
```

This avoids spending additional observations on a permanent validation set.

### Strategy B — Nested cross-validation

Useful when the dataset is too small for a stable external holdout and a rigorous estimate of the full model-selection procedure is needed.

### Strategy C — Repeated cross-validation

Useful for describing uncertainty due to partition realization.

### Strategy D — External test data

If a genuinely independent external dataset exists, the internal dataset can be used more efficiently for training and cross-validation.

---

## 25. Leave-one-out cross-validation

Leave-one-out cross-validation uses one observation as validation and all others as training in each iteration.

It maximizes training-set size but is not automatically the best method for small data.

Potential problems include:

- high computational cost;
- high variance of some performance estimates;
- strong dependence among fitted models;
- incompatibility with certain grouped or structured datasets.

It should be selected for a reason, not simply because the dataset is small.

---

# Part VIII. Advanced partitioning strategies

## 26. Why random splitting can be misleading

Rows in a table are not always independent.

Two observations may be strongly related because they represent:

- repeated measurements from the same person;
- samples from the same patient;
- proteins from the same homologous family;
- compounds sharing a molecular scaffold;
- plots from the same farm;
- images from the same subject;
- measurements from neighboring locations;
- observations collected on adjacent dates.

If related observations are placed in both TRAIN and TEST, the evaluation may measure memorization of shared structure rather than genuine generalization.

This motivates structure-aware splitting.

---

## 27. Group-aware splitting

When observations belong to known groups, entire groups can be assigned to one partition.

Examples:

```text
patient ID
family ID
hospital
farm
experimental batch
protein family
```

Common tools include:

```text
GroupShuffleSplit
GroupKFold
LeaveOneGroupOut
StratifiedGroupKFold
```

The key rule is:

> observations from the same protected group should not be divided across training and evaluation partitions.

---

## 28. Cluster-aware splitting

Sometimes natural groups are not supplied directly and must first be inferred from similarity.

A **cluster-aware split** can:

1. compute or obtain a similarity representation;
2. cluster related observations;
3. assign complete clusters to partitions.

This reduces the chance that highly similar observations appear on both sides of the split.

The scientific meaning depends entirely on how similarity and clustering are defined.

---

## 29. Distance-aware splitting

A **distance-aware split** explicitly uses distances or similarities between observations to control how close evaluation examples may be to training examples.

The goal is often to create a harder and more realistic generalization regime.

For example:

```text
random split
    may allow TEST observations very close to TRAIN

distance-aware split
    restricts that proximity
```

Distance may be defined using:

- raw numerical features;
- learned embeddings;
- sequence representations;
- structural descriptors;
- domain-specific similarity measures.

The resulting benchmark can answer a different question from a random split.

---

## 30. Homology-aware splitting

Homology-aware splitting is particularly important for proteins, peptides, nucleic acids, and related biological sequences.

Closely homologous sequences can share function and local sequence patterns. If homologs appear in both training and test sets, model performance may be inflated.

A homology-aware workflow may:

1. compute sequence similarity;
2. construct sequence clusters at a chosen identity or coverage criterion;
3. assign whole clusters to partitions;
4. verify residual train-test similarity.

The exact procedure depends on the biological question and similarity definition.

A model evaluated under random sequence splitting and the same model evaluated under homology-aware splitting are not necessarily being tested on the same generalization problem.

---

## 31. Scaffold-aware splitting

In molecular machine learning, compounds sharing a molecular scaffold can be strongly related.

A scaffold-aware split assigns related chemical scaffolds to one side of the partition so that the test set better approximates generalization toward new chemical families.

This is analogous in spirit to homology-aware splitting for biological sequences.

---

## 32. Temporal splitting

When the intended deployment is future prediction, time order should often be respected.

For example:

```text
TRAIN       2018–2023
VALIDATION  2024
TEST        2025
```

Randomly mixing 2025 observations into the training set would not reproduce the intended future-prediction problem.

`TimeSeriesSplit` is one common implementation pattern, although real temporal studies may require customized windows.

---

## 33. Spatial splitting

Geographical observations can be strongly autocorrelated.

Random splitting may place neighboring locations in both training and test sets.

Spatial blocking or region-aware splitting can instead ask whether the model generalizes to new geographical areas.

This may be relevant for:

- ecology;
- remote sensing;
- agriculture;
- epidemiology;
- environmental modelling.

---

## 34. Which constraint has priority?

A common mistake is to preserve class balance while violating the true dependency structure.

Suppose observations come from repeated measurements of the same patients. A perfectly stratified row-level split that places the same patient in TRAIN and TEST is usually inferior to a slightly less balanced patient-level split.

In many applications:

```text
dependency constraint
        >
perfect marginal stratification
```

The split must reflect the deployment question first.

---

# Part IX. What the current notebooks will implement

## 35. Scope of the demonstrations

The current tabular datasets are synthetic and contain independent rows. Therefore, the accompanying notebooks focus on the most common introductory strategies.

### Task 1 — Binary Alzheimer classification

The notebook demonstrates:

```text
random train/test
stratified train/test
stratified train/validation/test
```

and compares their class distributions.

### Task 2 — Multiclass sleep-profile classification

The notebook demonstrates the same strategies with four classes and shows why multiclass stratification is especially useful when class proportions differ.

### Task 3 — Crop-yield regression

The notebook demonstrates:

```text
random train/test
random train/validation/test
optional quantile-bin stratification
```

without converting the regression target into a classification target.

Distance-aware, homology-aware, cluster-aware, group-aware, temporal, and spatial splitting are described in this document but are intentionally not implemented in these introductory notebooks.

---

# Part X. Diagnosing a split

## 36. A split should be inspected, not merely created

After creating partitions, verify at minimum:

```text
1. partition sizes are correct
2. no identifier appears in more than one partition
3. all intended observations are assigned
4. target composition is reasonable
5. predictor distributions are not accidentally pathological
6. the split respects any required groups or structure
```

A successful function call does not guarantee a scientifically sensible split.

---

## 37. Check partition overlap

For identifier-based tables:

```python
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
```

Then verify complete coverage:

```python
all_split_ids = set(train_ids) | set(val_ids) | set(test_ids)
assert all_split_ids == set(df[id_column])
```

These simple checks prevent surprisingly common bookkeeping errors.

---

## 38. Classification target diagnostics

For classification, inspect counts and proportions:

```text
                 TRAIN   VALIDATION   TEST
class A          ...       ...        ...
class B          ...       ...        ...
```

Useful plots include:

- grouped bar plots of class prevalence;
- count plots;
- differences in percentage points relative to the full dataset.

The objective is usually similarity, not exact equality.

---

## 39. Regression target diagnostics

For regression, inspect:

- mean;
- median;
- standard deviation;
- interquartile range;
- lower and upper quantiles;
- complete target distributions.

Useful plots include:

```text
histogram + KDE
ECDF
boxplot
violin plot
```

Distribution distances such as a Kolmogorov-Smirnov statistic or Wasserstein distance can be useful descriptive diagnostics, but they should not become automatic pass/fail criteria without context.

---

## 40. Numerical predictor diagnostics

Even when the target is well preserved, predictor space may differ by chance.

For a numerical feature, a simple standardized mean difference relative to TRAIN is:

\[
\mathrm{SMD} = \frac{\bar{x}_{split} - \bar{x}_{train}}{s_{train}}
\]

This expresses the difference in units of the training standard deviation.

Small absolute values indicate similar means, but there is no universal threshold that proves two partitions are equivalent.

SMD also does not detect every distributional difference.

It is one diagnostic among several.

---

## 41. Categorical predictor diagnostics

For binary categorical predictors, compare prevalence:

```text
P(feature = 1 | TRAIN)
P(feature = 1 | VALIDATION)
P(feature = 1 | TEST)
```

Differences can be expressed in percentage points relative to TRAIN.

For general categorical variables, compare the complete category-frequency distribution rather than only one category.

---

## 42. Multivariate space diagnostics

Univariate summaries can look similar even when multivariate structure differs.

A useful exploratory diagnostic is to project observations into a lower-dimensional space.

For example:

```text
TRAIN numerical predictors
        ↓
fit StandardScaler on TRAIN
        ↓
fit PCA on TRAIN
        ↓
transform TRAIN, VALIDATION, TEST
        ↓
plot PC1 versus PC2 by partition
```

Important: PCA used in this way is a **diagnostic visualization**, not proof that the distributions are identical.

If PCA will later be part of the predictive model, it must also be fitted within the training data or training fold.

---

## 43. Why fit the diagnostic projection on TRAIN?

If PCA is fitted using the complete dataset, the test set influences the projection directions.

For a purely descriptive publication figure this may sometimes be acceptable if clearly stated, but when the purpose is to demonstrate leakage-safe machine-learning practice it is better to fit the transformation using TRAIN and only transform held-out observations.

The notebooks follow that rule.

---

# Part XI. Splitting and data leakage

## 44. Split before learning data-dependent preprocessing

The conceptual machine-learning order is:

```text
raw / curated dataset
       ↓
define TRAIN / VALIDATION / TEST
       ↓
learn preprocessing from TRAIN
       ↓
apply learned transformations to held-out data
       ↓
fit models using TRAIN
       ↓
use VALIDATION or cross-validation for decisions
       ↓
final TEST evaluation
```

Operations that commonly learn from data include:

- imputation;
- scaling;
- target encoding;
- feature selection;
- supervised dimensionality reduction;
- PCA used as a modelling transformation;
- resampling procedures such as SMOTE.

---

## 45. Why earlier notebooks sometimes used the full dataset

The notebook sequence in this repository is pedagogical.

EDA notebooks inspect the available synthetic datasets so that students can learn the techniques. Feature-engineering notebooks demonstrate candidate construction and then restrict supervised feature selection to a training partition. Scaling notebooks create their own training split before fitting scalers.

The later **full preparation for machine learning** notebook will combine these ideas in the proper operational order.

Therefore:

> notebook number is a teaching sequence, not a claim that every operation should literally be fitted to the full dataset before splitting.

---

## 46. Leakage through duplicated or related observations

Even a technically correct `train_test_split` can leak information when duplicate or near-duplicate observations occur across partitions.

Examples include:

- duplicate patients;
- replicated measurements;
- homologous proteins;
- nearly identical peptides;
- augmented versions of the same image;
- repeated chemical scaffolds.

This is one reason advanced structure-aware splitting exists.

---

# Part XII. Choosing a strategy by task

## 47. Binary classification

A practical introductory default is:

```text
Stratified train/test split
```

or:

```text
Stratified train/validation/test split
```

when a fixed validation set is needed.

For model selection with limited data:

```text
stratified train/test
+
StratifiedKFold inside TRAIN
```

is often efficient.

---

## 48. Multiclass classification

Stratification remains useful, but inspect **every class**, not only the majority class.

Potential problems include:

- rare classes disappearing from a partition;
- tiny validation counts for one class;
- apparently stable global accuracy despite unstable minority-class evaluation.

The minimum class frequency often constrains the number of feasible folds.

---

## 49. Regression

For ordinary independent regression data, a common starting point is:

```text
random holdout
```

followed by inspection of the target distribution and predictor space.

Possible alternatives include:

```text
KFold
RepeatedKFold
quantile-bin stratified holdout
```

when appropriate.

If the task contains groups, time, space, or repeated subjects, those structures should usually determine the split instead.

---

# Part XIII. Common mistakes

## 50. Splitting after scaling

Incorrect:

```text
full data
→ fit scaler
→ split
```

Correct:

```text
split
→ fit scaler on TRAIN
→ transform held-out data
```

---

## 51. Feature selection before splitting

Incorrect:

```text
use every label to select the best features
→ split
→ evaluate
```

The test labels have already influenced the feature space.

Supervised feature selection belongs inside training data.

---

## 52. Repeatedly checking TEST

If test performance is consulted after every change, the test set gradually becomes another validation set.

A clean workflow keeps TEST hidden until the workflow is frozen.

---

## 53. Using only one metric from one split

A single score from one random seed can exaggerate certainty.

Depending on the study, consider:

- multiple metrics;
- confidence intervals;
- repeated splits;
- cross-validation distributions;
- external validation.

---

## 54. Forcing the test set to look perfect

A test set does not need to reproduce every training statistic exactly.

Random sampling naturally creates differences.

Repeatedly changing the seed until a particularly convenient split appears is a form of selection and can bias evaluation.

The split strategy should be defined first and then documented.

---

## 55. Ignoring the deployment scenario

The statistically neatest split is not always scientifically correct.

Examples:

```text
future deployment       → temporal split may matter
new patients            → patient-group split
new protein families    → homology-aware split
new chemical scaffolds  → scaffold split
new farms/regions       → group/spatial split
```

The evaluation should reproduce the intended form of novelty.

---

# Part XIV. Reproducibility

## 56. Save split assignments

For reproducible experiments, it is often useful to export an explicit assignment table:

```text
sample_id   split
P0001       train
P0002       train
P0003       validation
P0004       test
```

This is more robust than relying only on a random seed because it records the realized partition directly.

The notebooks export such assignment files for demonstration.

---

## 57. Record the splitting configuration

Useful metadata include:

```text
strategy
random_state
train fraction
validation fraction
test fraction
stratification variable
group variable
number of folds when applicable
```

For advanced studies, also record similarity thresholds, clustering parameters, temporal cutoffs, or homology criteria.

---

# Part XV. Practical decision guide

## 58. If the task is binary classification

Start with:

```text
stratified holdout
```

if rows are independent.

Use a fixed validation set when repeated model-development decisions require one. Otherwise, consider cross-validation within TRAIN.

---

## 59. If the task is multiclass classification

Start with:

```text
stratified holdout
```

and inspect counts for every class.

If one class is extremely rare, the problem may require more data or a different validation design.

---

## 60. If the task is regression

Start with:

```text
random holdout
```

for independent observations.

Inspect the target distribution in every partition.

Consider quantile-bin stratification only when it solves a concrete sampling problem and no stronger grouping constraint applies.

---

## 61. If observations have groups

Use a group-aware method.

Do not split rows from the same protected group across training and evaluation merely to improve class balance.

---

## 62. If deployment is temporal

Use a time-respecting split.

Do not train on future observations to predict the past.

---

## 63. If observations are highly similar biological sequences

Consider homology-aware, cluster-aware, or distance-aware evaluation.

A random split may measure interpolation among close relatives rather than generalization to new sequence space.

---

## 64. If data are scarce

Prefer designs that use training observations efficiently:

```text
train/test + cross-validation inside TRAIN
```

or a carefully designed nested/repeated cross-validation procedure.

Do not create a tiny permanent validation set simply because `70/15/15` is common in tutorials.

---

# Part XVI. Checklist

## 65. Before splitting

Ask:

- What form of generalization do I want to measure?
- Are rows independent?
- Are there repeated subjects or groups?
- Is there temporal or spatial structure?
- Could near-duplicates cross partitions?
- Is class stratification appropriate?
- Is an external test set available?

## 66. After splitting

Verify:

- partition sizes;
- identifier disjointness;
- complete sample coverage;
- class counts for classification;
- target distribution for regression;
- numerical predictor differences;
- categorical prevalence differences;
- multivariate overlap when useful;
- reproducibility metadata.

## 67. Before final evaluation

Verify:

- TEST has not influenced feature selection;
- TEST has not influenced scaling or imputation;
- TEST has not been used to tune hyperparameters;
- any grouping or similarity constraints remain satisfied;
- the split still represents the intended deployment scenario.

---

# 68. Final perspective

Dataset splitting is part of the scientific definition of a machine-learning experiment.

For simple independent tabular data, random and stratified partitions are often appropriate and easy to understand. For regression, random splitting combined with careful target-distribution diagnostics is a reasonable starting point. Cross-validation can use limited training data more efficiently and provides information about partition variability.

More complex domains require more careful definitions of independence. Group-aware, cluster-aware, temporal, spatial, scaffold-aware, distance-aware, and homology-aware strategies exist because many datasets violate the assumptions underlying ordinary row-wise random splitting.

The central principle is:

> **The split should reproduce the kind of novelty that the trained model will face in practice.**

A useful workflow is therefore:

```text
define the deployment question
        ↓
identify dependencies and leakage risks
        ↓
choose the splitting strategy
        ↓
generate reproducible partitions
        ↓
inspect target and predictor space
        ↓
freeze the evaluation design
        ↓
fit all data-dependent operations using training data only
```

The current notebooks stop at partition creation and diagnostics. The next end-to-end preparation notebook will combine splitting with the other preparation steps required before model training.
