# End-to-End Preparation of Tabular Data for Machine Learning

## 1. Purpose of this document

The previous tabular-data demonstrations treated preprocessing, exploratory data analysis, feature engineering, feature selection, scaling, and dataset splitting as separate topics. That separation is useful while learning each concept, but a real machine-learning workflow must connect these decisions without allowing information from validation or test observations to influence the fitted preparation steps.

This document describes the **decision logic of an end-to-end preparation workflow**. It intentionally does not repeat the detailed theory of every preprocessing, feature-selection, scaling, or splitting technique; those topics are documented separately. The focus here is how the pieces are ordered, what is learned from which data, what must be saved, and what should remain reusable when new observations arrive later.

The accompanying notebooks implement this workflow for three tabular tasks:

- Task 1: binary classification of synthetic Alzheimer status;
- Task 2: multiclass classification of synthetic sleep profiles;
- Task 3: regression of synthetic crop yield.

The notebooks stop at **model-ready data**. They do not train predictive models. Model fitting, validation, comparison, and final testing are the next stage.

---

# Part I. The central design decision

## 2. Separate deterministic preparation from fitted preparation

A useful machine-learning preparation workflow contains two conceptually different layers.

### 2.1 Deterministic preparation

Some operations are defined by the data contract or by domain knowledge and do not estimate parameters from the observed sample. Examples include:

- standardizing column names;
- checking the required schema;
- converting declared numerical fields to numerical types;
- checking identifier uniqueness;
- validating binary indicators;
- constructing a predefined ratio, difference, count, or interaction;
- excluding an identifier from the predictor matrix.

These operations should be reproducible and should behave identically for training data and future external data.

### 2.2 Fitted preparation

Other operations estimate quantities from the available observations. Examples include:

- median or mode values used for imputation;
- means and standard deviations used by a scaler;
- category vocabularies learned by an encoder;
- feature-selection scores and selected variables;
- PCA loadings when PCA is part of a predictive workflow.

These operations **must be learned from training data only**.

The distinction can be summarized as:

```text
raw table
   ↓
deterministic schema cleaning
   ↓
deterministic feature construction
   ↓
TRAIN / VALIDATION / TEST split
   ↓
fit data-dependent preparation on TRAIN only
   ↓
transform TRAIN, VALIDATION, and TEST
```

The same fitted object can later transform genuinely new observations.

---

# Part II. Decide the evaluation structure before fitting transformations

## 3. The split is part of the experimental design

The notebooks use a fixed train/validation/test demonstration because the next teaching block will explicitly separate model training, model selection, and final testing.

For the classification tasks, the demonstration uses a stratified 70/15/15 split. Stratification preserves the observed class composition while keeping the three partitions disjoint.

For the regression task, the demonstration uses a random 70/15/15 split. The continuous target is not converted into a categorical target merely to make the split look more balanced. Quantile-based stratification remains an optional design choice when the application justifies it.

These fractions are **not universal defaults**. With limited data, reserving both a validation set and a test set can make the training sample unnecessarily small. A common alternative is:

```text
TRAIN + cross-validation
TEST held out once
```

When hyperparameter tuning uses cross-validation, all fitted preparation steps must be refitted inside each training fold. A preparation object fitted once on the complete training partition must not be used to preprocess every fold before cross-validation, because that would leak fold-level information.

The fixed 70/15/15 exports in these notebooks are therefore appropriate for later **fixed-split demonstrations**. Formal cross-validation should combine an unfitted preparation template with the estimator so the complete procedure is refitted inside each fold.

---

# Part III. Decide what belongs in X and what does not

## 4. Identifiers are carried for traceability, not prediction

Each dataset has a unique identifier:

- `patient_id`;
- `participant_id`;
- `plot_id`.

The identifier is preserved in split manifests and model-ready exports so predictions can later be traced back to observations. It is not included in the predictor matrix.

This distinction is important. A column may be operationally essential without being a valid predictor.

## 5. Targets remain outside predictor preparation

The target is separated before fitted predictor transformations are learned.

For binary classification, the Alzheimer target is already represented numerically as 0/1.

For multiclass classification, the human-readable sleep-profile labels are preserved and a fitted target encoder is saved as a separate artifact. The encoder is not treated as a predictor transformation.

For regression, crop yield remains in its original units (`t/ha`) in this preparation workflow. Target scaling is postponed until a model requires it. If target scaling is later used, the fitted target transformation must be saved and predictions must be returned to the original unit before interpretation.

---

# Part IV. Feature engineering decisions

## 6. Reuse explicit feature recipes

The end-to-end notebooks reuse a small set of interpretable engineered variables introduced in the feature-engineering demonstrations. The important decision is not the exact formula; it is that the formulas are **explicit, deterministic, documented, and identical for every partition and future dataset**.

Feature engineering should not be improvised separately for training and test data.

A useful boundary is:

```text
raw predictors
      ↓
known deterministic feature recipe
      ↓
candidate predictor table
      ↓
fitted statistical preparation
```

If a future production system becomes more complex, the deterministic feature recipe should be moved from a notebook into tested reusable source code. In these teaching notebooks it remains visible so students can inspect every step.

---

# Part V. A reusable scikit-learn preparation object

## 7. Why use a preparation pipeline here

Earlier notebooks intentionally demonstrated scaling and related operations without pipelines. The end-to-end workflow is different: the goal is now to preserve a fitted transformation that can be applied consistently later.

A scikit-learn preparation pipeline provides three important properties:

1. the order of fitted transformations is explicit;
2. fitting occurs through one interface;
3. the fitted object can be serialized and reused.

The demonstration pipeline contains a numerical branch and a binary-categorical branch. Numerical predictors are imputed when necessary and standardized. Binary indicators are imputed when necessary but retain their 0/1 interpretation. A supervised feature selector is then fitted using the training target.

The exact scaler or selection criterion is not the main lesson in this block. The main lesson is the **boundary of fitting**:

```text
preparation.fit(X_train, y_train)

preparation.transform(X_train)
preparation.transform(X_validation)
preparation.transform(X_test)
```

There is no `.fit()` call on validation or test data.

## 8. Mixed data require explicit column roles

A mixed tabular dataset should not be transformed by blindly applying the same operation to every column.

The notebooks therefore maintain explicit lists of:

- numerical predictors;
- binary categorical predictors;
- engineered numerical predictors;
- identifier;
- target.

For these synthetic datasets, categorical predictors are binary indicators. A dataset containing nominal string categories would normally add an encoding branch, for example `OneHotEncoder(handle_unknown="ignore")`, fitted on training data only.

---

# Part VI. Feature selection in an end-to-end workflow

## 9. Selection must obey the same fitting boundary

Feature selection uses the target and is therefore supervised. It cannot be performed on the full dataset before splitting.

The notebooks use one transparent filter-based selector to demonstrate the integration. The selected set is learned from TRAIN and then frozen for VALIDATION, TEST, and later external observations.

The number of retained variables is a didactic choice in these notebooks, not a claim that one fixed proportion is optimal. Later model experiments can compare:

```text
all valid predictors
versus
selected predictors
```

under the same evaluation design.

Most importantly, if model development later uses cross-validation, supervised feature selection must move **inside the cross-validation pipeline**.

---

# Part VII. What should be exported

## 10. Data alone are not enough

Exporting only a transformed CSV is insufficient for reproducibility. A model-ready table does not record how a future observation should be transformed.

The notebooks therefore create several complementary artifacts.

### 10.1 Model-ready split tables

The TRAIN, VALIDATION, and TEST matrices are exported with identifiers and targets. These are convenient for simple fixed-split model demonstrations.

### 10.2 Split assignments

A split manifest records which identifier belongs to which partition. This makes the partition an explicit experimental artifact rather than an implicit consequence of a random seed.

### 10.3 An unfitted preparation template

An unfitted scikit-learn preparation object is exported for future workflows that need to refit preparation, particularly cross-validation.

### 10.4 A fitted preparation object

A fitted preparation object is exported for transforming validation, test, or genuinely new observations under the fixed training-derived preparation.

### 10.5 Target encoder when necessary

The multiclass task exports the fitted mapping between human-readable classes and numerical target codes.

### 10.6 Preparation manifest

The manifest records the important decisions, including:

- dataset and task type;
- identifier and target;
- split strategy;
- random seed;
- feature roles;
- engineered features;
- selected features;
- output dimensions;
- artifact locations;
- software versions;
- intended use and cross-validation warning.

This is part of the reproducibility record.

---

# Part VIII. Applying the preparation to external data

## 11. New data must follow the same contract

A future dataset should not trigger a new preparation strategy simply because it arrives later.

The expected path is:

```text
new raw observations
       ↓
validate the expected raw schema
       ↓
apply the same deterministic cleaning rules
       ↓
apply the same engineered-feature recipe
       ↓
load the fitted preparation object
       ↓
transform only
       ↓
model-ready matrix
```

The fitted preparation object must **not** be refitted on the external batch. Re-fitting would change the feature representation and would make the future data inconsistent with the data used to train the model.

The notebooks demonstrate this by saving the fitted preparation object, reloading it from disk, and transforming a small held-out batch with the target removed.

---

# Part IX. Training, validation, and testing after preparation

## 12. What TRAIN, VALIDATION, and TEST will do next

Once the preparation workflow has been frozen for a fixed split, the next modelling stage can use the partitions as follows:

```text
TRAIN
  fit candidate models

VALIDATION
  compare model/hyperparameter decisions

TEST
  evaluate the final frozen choice once
```

The test set should not repeatedly influence choices. If the test score is inspected after every modelling change, the test set gradually becomes another validation set.

A clean fixed-split teaching workflow therefore has three boundaries:

```text
preparation decisions  -> learned from TRAIN
model decisions        -> informed by TRAIN + VALIDATION
final performance      -> estimated on TEST
```

For more rigorous benchmarking, the same principles can be implemented with cross-validation or nested cross-validation.

---

# Part X. Portability and artifact lifetime

## 13. Serialized objects are versioned computational artifacts

A `joblib` or pickle artifact should be treated as a computational object, not as a universal archival format. Loading generally assumes a compatible Python/scikit-learn environment.

For that reason, the preparation manifest records relevant package versions. The environment specification for the project should be retained alongside serialized preparation objects.

For long-term reproducibility, keep both:

```text
human-readable decisions and schema
+
serialized fitted object
```

The serialized object enables immediate reuse; the manifest and source notebook explain how to reconstruct it.

---

# Part XI. Decisions used in the three demonstrations

## 14. Task 1: Alzheimer binary classification

The demonstration uses:

- raw schema validation and deterministic cleaning;
- predefined clinical/anthropometric engineered features;
- stratified 70/15/15 splitting;
- median imputation and standardization for numerical variables;
- mode imputation for binary indicators;
- supervised training-only feature selection;
- the original binary target without a target encoder.

## 15. Task 2: Sleep-profile multiclass classification

The demonstration uses:

- raw schema validation and deterministic cleaning;
- predefined sleep/circadian engineered features;
- stratified 70/15/15 splitting;
- the same separation between numerical and binary predictor preparation;
- supervised multiclass feature selection learned on TRAIN;
- a separately serialized target encoder for the four sleep-profile classes.

## 16. Task 3: Crop-yield regression

The demonstration uses:

- raw schema validation and deterministic cleaning;
- predefined water, nutrient, nonlinear, and interaction features;
- random 70/15/15 splitting;
- numerical imputation and standardization;
- mode imputation for binary management indicators;
- supervised regression feature selection learned on TRAIN;
- the yield target retained in `t/ha`.

Target scaling is intentionally excluded from the default workflow so that the later regression demonstrations retain an interpretable response variable.

---

# Part XII. Final checklist

## 17. Before calling a dataset model-ready

A preparation workflow should be able to answer the following questions clearly:

- What is the raw input schema?
- Which column is the identifier?
- Which column is the target?
- Which transformations are deterministic?
- Which transformations estimate parameters from data?
- Was the split created before those fitted transformations?
- Which operations were fitted only on TRAIN?
- Are VALIDATION and TEST transformed without refitting?
- Which features reach the model?
- Can the fitted transformation be loaded later?
- Can a new target-free batch be transformed using the same artifact?
- Is the target encoding or target transformation saved when required?
- Are split assignments recorded?
- Are software versions and major decisions recorded?
- Is there a clear warning about refitting preparation inside cross-validation?

If these questions have explicit answers, the transition from dataset exploration to model development is reproducible and auditable.

---

## 18. Main message

The objective of an end-to-end preparation workflow is not to create the most complicated pipeline possible. It is to ensure that **the exact same preparation logic can be learned from the correct data, frozen, audited, and applied consistently later**.

A useful mental model is:

```text
raw data
   ↓
deterministic contract
   ↓
split
   ↓
fit preparation on TRAIN
   ↓
freeze preparation
   ↓
transform TRAIN / VALIDATION / TEST
   ↓
export model-ready data + reusable artifacts
   ↓
model training and evaluation
```

That frozen boundary is what makes the next machine-learning stage meaningful.
