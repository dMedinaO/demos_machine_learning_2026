# Machine Learning 2026 — Educational Demos

This repository contains the practical material for an introductory-to-intermediate machine-learning workflow focused on **reproducible supervised learning with tabular data**.

The material is organized as a sequence of small, explicit demonstrations. Each notebook focuses on one stage of the workflow, while the later notebooks integrate those stages into a complete preparation, model-selection, optimization, and final-evaluation process.

The current tabular module covers three complementary supervised-learning settings:

| Task | Dataset | Problem type | Target |
|---|---|---|---|
| `task1_alzheimer` | Synthetic Alzheimer cohort | Binary classification | `alzheimer` |
| `task2_sleep_profiles` | Synthetic sleep profiles | Multiclass classification | `sleep_profile` |
| `task3_crop_yield` | Synthetic crop-yield study | Regression | `yield_t_ha` |

The datasets are synthetic and designed for teaching. They contain realistic combinations of continuous and categorical variables, nonlinear relationships, interactions, class-specific patterns, and enough structure to demonstrate common machine-learning decisions without relying on sensitive real-world data.

---

## Repository structure

```text
.
├── data/
│   ├── synthetic_alzheimer/
│   │   ├── raw_data/
│   │   └── processed_data/
│   ├── synthetic_sleep_profiles/
│   │   ├── raw_data/
│   │   └── processed_data/
│   └── synthetic_crop_yield/
│       ├── raw_data/
│       └── processed_data/
│
├── docs/
│   ├── preliminary_steps.md
│   ├── tabular_data_preprocessing.md
│   ├── exploratory_data_analysis_categorical_labels.md
│   ├── tabular_feature_engineering_and_selection.md
│   ├── tabular_standardization_and_scaling.md
│   ├── tabular_dataset_splitting.md
│   ├── tabular_end_to_end_ml_preparation.md
│   ├── classical_supervised_learning_algorithms_and_metrics.md
│   ├── hyperparameter_optimization_and_model_selection.md
│   └── predictive_modeling_challenges_and_validation.md
│
├── notebooks/
│   └── tabular/
│       ├── task1_alzheimer/
│       ├── task2_sleep_profiles/
│       └── task3_crop_yield/
│
├── results/
├── scripts/
└── LICENSE
```

---

# 1. Current tabular-learning module

Each task follows the same general sequence, but the statistical treatment changes according to the target type.

```text
01  preprocessing
02  exploratory data analysis
03  feature engineering and feature selection
04  standardization and scaling
05  dataset partitioning
06  end-to-end ML preparation
07  model screening and selection
08  hyperparameter optimization with Optuna
09  final TEST evaluation
```

The notebooks are intentionally separated by topic before being integrated. The goal is to understand each decision independently before using a complete machine-learning workflow.

---

## 1.1 Task 1 — Alzheimer binary classification

Path:

```text
notebooks/tabular/task1_alzheimer/
```

Dataset:

```text
data/synthetic_alzheimer/
```

Raw dataset dimensions:

```text
1000 observations × 31 columns
```

Target:

```text
alzheimer
```

This task introduces the binary-classification workflow and includes class-conditioned EDA, binary feature comparisons, classification metrics, probability-based evaluation, model screening, Optuna optimization, and final protected TEST evaluation.

---

## 1.2 Task 2 — Sleep-profile multiclass classification

Path:

```text
notebooks/tabular/task2_sleep_profiles/
```

Dataset:

```text
data/synthetic_sleep_profiles/
```

Raw dataset dimensions:

```text
1200 observations × 23 columns
```

Target:

```text
sleep_profile
```

The target contains four classes:

```text
healthy_sleep
insomnia
sleep_apnea
circadian_disruption
```

This task extends the classification workflow to multiclass problems. It emphasizes class-specific statistics, macro-averaged metrics, confusion patterns between classes, multiclass feature relevance, and model selection under a multiclass objective.

---

## 1.3 Task 3 — Crop-yield regression

Path:

```text
notebooks/tabular/task3_crop_yield/
```

Dataset:

```text
data/synthetic_crop_yield/
```

Raw dataset dimensions:

```text
1200 observations × 21 columns
```

Target:

```text
yield_t_ha
```

This task focuses on continuous-target prediction. It includes correlation and nonlinear target relationships, regression-specific feature engineering, target-distribution analysis during splitting, regression metrics, residual diagnostics, and final prediction-error analysis in the original target units.

---

# 2. Notebook sequence

## `01_tabular_data_preprocessing.ipynb`

Introduces the mechanical preparation of tabular data before analysis.

Main topics include:

- reading tabular files and metadata;
- checking schema consistency;
- identifying numerical, categorical, identifier, and target columns;
- type conversion;
- missing-value inspection;
- duplicate detection;
- basic validity checks;
- deterministic column-name normalization;
- export of processed data and preprocessing reports.

This stage deliberately avoids exploratory statistical interpretation.

---

## `02_exploratory_data_analysis_*.ipynb`

Performs exploratory data analysis appropriate to the task type.

The classification notebooks emphasize:

- class composition;
- numerical distributions by label;
- categorical prevalence by label;
- outlier diagnostics;
- statistical tests and effect sizes;
- multiple-testing correction;
- correlation structure;
- PCA and nonlinear projections.

The regression notebook instead focuses on:

- target distribution;
- numerical and categorical associations with a continuous target;
- Pearson, Spearman, and Kendall relationships;
- nonlinear and quantile-based relationships;
- residual-style exploratory views;
- PCA and nonlinear projections colored by the continuous target.

---

## `03_feature_engineering_and_selection.ipynb`

Demonstrates how the original feature space can be expanded and reduced.

Examples include:

- deterministic derived variables;
- ratios and composite measures;
- interaction terms;
- nonlinear terms;
- filter-based selection;
- mutual information;
- regularized embedded selection;
- consensus between multiple feature-selection criteria.

Supervised feature selection is performed using training information only.

---

## `04_standardization_and_scaling.ipynb`

Demonstrates common scaling strategies without yet constructing a complete production pipeline.

Methods include:

- `StandardScaler`;
- `MinMaxScaler`;
- `RobustScaler`;
- `MaxAbsScaler`;
- Yeo–Johnson transformation;
- quantile transformation.

The notebooks emphasize that scaling is fitted on TRAIN and applied unchanged to later data.

For regression, predictor scaling and optional target scaling are treated as separate decisions.

---

## `05_dataset_partitioning.ipynb`

Introduces dataset splitting as an experimental-design decision rather than only a software command.

Topics include:

- TRAIN/TEST versus TRAIN/VALIDATION/TEST;
- random splitting;
- stratified splitting for classification;
- regression splitting;
- optional quantile-stratified regression examples;
- overlap checks;
- split manifests;
- class or target distribution comparisons;
- standardized mean differences;
- categorical prevalence differences;
- PCA-based inspection of partitioned spaces.

The associated documentation also introduces more advanced strategies such as group-aware, cluster-aware, temporal, spatial, distance-aware, and homology-aware partitioning.

---

## `06_end_to_end_ml_preparation.ipynb`

Integrates the previous preparation decisions into reusable scikit-learn workflows.

This is the first notebook where the project explicitly constructs reusable preparation objects rather than isolated demonstrations.

The workflow separates:

```text
deterministic preparation
```

from operations that must learn parameters from TRAIN:

```text
imputation
scaling
feature selection
```

The notebooks export:

- ML-ready TRAIN/VALIDATION/TEST data;
- fitted preparation pipelines;
- unfitted pipeline templates for later cross-validation;
- selected feature names;
- preparation manifests;
- target encoders when required.

A reload test confirms that a serialized fitted preparation object can transform new observations without access to their target.

---

## `07_model_screening_and_selection.ipynb`

Performs a small classical-model benchmark before formal hyperparameter optimization.

The purpose is **algorithm-family screening**, not exhaustive tuning.

Several reasonable configurations are tested for model families such as:

- linear or logistic models;
- K-nearest neighbors;
- support-vector methods;
- Random Forest;
- Extra Trees;
- gradient-boosting methods.

Only TRAIN and VALIDATION are used.

The notebooks analyze:

- primary and secondary metrics;
- TRAIN–VALIDATION gaps;
- family-level performance;
- confusion matrices or residuals;
- ROC/precision-recall behavior where appropriate.

The best three **algorithm families** are exported for formal optimization.

---

## `08_hyperparameter_optimization_optuna.ipynb`

Optimizes the three selected model families using **Optuna**.

The optimization uses cross-validation on the development data and keeps the protected TEST set outside the process.

A preparation-pipeline template is fitted independently inside each cross-validation training fold so that:

```text
preprocessing
feature selection
scaling
model fitting
```

remain leakage-safe during optimization.

After optimization:

```text
TRAIN + VALIDATION
```

are combined and the selected preparation + estimator workflow is refitted on the complete development set.

The notebooks export:

- Optuna studies;
- trial histories;
- best hyperparameters;
- fold-level scores;
- parameter-importance summaries;
- the final fitted optimized model.

---

## `09_final_test_evaluation.ipynb`

Performs the final protected evaluation.

The notebook does **not** fit a model and does **not** change any modeling decision.

It:

1. loads the optimized frozen model;
2. opens TEST;
3. generates predictions;
4. calculates final metrics;
5. produces final diagnostic figures;
6. exports predictions and a final model summary.

For classification this includes confusion matrices, discrimination metrics, probability diagnostics, and class-level results.

For regression this includes MAE, RMSE, $R^2$, residual diagnostics, and error analysis across the target range.

The methodological rule is simple:

> If the final TEST result is disappointing, TEST must not be used to retune the model. The next scientific step is new development data, a revised predefined workflow, or an independent evaluation scenario.

---

# 3. Documentation

The `docs/` directory provides the conceptual background used by the notebooks.

| Document | Main purpose |
|---|---|
| `preliminary_steps.md` | Python environments, package installation, Jupyter kernels, environment export and reproducibility basics |
| `tabular_data_preprocessing.md` | Structural cleaning and preparation of tabular datasets before analysis |
| `exploratory_data_analysis_categorical_labels.md` | EDA principles, visualizations, statistical tests, effect sizes, PCA and nonlinear projections for categorical labels |
| `tabular_feature_engineering_and_selection.md` | Feature construction, reduction, filter/wrapper/embedded methods, mixed data and leakage-aware feature selection |
| `tabular_standardization_and_scaling.md` | Standardization, scaling, transformations and their appropriate use |
| `tabular_dataset_splitting.md` | TRAIN/VALIDATION/TEST design, random/stratified/regression splitting and advanced partitioning concepts |
| `tabular_end_to_end_ml_preparation.md` | Design decisions for reusable, leakage-safe preparation workflows |
| `classical_supervised_learning_algorithms_and_metrics.md` | Classical classification/regression algorithms, binary/multiclass/regression metrics and interpretation |
| `hyperparameter_optimization_and_model_selection.md` | Hyperparameter search, Optuna concepts, CV integration and model-selection decisions |
| `predictive_modeling_challenges_and_validation.md` | Overfitting, dimensionality, leakage, small data, shift, calibration, validation and generalization |

The documentation is intended to be read alongside the notebooks rather than as an independent textbook.

---

# 4. Data and generated artifacts

Each synthetic task uses the same basic data organization:

```text
data/<dataset>/
├── raw_data/
└── processed_data/
```

Later notebooks can additionally generate task-specific artifacts such as:

```text
feature_data/
ml_ready/
```

The `results/` directory is used for outputs generated during demonstrations, including:

- figures;
- statistical tables;
- split assignments;
- manifests;
- serialized preparation objects;
- Optuna studies;
- fitted final models;
- TEST predictions;
- final reports.

The exact artifact set depends on which notebooks have been executed locally.

---

# 5. How to use the material

A recommended first pass is to work sequentially within one task.

For example:

```text
notebooks/tabular/task1_alzheimer/
```

and execute:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

Then repeat the same conceptual stages for the multiclass and regression tasks.

The repetition is intentional. The workflow structure remains recognizable while the statistical decisions change with the target type.

Before running the notebooks, see:

```text
docs/preliminary_steps.md
```

for environment and package-management guidance.

---

# 6. Core methodological principles

The repository follows a few rules throughout the current material.

### Keep TEST protected

TEST is not used for:

- feature selection;
- preprocessing decisions;
- model-family screening;
- hyperparameter optimization;
- threshold selection.

### Fit learned preprocessing only from training data

Operations such as:

```text
imputation
scaling
feature selection
PCA
```

must learn their parameters from the appropriate training subset.

### Use pipelines when the complete workflow is being validated

The early notebooks intentionally demonstrate individual operations directly.

Once cross-validation and optimization are involved, fitted preprocessing and model training are combined so that every fold remains independent.

### Compare model families under the same evaluation design

A better score is meaningful only when candidate models were evaluated using compatible data partitions, metrics, and preparation rules.

### Separate model development from final evaluation

The current flow is deliberately staged as:

```text
screen
→ optimize
→ freeze
→ TEST
```

rather than repeatedly modifying a model after observing TEST results.

---

# 7. Current status

The first tabular supervised-learning module is now implemented end to end for:

- binary classification;
- multiclass classification;
- regression.

The current material covers the complete fixed-partition workflow from raw data through final TEST evaluation.

```text
raw data
   ↓
preprocessing
   ↓
EDA
   ↓
feature engineering / selection
   ↓
scaling
   ↓
partitioning
   ↓
reusable ML preparation
   ↓
model-family screening
   ↓
Optuna HPO
   ↓
final frozen model
   ↓
TEST evaluation
```

---

# 8. Next modules

The repository will continue beyond the current fixed-partition tabular workflow.

## 8.1 Cross-validation and repeated validation

The next block will treat cross-validation as a topic in its own right rather than only as an internal component of hyperparameter optimization.

Planned demonstrations include:

- K-fold cross-validation;
- stratified K-fold classification;
- repeated cross-validation;
- fold-level performance distributions;
- stability across partitions;
- model comparison under repeated resampling;
- integration of preprocessing inside CV;
- nested validation concepts;
- comparison between fixed holdout and repeated-resampling estimates.

Later extensions can introduce domain-aware schemes such as:

```text
group-aware
cluster-aware
distance-aware
homology-aware
temporal
spatial
```

when the data structure requires them.

---

## 8.2 Embeddings and numerical representations

A subsequent module will move from ordinary tabular descriptors toward learned or high-dimensional numerical representations.

Planned topics include:

- embedding matrices as machine-learning inputs;
- dimensionality and representation geometry;
- preprocessing of embedding spaces;
- PCA, UMAP and t-SNE for representation inspection;
- distance and similarity in embedding space;
- feature reduction for high-dimensional representations;
- classical machine learning on embeddings;
- comparison between engineered descriptors and learned representations.

This will provide a bridge from conventional tabular machine learning to representation-based workflows used in areas such as sequence, text, image, and biomolecular machine learning.

---

## 8.3 Later extensions

Additional modules can progressively introduce:

- stronger cross-validation designs;
- uncertainty and calibration;
- external validation;
- domain shift;
- ensemble strategies;
- interpretability;
- representation benchmarking;
- domain-specific biological examples.

These extensions will build on the same core principle used throughout the current repository: **data preparation, model development, model selection, and final evaluation should remain explicit and reproducible.**

---

# 9. Scope

The repository is educational. The synthetic examples are designed to illustrate machine-learning methodology and should not be interpreted as clinical, biological, or agronomic predictive systems.

The emphasis is on understanding **how a predictive workflow is constructed and evaluated**, not on maximizing benchmark scores.

---

## License

See [`LICENSE`](LICENSE).
