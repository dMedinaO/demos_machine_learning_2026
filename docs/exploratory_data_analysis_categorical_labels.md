# Exploratory Data Analysis for Tabular Data with Categorical Labels

## 1. Purpose of this document

Exploratory Data Analysis, usually abbreviated as **EDA**, is the systematic process used to understand a dataset before fitting predictive models or drawing formal conclusions from it.

EDA is not simply the production of plots. A useful exploratory analysis should answer a sequence of questions about the data:

- What observations and variables are available?
- What type of information does each variable contain?
- How are individual variables distributed?
- Are there unusual observations or extreme values?
- Which variables appear related to one another?
- Are some variables redundant?
- Do groups defined by categorical variables differ from one another?
- Does the dataset contain visible substructure?
- How is the dataset organized in multivariate space?
- Are apparent patterns large enough to be practically meaningful, or are they only statistically detectable?

The purpose of EDA is therefore to build a **structured understanding of the dataset** that can guide later decisions about modelling, feature engineering, validation, interpretation, and experimental design.

This document focuses on **tabular datasets whose target or label is categorical**, such as:

```text
control / disease
0 / 1
negative / positive
class_A / class_B / class_C
```

A separate document and notebook will later address exploratory analysis when the response variable is **numerical**, where the statistical questions and recommended methods are different.

The practical example associated with this document will be developed in:

```text
notebooks/tabular/task1_alzheimer/
```

and will use the processed tabular dataset generated in the preprocessing notebook.

---

## 2. Where EDA begins in the workflow

A clean workflow separates data preprocessing from exploratory analysis.

A typical sequence is:

```text
raw data
   ↓
preprocessing
   ↓
processed data
   ↓
exploratory data analysis
   ↓
data splitting and modelling
```

The preprocessing stage verifies that the table can be used reliably. It addresses issues such as malformed values, duplicated rows, incorrect data types, inconsistent column names, invalid categories, and other structural problems.

EDA begins **after the basic structure of the dataset has been validated**.

This distinction is important. During EDA, we want to study the properties of the data rather than continually modify the dataset while looking at it.

Some exploratory findings may reveal that preprocessing decisions should be revisited. For example, EDA may reveal a biologically impossible extreme value, a hidden batch effect, or a categorical level that was not previously recognized. If this occurs, the preprocessing pipeline should be updated explicitly and rerun rather than silently altering the data inside the EDA notebook.

---

# Part I. Dataset-level summary

## 3. Start with a global summary of the dataset

Before analyzing individual variables, the dataset should be described as a whole.

At minimum, record:

- number of observations;
- number of variables;
- identifier columns;
- numerical variables;
- categorical variables;
- target or label column;
- number of classes in the label;
- number of observations per class;
- remaining missing values, if any;
- duplicated observations or identifiers, if any remain intentionally;
- constant or near-constant variables;
- source or provenance information when available.

This information creates a compact **dataset profile**.

For example:

```text
Observations: 1000
Variables: 31
Identifier columns: 1
Numerical variables: 18
Categorical variables: 11
Target: diagnosis
Classes: control, disease
```

This is not yet a scientific result. It is the structural description needed to interpret everything that follows.

---

## 4. Classify variables by analytical role

A column's storage type is not necessarily the same as its analytical role.

For example, a column containing values `0` and `1` may be stored as an integer but conceptually represent a categorical variable.

For EDA it is useful to distinguish at least the following roles:

| Role | Example | Typical use in EDA |
|---|---|---|
| Identifier | `patient_id` | Tracking only; normally excluded from statistics |
| Numerical continuous | `age`, `bmi`, `cholesterol` | Distribution, correlations, group comparisons |
| Numerical discrete | `visits`, `score` | Distribution and group comparisons |
| Binary categorical | `smoker`, `diabetes` | Counts, proportions, contingency analysis |
| Multiclass categorical | `education`, `region` | Counts, proportions, contingency analysis |
| Target label | `diagnosis` | Defines groups for supervised comparisons |

This classification should be explicit because it determines which summaries, plots, and statistical tests are appropriate.

Identifiers should generally not be treated as predictive measurements simply because they are numerical.

---

# Part II. Univariate analysis

## 5. What is univariate analysis?

**Univariate analysis** studies one variable at a time.

The goal is to understand the distribution, range, frequency structure, and unusual values of each feature independently of the others.

This is the natural first analytical layer of an EDA.

---

## 6. Numerical variables

For every numerical variable, calculate a compact descriptive summary.

Useful statistics include:

- number of valid observations;
- number and percentage of missing observations;
- mean;
- standard deviation;
- median;
- first quartile;
- third quartile;
- interquartile range;
- minimum;
- maximum;
- selected percentiles when useful;
- number of unique values.

The **mean and standard deviation** are informative when the distribution is reasonably symmetric, while the **median and interquartile range** are generally more robust to skewness and extreme observations.

A single statistic should not be used to describe every variable automatically.

### 6.1 Recommended visualizations for numerical variables

Useful visualizations include:

| Plot | Main question |
|---|---|
| Histogram | What is the overall distribution shape? |
| KDE / density plot | Is the distribution unimodal, skewed, or multimodal? |
| Boxplot | Where are the median, quartiles, and potential extreme observations? |
| Violin plot | What is the distribution shape together with group summaries? |
| ECDF | What fraction of observations lie below a given value? |

Histograms are usually an excellent first choice because they preserve the intuitive concept of counts. Density curves can complement them, but the appearance of a KDE depends strongly on bandwidth selection and should not be interpreted as if it were the raw data itself.

An **ECDF**, or empirical cumulative distribution function, is often underused but very useful because it does not require choosing histogram bins or a kernel bandwidth.

---

## 7. Shape of a numerical distribution

When inspecting a distribution, consider:

### Symmetry

Is the distribution approximately symmetric around its center?

### Skewness

Does the distribution contain a long tail toward low or high values?

### Multimodality

Are there two or more apparent peaks?

Multimodality can indicate the presence of subpopulations, experimental batches, demographic groups, hidden categories, or mixtures of processes.

### Heavy tails

Are extreme observations more frequent than expected under a Gaussian distribution?

### Truncation or censoring

Does the variable have an artificial lower or upper boundary caused by measurement or acquisition rules?

### Discretization

Does a supposedly continuous variable contain only a small number of repeated values?

This can occur when measurements have been rounded, binned, or collected using an ordinal scale.

---

## 8. Categorical variables

For a categorical variable, the main univariate summaries are:

- number of categories;
- count per category;
- proportion per category;
- missing category frequency;
- rare categories;
- dominant categories.

A frequency table is usually the first summary to inspect.

### 8.1 Recommended plots for categorical variables

The preferred visualization is usually a **bar plot**.

The y-axis can represent either:

```text
count
```

or

```text
proportion
```

Proportions are particularly useful when comparing categorical distributions across datasets or groups with different sample sizes.

Pie charts are generally less informative for scientific EDA because humans compare lengths more accurately than angles or areas. Bar plots usually communicate the same information more clearly.

---

# Part III. Missingness as an exploratory property

## 9. Missing values during EDA

Missing values should normally already have been identified during preprocessing, but their **distribution and structure** may still be scientifically informative.

EDA should distinguish between:

```text
How much information is missing?
```

and

```text
Is the missingness associated with particular variables or groups?
```

Useful summaries include missing counts and percentages by column.

For datasets with substantial missingness, a missingness matrix or heatmap can help determine whether several variables tend to be missing together.

Missingness should not automatically be interpreted as random. It may reflect acquisition procedures, patient subgroups, experimental failures, measurement limits, or study design.

Imputation belongs to the modelling/preprocessing strategy and must be handled carefully to avoid information leakage. EDA may characterize missingness without prematurely applying an imputation procedure to the full dataset.

---

# Part IV. Outliers and unusual observations

## 10. What is an outlier?

An **outlier** is an observation that appears unusually distant from the majority of observations according to some definition.

An outlier is not automatically an error.

It may represent:

- a data-entry mistake;
- a measurement error;
- an unusual but valid individual;
- a rare biological state;
- a different experimental subgroup;
- a different acquisition process;
- a genuinely important extreme case.

For this reason, outlier detection and outlier removal are different tasks.

**Never remove observations solely because an automated rule labels them as outliers.**

---

## 11. Common univariate outlier rules

### 11.1 Interquartile range rule

A traditional rule identifies observations outside:

```text
Q1 - 1.5 × IQR
Q3 + 1.5 × IQR
```

where:

```text
IQR = Q3 - Q1
```

This rule is simple and robust, but it is not a universal definition of an erroneous observation.

Highly skewed but valid distributions can naturally produce many observations outside these limits.

### 11.2 Standard z-score

A z-score is calculated as:

```text
z = (x - mean) / standard_deviation
```

Large absolute z-scores are sometimes used to flag extreme observations.

However, both the mean and standard deviation are sensitive to extreme values, making this method less robust for strongly skewed distributions.

### 11.3 Robust z-score using the median and MAD

A more robust alternative is based on the **median absolute deviation**, or MAD.

The MAD is defined as:

```text
MAD = median(|x - median(x)|)
```

Robust methods are often more appropriate when the data contain skewness or extreme observations.

---

## 12. Multivariate outliers

An observation may look normal when each variable is inspected independently but become unusual when several variables are considered jointly.

For example, a patient may have individually plausible values for age, weight, and blood pressure, but their particular combination may be highly unusual.

Potential approaches include:

- distance in standardized feature space;
- Mahalanobis distance when its assumptions are reasonable;
- PCA score-space inspection;
- dedicated anomaly-detection methods in more advanced settings.

These approaches should be used as **diagnostic tools**, not automatic deletion mechanisms.

---

# Part V. Bivariate analysis

## 13. What is bivariate analysis?

**Bivariate analysis** studies the relationship between two variables.

The appropriate method depends on the types of the two variables.

Three common combinations occur in tabular datasets:

```text
numerical ↔ numerical
numerical ↔ categorical
categorical ↔ categorical
```

Each combination should be analyzed differently.

---

## 14. Numerical versus numerical variables

The first visualization should usually be a **scatter plot**.

A scatter plot can reveal:

- positive or negative association;
- linear or nonlinear relationships;
- heteroscedasticity;
- clusters;
- saturation effects;
- unusual observations;
- gaps in the sampled space.

For very large datasets, point overlap can hide structure. Useful alternatives include transparency, hexbin plots, two-dimensional density plots, or subsampling for visualization.

---

## 15. Correlation

Correlation summarizes the strength and direction of association between numerical variables.

Correlation is not equivalent to causation, and a low correlation does not imply absence of all forms of dependence.

### 15.1 Pearson correlation

Pearson correlation measures the strength of a **linear relationship** between two numerical variables.

It is most informative when the association is approximately linear and not dominated by strong outliers.

### 15.2 Spearman correlation

Spearman correlation is based on ranks and measures **monotonic association**.

It is useful when:

- the relationship is monotonic but not linear;
- distributions are skewed;
- extreme values make Pearson correlation unstable;
- variables are ordinal or effectively rank-based.

### 15.3 Kendall correlation

Kendall's tau is another rank-based measure of monotonic association. It can be useful with small datasets, many ranking questions, or when a rank-based interpretation is desirable.

### 15.4 Correlation matrices

A correlation matrix allows many numerical features to be inspected simultaneously.

A heatmap is useful for identifying:

- highly related features;
- potential redundant measurements;
- groups of correlated variables;
- unexpected relationships.

However, correlation heatmaps should not replace inspection of important pairwise scatter plots. Two variables can have a similar correlation coefficient while displaying very different relationships.

---

## 16. Numerical versus categorical variables

When a numerical variable is compared across categories, the objective is to understand whether its distribution differs among groups.

Useful plots include:

- boxplots;
- violin plots;
- strip plots;
- swarm plots for reasonably small datasets;
- overlapping histograms with care;
- ECDF curves by group;
- combined violin and point displays.

For a categorical target label, this becomes one of the central analyses in the EDA.

For example:

```text
BMI by diagnosis
age by diagnosis
cholesterol by diagnosis
```

The analysis should compare **complete distributions**, not only group means.

Two groups can have similar means while differing substantially in variance, skewness, tails, or multimodality.

---

## 17. Categorical versus categorical variables

Relationships between categorical variables are summarized using a **contingency table**.

For example:

```text
                diagnosis
smoker        control   disease
no               320       280
yes              180       220
```

Useful visualizations include:

- grouped bar plots;
- stacked bar plots;
- normalized stacked bar plots;
- mosaic plots in appropriate cases.

When comparing groups of different sizes, proportions are often more informative than raw counts.

---

# Part VI. Statistical tests in EDA

## 18. Why use statistical tests during EDA?

Visualizations reveal patterns, while statistical tests help quantify whether observed differences are compatible with a specified null hypothesis.

A statistical test should never be chosen only because it is familiar. The choice depends on:

- variable type;
- number of groups;
- independence or pairing of observations;
- distributional assumptions;
- variance structure;
- sample size;
- scientific question.

Statistical significance is not equivalent to practical importance.

A complete EDA should therefore combine:

```text
visualization + descriptive statistics + hypothesis test + effect size
```

when formal group comparison is useful.

---

## 19. Comparing a numerical variable between two categorical groups

Suppose the label has two categories:

```text
control
disease
```

and we want to compare a numerical feature between them.

### 19.1 Student's t-test

The classical independent-samples t-test compares group means under assumptions that include independent observations and, in its standard form, equal population variances.

It should not be applied automatically to every variable.

### 19.2 Welch's t-test

Welch's t-test compares group means without assuming equal variances.

For independent groups, Welch's test is often preferable to the equal-variance version unless there is a specific reason to assume equal variances.

### 19.3 Mann-Whitney U test

The Mann-Whitney U test is a rank-based alternative that does not require normality.

It is sometimes described as a test of medians, but that interpretation is only valid under additional assumptions about distribution shape. More generally, it tests whether observations from one group tend to have larger ranks than observations from the other group.

The choice between parametric and rank-based tests should reflect the scientific question and data structure rather than a mechanical normality-test decision tree.

---

## 20. Comparing a numerical variable across more than two groups

When the categorical variable has three or more groups, possible tests include:

### One-way ANOVA

Tests whether group means differ under the assumptions of the classical ANOVA model.

### Welch's ANOVA

Useful when group variances differ substantially.

### Kruskal-Wallis test

A rank-based alternative for comparing several independent groups.

A significant omnibus test indicates that at least one group differs, but it does not identify which groups differ.

If pairwise post-hoc comparisons are performed, they should use an appropriate multiple-comparison correction.

---

## 21. Comparing categorical variables

### 21.1 Chi-square test of independence

The chi-square test evaluates whether two categorical variables are statistically associated.

It is appropriate when expected contingency-table counts are sufficiently large.

### 21.2 Fisher's exact test

Fisher's exact test is useful for small contingency tables, especially when expected counts are low.

For a standard binary-by-binary comparison, it provides an exact test of association.

---

# Part VII. Effect sizes

## 22. Statistical significance is not enough

A p-value answers a limited question about compatibility with a null hypothesis. It does not measure the magnitude or scientific relevance of a difference.

With a sufficiently large sample, very small differences can become statistically significant.

Therefore, report an **effect size** whenever group differences are evaluated.

Useful examples include:

| Comparison | Possible effect size |
|---|---|
| Two numerical groups | Cohen's d, Hedges' g, Cliff's delta |
| Several numerical groups | eta squared, omega squared |
| Two categorical variables | Cramér's V |
| Binary 2 × 2 table | odds ratio, risk ratio when appropriate |
| Rank-based two-group comparison | rank-biserial correlation or Cliff's delta |

The effect size should be interpreted together with the original variable scale and the scientific context.

For skewed or non-Gaussian data, **Cliff's delta** can provide a useful distribution-free measure of how frequently observations from one group are larger than observations from another.

---

# Part VIII. Multiple testing

## 23. Why multiple testing matters

A tabular dataset may contain dozens or hundreds of variables.

If each variable is independently tested against the categorical label using a significance threshold of 0.05, some apparently significant results will occur by chance.

This is the **multiple-testing problem**.

If many related hypotheses are evaluated, p-values should be corrected.

Common approaches include:

### Bonferroni correction

Controls the family-wise error rate but can be very conservative.

### Benjamini-Hochberg false discovery rate

Controls the expected proportion of false discoveries among rejected hypotheses and is often useful in exploratory high-dimensional analyses.

For EDA with many variables, the Benjamini-Hochberg procedure is frequently more informative than applying dozens of uncorrected p-values.

The analysis should ideally retain both:

```text
raw p-value
adjusted p-value
```

rather than overwriting the original result.

---

# Part IX. Assumptions and diagnostic thinking

## 24. Avoid mechanical assumption testing

It is common to see workflows such as:

```text
Shapiro-Wilk test
        ↓
normal → t-test
not normal → Mann-Whitney
```

This is usually too simplistic.

Normality tests can become extremely sensitive with large sample sizes and underpowered with small samples. A tiny departure from Gaussianity may become statistically significant even when a mean-based method remains perfectly useful.

Instead, combine:

- distribution plots;
- Q-Q plots when relevant;
- sample size;
- robustness of the intended test;
- presence of extreme observations;
- equality or inequality of variances;
- scientific interpretation.

Statistical test selection should be based on the **analysis question and data-generating structure**, not a single preliminary p-value.

---

# Part X. Analysis with respect to a categorical label

## 25. The label changes the questions, not the raw data

When the dataset contains a categorical target, EDA should study how features behave across its classes.

For a binary classification problem, for example:

```text
label = 0
label = 1
```

we want to know whether variables show:

- shifts in central tendency;
- changes in dispersion;
- different distribution shapes;
- class-specific extreme observations;
- different category frequencies;
- different correlation structures;
- regions of multivariate separation or overlap.

Importantly, EDA should not be reduced to searching for variables with the smallest p-values.

The objective is to understand the **structure of the classification problem**.

---

## 26. Numerical features by categorical label

For each relevant numerical feature, compare distributions across classes using a consistent set of plots and statistics.

A good comparison may include:

```text
count per class
median and IQR per class
mean and standard deviation per class
boxplot or violin plot
appropriate hypothesis test
effect size
adjusted p-value when many features are tested
```

This gives both descriptive and inferential context.

Features should not be ranked solely by p-value. A feature with a tiny p-value but negligible effect size may be less useful than one with a larger, interpretable group difference.

---

## 27. Categorical features by categorical label

For each categorical feature, construct a contingency table against the label.

Inspect both raw counts and conditional proportions.

For example, rather than only asking:

```text
How many smokers belong to each class?
```

also ask:

```text
Within each diagnosis group, what proportion are smokers?
```

This distinction matters whenever class sizes differ.

The chi-square test, Fisher's exact test, Cramér's V, odds ratios, or related statistics may then be used depending on the table structure and question.

---

## 28. Class imbalance

The label distribution itself must be examined.

For every categorical target report:

- class counts;
- class proportions;
- number of classes;
- presence of rare classes.

Class imbalance does not automatically imply that the dataset must be artificially balanced.

It is first a property of the dataset that needs to be understood.

Decisions about resampling, weighting, or evaluation metrics belong to the modelling workflow and should not be silently applied during EDA.

---

# Part XI. Multivariable analysis

## 29. Why multivariable analysis is necessary

Univariate and bivariate analyses cannot fully describe a dataset in which many variables interact simultaneously.

A dataset may contain structure that is invisible when variables are studied independently.

Multivariable EDA attempts to understand the geometry of observations in a higher-dimensional feature space.

Typical goals include:

- identifying correlated groups of variables;
- detecting redundant dimensions;
- identifying broad sample clusters;
- visualizing overlap between categorical classes;
- detecting gradients or hidden subgroups;
- identifying multivariate extreme observations.

---

## 30. Feature scaling before multivariable analysis

Many multivariable methods depend on distances or variances.

Suppose one variable is measured from 0 to 1 and another from 0 to 100,000. Without scaling, the second variable can dominate the geometry even if it is not more informative.

A common transformation is standardization:

```text
z = (x - mean) / standard_deviation
```

which gives each feature approximately zero mean and unit variance.

Robust scaling using the median and IQR may be preferable when distributions contain strong outliers.

For exploratory visualization, transformations should always be explicitly stated.

Later, during machine-learning evaluation, any learned scaling operation must be fitted only on the training data to avoid leakage. The exploratory full-dataset visualization and the predictive pipeline are conceptually different operations and should not be confused.

---

# Part XII. Principal Component Analysis

## 31. What is PCA?

**Principal Component Analysis**, or PCA, is a linear dimensionality-reduction method.

Suppose a dataset contains many correlated numerical variables. PCA constructs new variables called **principal components**.

Each principal component is a linear combination of the original standardized features.

The first component captures the largest possible amount of variance in the data. The second component captures the largest remaining variance subject to being orthogonal to the first, and so on.

Conceptually:

```text
original feature space
        ↓
linear rotation
        ↓
principal component space
```

PCA does not use the categorical label to construct the components.

Therefore, PCA is an **unsupervised** transformation.

---

## 32. PCA scores and loadings

Two concepts are particularly important.

### Scores

The **scores** describe where each observation lies in principal-component space.

A scatter plot of:

```text
PC1 versus PC2
```

shows observations projected into the first two principal components.

### Loadings

The **loadings** describe how strongly each original feature contributes to each principal component.

Large positive or negative loadings indicate features that strongly influence the direction represented by that component.

Interpreting PCA therefore requires examining both observations and feature contributions.

---

## 33. Explained variance

Each principal component explains a proportion of the total variance.

Useful summaries include:

- variance explained by each component;
- cumulative explained variance;
- scree plot.

A scree plot helps determine how quickly information is concentrated into the first components.

There is no universal rule stating that PCA must preserve a specific percentage of variance for EDA. The appropriate number of components depends on the purpose of the analysis.

---

## 34. PCA colored by categorical label

Although PCA should be calculated **without using the label**, the resulting projection can be colored by the categorical target.

For example:

```text
PCA coordinates calculated from numerical features
        ↓
points colored by diagnosis
```

This allows us to inspect whether categorical groups occupy different regions of the major linear variation in the dataset.

However, visible overlap does not prove that classes are impossible to predict, and visible separation does not prove that a predictive model will generalize.

PCA is an exploratory visualization, not a validation procedure.

---

# Part XIII. Nonlinear dimensionality reduction

## 35. Why use nonlinear projections?

PCA can only represent linear combinations of features.

Complex datasets may contain curved manifolds, local neighborhoods, or nonlinear structures that are not visible in a linear projection.

Methods such as **t-SNE** and **UMAP** are commonly used for nonlinear visualization.

These methods should be treated as exploratory representations rather than definitive maps of the original data space.

---

## 36. t-SNE

**t-distributed Stochastic Neighbor Embedding**, or t-SNE, attempts to preserve local neighborhood relationships when projecting high-dimensional observations into two or three dimensions.

It is particularly useful for visualizing local clusters.

Important cautions include:

- distances between far-away clusters are not necessarily meaningful;
- cluster sizes can be visually distorted;
- results depend on parameters such as perplexity;
- different random seeds can produce different layouts;
- apparent islands in a t-SNE plot are not automatically biological or statistically distinct groups.

A fixed random seed should be used when reproducibility is important.

---

## 37. UMAP

**Uniform Manifold Approximation and Projection**, or UMAP, is another nonlinear dimensionality-reduction method frequently used for exploratory visualization.

Important parameters include:

### `n_neighbors`

Controls the balance between local and broader neighborhood structure.

Smaller values emphasize very local structure. Larger values incorporate broader neighborhoods.

### `min_dist`

Controls how tightly observations can be packed in the low-dimensional representation.

### Distance metric

The chosen metric determines how similarity is measured in the original feature space.

As with t-SNE, different reasonable parameter values can produce different visual structures.

UMAP should therefore be treated as a **view of the data under a specified representation**, not as ground truth.

---

## 38. Coloring nonlinear projections by categorical labels

A useful workflow is:

```text
numerical features
        ↓
standardization
        ↓
PCA / UMAP / t-SNE
        ↓
2D coordinates
        ↓
color points by categorical label
```

The label is used for visualization **after** the unsupervised projection has been calculated.

This helps answer questions such as:

- Do classes occupy distinct regions?
- Is one class broadly distributed while another is compact?
- Are there mixed neighborhoods?
- Are there possible subgroups within a class?
- Are extreme observations associated with a particular class?

These observations are exploratory hypotheses. They do not replace predictive evaluation.

---

# Part XIV. Pairwise and multifeature visualizations

## 39. Pair plots

A pair plot displays distributions and pairwise relationships among several variables.

It can be useful when the number of selected features is small.

For example, with four numerical variables it becomes possible to inspect all pairwise scatter plots while coloring observations by the categorical label.

Pair plots become difficult to interpret when too many variables are included. They should therefore be used selectively rather than generated automatically for an entire high-dimensional dataset.

---

## 40. Heatmaps

Heatmaps are useful for representing matrices such as:

- feature correlations;
- missingness patterns;
- standardized group summaries;
- contingency structures.

A heatmap is most useful when rows and columns have a clear meaning and the color scale represents a quantitatively interpretable value.

Avoid decorative heatmaps that encode too much information without clear labels or scales.

---

# Part XV. Redundancy and collinearity

## 41. Redundant variables

Highly correlated variables can contain overlapping information.

During EDA, identify strongly related variables and ask why the relationship exists.

Possible explanations include:

- one feature is mathematically derived from another;
- two variables measure the same biological process;
- one measurement is a proxy for another;
- acquisition procedures introduce duplicated information.

Correlation alone does not mean one variable must be removed.

The correct decision depends on modelling objectives, interpretability, domain knowledge, and the intended algorithm.

---

## 42. Variance Inflation Factor

The **Variance Inflation Factor**, or VIF, is sometimes used to evaluate multicollinearity in regression-oriented settings.

For a feature `j`:

```text
VIF_j = 1 / (1 - R_j²)
```

where `R_j²` is obtained by predicting that feature from the remaining predictors.

A high VIF indicates that a variable is strongly explained by other variables.

VIF can be informative when linear-model interpretability matters, but thresholds such as 5 or 10 should not be treated as universal laws.

---

# Part XVI. Distribution comparison across classes

## 43. Do not compare classes using only summary statistics

Consider two classes with identical means and standard deviations. Their distributions could still differ substantially in shape.

Therefore, class comparison should combine numerical summaries with graphical inspection.

Useful combinations include:

```text
boxplot + individual observations
violin plot + median/IQR
ECDF by class
histogram or density by class
```

Whenever possible, the visualization should show the observations or distribution directly rather than only an aggregate statistic.

---

# Part XVII. Relationships can differ by class

## 44. Class-conditional relationships

A correlation observed in the full dataset may not hold within individual classes.

For example:

```text
feature_A ↔ feature_B
```

may show a strong global correlation simply because the two classes occupy different regions of both variables.

Therefore, important relationships can be inspected:

```text
globally
and
within each label
```

This can reveal:

- class-specific associations;
- Simpson's paradox;
- different covariance structures;
- possible interaction effects.

The objective is not to test every possible pair exhaustively, but to investigate relationships suggested by domain knowledge or earlier EDA results.

---

# Part XVIII. Sample size and uncertainty

## 45. Always interpret patterns relative to sample size

Small groups produce unstable estimates.

A category represented by five observations should not be interpreted with the same confidence as a category containing five hundred observations.

Whenever distributions are compared across groups, report group sizes.

Confidence intervals or bootstrap intervals can be useful when uncertainty around an estimated statistic matters.

This is especially important for rare classes or sparse categorical levels.

---

# Part XIX. Recommended EDA sequence

## 46. A practical order of operations

A robust exploratory analysis of a tabular classification dataset can follow this sequence:

### Stage 1 — Load the processed dataset

Use the dataset exported from the preprocessing workflow. Do not silently modify the raw input during EDA.

### Stage 2 — Confirm analytical roles

Identify identifiers, numerical features, categorical features, and the categorical target.

### Stage 3 — Produce the dataset summary

Report dimensions, variable types, class counts, missingness, and other basic structural information.

### Stage 4 — Perform univariate analysis

Study numerical and categorical variables individually.

### Stage 5 — Inspect unusual observations

Identify potential outliers and determine whether they are plausible, erroneous, or simply extreme.

### Stage 6 — Perform bivariate analysis

Study numerical-numerical, numerical-categorical, and categorical-categorical relationships.

### Stage 7 — Analyze features with respect to the categorical label

Compare numerical distributions and categorical frequencies across classes.

### Stage 8 — Quantify differences

Use appropriate statistical tests, effect sizes, and multiple-testing corrections where relevant.

### Stage 9 — Study multivariable structure

Inspect correlations, redundancy, PCA, and selected nonlinear projections.

### Stage 10 — Interpret rather than merely display

Summarize which patterns are stable, which require further investigation, and which may influence later modelling choices.

---

# Part XX. What should be reported from an EDA?

## 47. Recommended outputs

A good EDA should leave behind reproducible artifacts rather than only notebook output.

Useful outputs may include:

```text
results/
└── exploratory_data_analysis/
    ├── dataset_summary.csv
    ├── numerical_summary.csv
    ├── categorical_summary.csv
    ├── missingness_summary.csv
    ├── correlation_matrix.csv
    ├── group_comparisons.csv
    ├── contingency_tests.csv
    └── figures/
```

The exact structure can vary depending on the project.

The important principle is that important numerical results should be exportable and reproducible rather than existing only as visually inspected notebook cells.

---

# Part XXI. Common mistakes in exploratory analysis

## 48. Mistake: performing EDA on unvalidated raw data

Structural errors can be mistaken for scientific patterns.

Perform basic preprocessing first.

## 49. Mistake: treating every integer as numerical

Binary and ordinal variables often require categorical interpretation.

## 50. Mistake: using only `describe()`

A table of descriptive statistics is only the beginning of EDA.

## 51. Mistake: using only visualizations

Plots should be complemented with numerical summaries and, where appropriate, statistical quantification.

## 52. Mistake: using only p-values

Always consider effect size, sample size, distribution shape, and scientific relevance.

## 53. Mistake: testing dozens of variables without correction

Multiple testing increases false discoveries.

## 54. Mistake: deleting all outliers

An extreme observation may be biologically or scientifically important.

## 55. Mistake: interpreting correlation as causation

Correlation describes association, not mechanism.

## 56. Mistake: interpreting UMAP or t-SNE as proof of clusters

Low-dimensional projections can distort the original geometry.

## 57. Mistake: allowing the label to determine an unsupervised projection

If the purpose is exploratory unsupervised visualization, PCA, UMAP, or t-SNE should be computed from the selected feature representation and the label should be added afterward for interpretation.

## 58. Mistake: using EDA performance as model validation

Class separation in a plot does not quantify predictive generalization.

Model performance must later be evaluated using held-out data and an appropriate validation design.

---

# Part XXII. EDA and data leakage

## 59. Exploratory analysis versus predictive workflow

EDA is often performed on the available dataset to understand its global structure. Predictive model development has stricter rules.

Operations that learn parameters from the data, such as:

```text
imputation
standardization
feature selection
supervised dimensionality reduction
```

must later be fitted correctly within the training data of the modelling workflow.

For example, an exploratory PCA can be useful for understanding the complete dataset. However, if PCA becomes part of a predictive model, the PCA transformation must be fitted only using the training partition and then applied to validation or test samples.

Understanding this distinction is essential for avoiding leakage.

---

# Part XXIII. Visualization principles

## 60. Scientific figures should answer questions

Every visualization should have a reason to exist.

Before generating a plot, ask:

```text
What question will this plot answer?
```

A useful plot should have:

- clearly named axes;
- units where applicable;
- a readable title or caption;
- a legend only when needed;
- category labels that can be understood;
- an appropriate scale;
- enough visual information to interpret sample size or distribution.

Avoid adding complexity that does not improve interpretation.

---

## 61. Recommended visualization choices

| Analytical question | Recommended visualization |
|---|---|
| Distribution of one numerical feature | Histogram, ECDF, boxplot |
| Distribution of one categorical feature | Bar plot |
| Numerical relationship | Scatter plot |
| Numerical feature across categories | Boxplot, violin plot, ECDF by group |
| Categorical association | Grouped or normalized bar plot |
| Many numerical associations | Correlation heatmap |
| Small selected feature set | Pair plot |
| Global linear geometry | PCA |
| Possible nonlinear local structure | UMAP or t-SNE |
| Outlier inspection | Boxplot, scatter plot, PCA score plot |

The purpose is not to use every visualization in every EDA. The correct set depends on the data and the questions being asked.

---

# Part XXIV. Reproducibility

## 62. An EDA should be reproducible

A scientific EDA should produce the same numerical results when rerun under the same conditions.

Good practices include:

- load data from defined project paths;
- avoid manual spreadsheet edits between notebook cells;
- document transformations;
- set random seeds for stochastic methods such as UMAP and t-SNE;
- export important tables;
- preserve the processed input dataset;
- record relevant parameters;
- avoid hidden notebook state;
- run the notebook from beginning to end before considering it complete.

The notebook should tell a coherent analytical story when executed sequentially from the first cell to the last.

---

# Part XXV. What EDA should not do yet

## 63. Keep exploratory analysis separate from model development

This EDA stage should not yet include:

```text
train/test splitting
model fitting
hyperparameter optimization
cross-validation
feature selection based on predictive performance
classification metrics
threshold optimization
model interpretation
```

Those operations belong to later stages of the machine-learning workflow.

The purpose of the current stage is to understand the dataset and the classification problem before attempting to solve it predictively.

---

# Part XXVI. Expected structure of the associated notebook

## 64. Planned practical example

The corresponding notebook in `notebooks/tabular/task1_alzheimer/` should implement the concepts from this document in a transparent sequence.

A suitable notebook structure is:

```text
1. Objectives and scope
2. Imports and reproducibility settings
3. Project paths
4. Load processed data
5. Define analytical roles
6. Dataset summary
7. Target class summary
8. Univariate numerical analysis
9. Univariate categorical analysis
10. Missingness review
11. Outlier diagnostics
12. Numerical-numerical relationships
13. Numerical features by categorical label
14. Categorical features by categorical label
15. Statistical tests
16. Effect sizes
17. Multiple-testing correction
18. Correlation and redundancy analysis
19. PCA
20. UMAP
21. t-SNE when educationally useful
22. Export tables and figures
23. Final EDA summary
```

The notebook should remain **exploratory and educational**. It should explain why each step is performed, not simply display a large number of plots.

---

# Final perspective

Exploratory Data Analysis is the stage in which a table becomes an understandable scientific object.

A strong EDA combines several complementary perspectives:

```text
structure
+
univariate distributions
+
bivariate relationships
+
group comparisons
+
statistical quantification
+
effect sizes
+
outlier diagnostics
+
multivariable geometry
+
careful visualization
+
domain interpretation
```

No single plot, correlation coefficient, p-value, or projection can summarize a complete dataset.

The objective is to combine these views into a coherent understanding of the data before making predictive modelling decisions.

For categorical-label datasets, particular attention should be paid to how numerical and categorical features differ across classes, whether apparent differences are practically meaningful, and whether the multivariable feature space shows overlap, gradients, subgroups, or possible sources of bias.

The next practical step is to implement this workflow in the tabular-data EDA notebook using the processed dataset generated previously.
