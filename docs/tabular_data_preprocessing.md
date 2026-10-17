# Tabular Data Preprocessing

## 1. Purpose

Before training a machine-learning model, the input data must be converted into a clean, consistent, and reproducible table. This preliminary stage is called **data preprocessing**.

The purpose of preprocessing is not to discover biological, statistical, or predictive patterns in the dataset. It is to make sure that the data can be read correctly, that every column has a clear role and data type, and that obvious data-quality problems are handled before any downstream analysis.

This document focuses only on **tabular datasets**, where observations are stored in rows and variables are stored in columns.

The first practical example in this repository will follow this workflow and will be available in:

```text
notebooks/tabular_data/
```

That notebook will provide an executable example of the steps described here.

---

## 2. What is a tabular dataset?

A tabular dataset is organized as a two-dimensional table.

- **Rows** usually represent observations, samples, experiments, patients, proteins, compounds, measurements, or other individual records.
- **Columns** represent variables or attributes associated with each observation.

For example:

| sample_id | age | group | measurement | status |
|-----------|-----|-------|-------------|--------|
| S001 | 25 | A | 1.42 | positive |
| S002 | 31 | B | 0.91 | negative |
| S003 | 29 | A | 1.18 | positive |

In this table:

- `sample_id` is an identifier.
- `age` is a numerical variable.
- `group` is a categorical variable.
- `measurement` is a numerical variable.
- `status` is a categorical variable that could later be used as a label or target.

Before using the table, each of these roles must be understood clearly.

---

## 3. Recommended Python tools

For the first tabular-data examples, we will mainly use **pandas**.

```bash
pip install pandas
```

It is also common to use NumPy for numerical operations:

```bash
pip install numpy
```

Then they can be imported as:

```python
import numpy as np
import pandas as pd
```

Additional libraries may be introduced later when they are actually needed.

---

# 4. Reading the dataset

The first preprocessing task is to load the dataset correctly.

Never assume that a file was read correctly simply because Python did not raise an error. After loading a file, always verify its structure.

## 4.1 CSV files

```python
import pandas as pd

file_path = "data/example.csv"
df = pd.read_csv(file_path)
```

If the file uses another separator, specify it explicitly.

For a semicolon-separated file:

```python
df = pd.read_csv(file_path, sep=";")
```

For a tab-separated file:

```python
df = pd.read_csv(file_path, sep="\t")
```

## 4.2 Excel files

```python
df = pd.read_excel("data/example.xlsx")
```

An Excel workbook may contain several sheets. If necessary, select one explicitly:

```python
df = pd.read_excel(
    "data/example.xlsx",
    sheet_name="Sheet1",
)
```

## 4.3 Parquet files

```python
df = pd.read_parquet("data/example.parquet")
```

Parquet is particularly useful for larger datasets because it preserves column data types and is generally more efficient than CSV.

---

# 5. Verify that the file was loaded correctly

Immediately after reading the data, inspect a small number of rows.

```python
print(df.head())
```

You can also inspect the final rows:

```python
print(df.tail())
```

Then check the dimensions of the table:

```python
print(df.shape)
```

The output is a tuple:

```text
(number_of_rows, number_of_columns)
```

For example:

```text
(1250, 18)
```

means that the dataset contains 1,250 rows and 18 columns.

This is a **structural check**, not an analysis of the dataset.

---

# 6. Inspect the columns

Before modifying anything, inspect the column names.

```python
print(df.columns)
```

A useful alternative is:

```python
for column in df.columns:
    print(column)
```

Each column should have a clear meaning.

Before continuing, try to identify whether each column is primarily:

- an identifier,
- a numerical variable,
- a categorical variable,
- free text,
- a date or time variable,
- a label or target,
- or metadata that should not be used directly as a model feature.

This classification is important because different column types require different preprocessing operations.

---

# 7. Clean column names

Column names obtained from external files are often inconsistent.

Examples include:

```text
Sample ID
sample-ID
 Measurement
GROUP NAME
```

A simple standardization strategy is:

```python
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)
```

This can transform:

```text
Sample ID
```

into:

```text
sample_id
```

The objective is not to enforce one universal naming convention. The important point is to use a **consistent convention throughout the project**.

After renaming columns, verify them again:

```python
print(df.columns.tolist())
```

---

# 8. Inspect data types

Use:

```python
print(df.dtypes)
```

or:

```python
df.info()
```

Typical pandas data types include:

```text
int64
float64
object
bool
datetime64
```

The detected type is not always the correct semantic type.

For example, a numerical column may be loaded as text if one row contains an unexpected string:

```text
12.5
11.9
not available
13.2
```

In that case, pandas may classify the entire column as `object`.

Always compare the detected type with what the column is supposed to represent.

---

# 9. Convert incorrect data types

## 9.1 Numerical variables

A numerical column stored as text can be converted with:

```python
df["measurement"] = pd.to_numeric(
    df["measurement"],
    errors="coerce",
)
```

Using:

```python
errors="coerce"
```

converts invalid values into missing values (`NaN`) instead of stopping the program.

Those missing values must then be inspected and handled explicitly.

## 9.2 Categorical variables

```python
df["group"] = df["group"].astype("category")
```

This is optional during early preprocessing, but it can make the intended role of the column clearer.

## 9.3 Boolean variables

Suppose a column contains:

```text
yes
no
yes
no
```

It can be converted with an explicit mapping:

```python
df["active"] = df["active"].map({
    "yes": True,
    "no": False,
})
```

Explicit mappings are preferable to hidden assumptions.

## 9.4 Dates

```python
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)
```

Again, invalid values become missing values and can then be identified.

---

# 10. Missing values

Missing data can appear in many forms.

Examples include:

```text
NaN
NA
N/A
None
null
-
unknown
missing
```

Some of these values may be interpreted automatically by pandas, while others may remain as ordinary strings.

## 10.1 Count missing values

```python
missing_values = df.isna().sum()
print(missing_values)
```

To inspect only columns containing missing values:

```python
missing_values = df.isna().sum()
print(missing_values[missing_values > 0])
```

## 10.2 Define additional missing-value symbols during import

```python
missing_tokens = [
    "NA",
    "N/A",
    "null",
    "None",
    "missing",
    "unknown",
]

df = pd.read_csv(
    "data/example.csv",
    na_values=missing_tokens,
)
```

This helps represent missing information consistently.

## 10.3 Do not immediately fill missing values

A common mistake is to replace every missing value immediately with zero, a mean, or the most frequent category.

For example, this should **not** be done automatically:

```python
# Do not use blindly.
df = df.fillna(0)
```

A missing value and a numerical zero usually represent different things.

At the preprocessing stage, the first objective is to:

1. identify missing values,
2. standardize how they are represented,
3. document them,
4. decide later how they should be handled according to the problem and modeling workflow.

---

# 11. Duplicated rows

Duplicated records are another common data-quality problem.

Count duplicated rows with:

```python
n_duplicates = df.duplicated().sum()
print(n_duplicates)
```

Inspect them with:

```python
duplicates = df[df.duplicated(keep=False)]
print(duplicates)
```

If duplicates are confirmed to be accidental copies, they can be removed:

```python
df = df.drop_duplicates().copy()
```

However, never remove duplicates automatically without understanding what a row represents.

Repeated rows may sometimes correspond to legitimate repeated measurements or replicated experiments.

---

# 12. Duplicated identifiers

A table may contain unique observations but repeated identifiers.

Suppose `sample_id` should uniquely identify each row.

Check this explicitly:

```python
n_duplicated_ids = df["sample_id"].duplicated().sum()
print(n_duplicated_ids)
```

Inspect repeated identifiers:

```python
duplicated_ids = df[
    df["sample_id"].duplicated(keep=False)
]

print(duplicated_ids)
```

A repeated identifier may indicate:

- accidental duplication,
- repeated measurements,
- multiple observations belonging to the same entity,
- or an incorrectly defined identifier.

The correct action depends on the dataset design.

---

# 13. Clean text values

Categorical columns often contain values that appear identical to a human but are different to Python.

For example:

```text
Control
control
CONTROL
 control
Control
```

These are different strings.

A basic normalization is:

```python
df["group"] = (
    df["group"]
    .astype("string")
    .str.strip()
    .str.lower()
)
```

After normalization, all previous values become:

```text
control
```

The appropriate normalization depends on the meaning of the variable. Do not convert text blindly if capitalization or spacing carries information.

---

# 14. Standardize categorical labels

Even after text cleaning, the same concept may have multiple representations.

For example:

```text
positive
pos
+
1
```

If all of them truly represent the same class, define an explicit mapping:

```python
label_mapping = {
    "positive": "positive",
    "pos": "positive",
    "+": "positive",
    "negative": "negative",
    "neg": "negative",
    "-": "negative",
}

df["status"] = df["status"].map(label_mapping)
```

Mappings should be written explicitly in the code so that preprocessing remains reproducible.

Avoid replacing labels manually inside a spreadsheet whenever possible.

---

# 15. Inspect unexpected categorical values

Before applying a mapping, inspect the values present in a categorical column.

```python
print(df["group"].unique())
```

For a compact count:

```python
print(df["group"].value_counts(dropna=False))
```

At this stage, this operation is used as a **data-quality check** to detect spelling differences, invalid categories, unexpected codes, or missing-value tokens.

It should not yet be interpreted as a statistical analysis of class composition.

---

# 16. Check numerical conversion problems

Suppose a column should contain only numbers.

Convert it using:

```python
converted = pd.to_numeric(
    df["measurement"],
    errors="coerce",
)
```

Then identify rows where conversion failed:

```python
invalid_mask = (
    converted.isna()
    & df["measurement"].notna()
)

invalid_rows = df.loc[invalid_mask]
print(invalid_rows)
```

This is useful for detecting values such as:

```text
<5
not measured
error
12 mg
```

These values should be understood and handled explicitly rather than silently converted.

---

# 17. Check impossible or invalid values

Some columns have known valid domains.

For example, if a percentage should be between 0 and 100:

```python
invalid_percentage = df[
    (df["percentage"] < 0)
    | (df["percentage"] > 100)
]
```

If an age cannot be negative:

```python
invalid_age = df[df["age"] < 0]
```

If a categorical variable has a fixed set of valid categories:

```python
valid_groups = {"control", "treated"}

invalid_groups = df[
    ~df["group"].isin(valid_groups)
    & df["group"].notna()
]
```

These checks should come from the meaning of the variables, not arbitrary assumptions.

---

# 18. Identifier columns

Identifiers require special attention.

Examples include:

```text
sample_id
patient_id
protein_id
experiment_id
record_id
```

Identifiers are useful for tracking observations, but they are usually **not meaningful numerical features** for machine-learning models.

For example:

```text
sample_001
sample_002
sample_003
```

contains ordering information for humans, but that ordering should not automatically become predictive information.

During preprocessing, identifier columns should be identified and preserved so that observations can always be traced back to their source.

Do not delete identifiers unnecessarily, but do not treat them as ordinary numerical features either.

---

# 19. Target or label columns

If the dataset contains the variable that will eventually be predicted, identify it explicitly.

For example:

```python
target_column = "status"
```

At this point, the main preprocessing tasks are to verify that:

- the column exists,
- its values are represented consistently,
- missing labels are identifiable,
- and invalid label values are detected.

The statistical properties of the target will be examined later during dataset analysis.

---

# 20. Feature columns and metadata

Not every column in a dataset should necessarily be used as an input feature.

A table may contain:

- identifiers,
- experimental metadata,
- acquisition information,
- file names,
- comments,
- source information,
- labels,
- and actual predictor variables.

It is useful to define these groups explicitly.

For example:

```python
id_columns = ["sample_id"]

target_column = "status"

metadata_columns = [
    "source",
    "collection_date",
]

feature_columns = [
    "age",
    "measurement_1",
    "measurement_2",
]
```

This makes later code easier to understand and reduces the risk of accidentally using inappropriate columns during modeling.

---

# 21. Remove unnecessary columns deliberately

Columns should not be removed simply because they appear inconvenient.

If a column is unnecessary, document the reason.

For example:

```python
columns_to_remove = [
    "unnamed: 0",
]

df = df.drop(columns=columns_to_remove)
```

A common CSV artifact is a column such as:

```text
Unnamed: 0
```

which may have been created when a previous DataFrame index was accidentally exported.

Even in this simple case, verify what the column contains before removing it.

---

# 22. Reset the row index when necessary

After filtering or removing rows, pandas may preserve the original row numbers.

For example:

```text
0
1
4
7
8
```

If the old index has no meaning, reset it:

```python
df = df.reset_index(drop=True)
```

Do not confuse the DataFrame index with a meaningful dataset identifier.

---

# 23. Preserve the raw data

A fundamental rule is:

> **Never overwrite the original raw dataset during preprocessing.**

A useful project structure is:

```text
data/
├── raw/
│   └── original_dataset.csv
└── processed/
    └── cleaned_dataset.csv
```

The raw file should remain unchanged.

Preprocessing scripts or notebooks should read from `data/raw/` and write cleaned data to `data/processed/`.

This makes it possible to reproduce the complete preprocessing workflow from the original source.

---

# 24. Save the cleaned dataset

After preprocessing, save the resulting table explicitly.

For CSV:

```python
df.to_csv(
    "data/processed/cleaned_dataset.csv",
    index=False,
)
```

For Parquet:

```python
df.to_parquet(
    "data/processed/cleaned_dataset.parquet",
    index=False,
)
```

Avoid overwriting the raw file.

---

# 25. Validate the final preprocessed table

Before moving to dataset analysis or machine learning, perform a final structural validation.

For example:

```python
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Duplicated rows:", df.duplicated().sum())
print("Missing values:")
print(df.isna().sum())
print("Data types:")
print(df.dtypes)
```

You should be able to answer the following questions:

- Can the file be read reproducibly?
- Are the column names consistent?
- Is the meaning of every column known?
- Are the data types correct?
- Are missing values represented consistently?
- Were accidental duplicates identified?
- Are identifiers clearly separated from features?
- Is the target column clearly identified, when one exists?
- Were invalid values detected?
- Is the cleaned table saved separately from the raw data?

If the answer to these questions is yes, the dataset is ready for the next stage.

---

# 26. Preprocessing is not dataset analysis

This distinction is important throughout this repository.

**Preprocessing** asks questions such as:

```text
Can the file be read correctly?
Are the columns correctly named?
Are the data types correct?
Are there missing values?
Are labels written consistently?
Are there duplicated records?
Are there invalid values?
```

**Dataset analysis** asks different questions, such as:

```text
How are numerical variables distributed?
Are the classes balanced?
Which variables are associated with each other?
Are there groups or clusters in the data?
Are some observations unusual?
```

Those questions belong to a later stage and will be covered separately.

Keeping preprocessing and analysis separate makes workflows easier to understand, debug, reproduce, and reuse.

---

# 27. Reproducible preprocessing

Whenever possible, preprocessing should be implemented as executable Python code instead of manual spreadsheet editing.

For example, prefer:

```python
df["group"] = (
    df["group"]
    .str.strip()
    .str.lower()
)
```

over manually editing hundreds of cells in Excel.

Code provides a record of exactly what was changed and allows the same operations to be repeated on another version of the dataset.

A good preprocessing workflow should therefore be:

- explicit,
- reproducible,
- minimally destructive,
- easy to inspect,
- and based on clearly documented decisions.

---

# 28. Keep preprocessing steps ordered

A simple tabular preprocessing workflow can follow this order:

```text
1. Define the input file
2. Read the dataset
3. Verify rows and columns
4. Inspect and standardize column names
5. Identify the role of each column
6. Inspect and correct data types
7. Standardize missing values
8. Clean text and categorical values
9. Inspect duplicated rows and identifiers
10. Detect invalid or impossible values
11. Remove only clearly unnecessary columns
12. Reset the index if needed
13. Validate the cleaned table
14. Save the processed dataset separately
```

The exact steps may change depending on the dataset, but preprocessing should always follow an explicit and reproducible sequence.

---

# 29. What comes later?

After preprocessing, the next stages may include:

- exploratory data analysis,
- visualization,
- feature engineering,
- train/test splitting,
- scaling and normalization,
- categorical encoding,
- feature selection,
- model training,
- and model evaluation.

These are intentionally outside the scope of this document.

Some transformations, especially scaling, imputation, categorical encoding, and feature selection, must be handled carefully after defining the training and test sets to avoid information leakage. They will therefore be introduced in the appropriate machine-learning sections instead of being applied blindly during initial preprocessing.

---

# 30. First repository example

The first practical demonstration of tabular preprocessing will be developed under:

```text
notebooks/tabular_data/
```

The notebook will follow the workflow described here using a concrete dataset and executable Python code.

The objective of that first example is to teach how to move from a raw tabular file to a clean, well-defined, and reproducible table **before performing dataset analysis or training a machine-learning model**.
