"""EDA and pre-processing for the clinical/demographic (tabular) dataset.

Dataset: Alzheimer's Disease Dataset (El Kharoua, Kaggle).
Prepares the tabular data for the XGBoost and MLP baseline models, and keeps
the demographic columns (Age, Gender, Ethnicity, EducationLevel) available for
the fairness/bias analysis.

Author: Group 15 (Data Science Professional Team Project, 7PAM2033)
Date: 2026-05
"""


# The module includes a descriptive header with project information
# dataset source and purpose of the script.

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Columns that are not predictive features.
ID_COLUMNS = ["PatientID", "DoctorInCharge"]# configuration values are defined as constant improving maintainability
TARGET_COLUMN = "Diagnosis"

# Demographic columns kept aside for fairness analysis (not dropped, but tracked).
DEMOGRAPHIC_COLUMNS = ["Age", "Gender", "Ethnicity", "EducationLevel"]

RANDOM_STATE = 42


# --- Loading -----------------------------------------------------------------

# The function correctly validates whether the file exists before loading the dataset
# consider adding validation to ensure the loaded CSV is not empty and contains the required target column
def load_clinical_data(csv_path: str | Path) -> pd.DataFrame:# function naming follows PEP8 standards and clearly describes the intended behaviour
 # Comprehensive docstrings are provided for parameters, return values, and exceptions
    """Load the clinical CSV into a DataFrame.

    Args:
        csv_path: Path to the El Kharoua Alzheimer's Disease CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Clinical CSV not found at: {path}")

    # consider validating the expected schema after loading the CSV to ensure all required columns are present before continuing the pipeline
    df = pd.read_csv(path)
    logger.info("Loaded clinical data: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


# --- EDA ---------------------------------------------------------------------
# The EDA function checks missing values and class balance which aligns with the data validation requirements
# Consider adding checks for invalid values such as negative ages or unexpected category
def run_eda(df: pd.DataFrame) -> None:#The EDA function is easy to understand and well documented, consider adding visualisations for support

    """Print a concise exploratory summary of the clinical data.

    Args:
        df: The raw clinical DataFrame.
    """
    logger.info("=== EDA: shape %s ===", df.shape)

    # Missing values per column (expect very few for this dataset).
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("No missing values found.")
    else:
        logger.info("Missing values:\n%s", missing.to_string())

    # Target balance: how many positive vs negative diagnoses.
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(dropna=False)
        ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
        logger.info("Target balance (%s):\n%s", TARGET_COLUMN, counts.to_string())
        logger.info("Imbalance ratio: %.2f", ratio)

    # Demographic spread, important for the fairness work.
    for col in DEMOGRAPHIC_COLUMNS:# Demographic attributes are retained for fairness analysis
        if col in df.columns:
            logger.info("Distribution of %s:\n%s", col, df[col].value_counts().to_string())

    # Duplicate rows.
    n_dupes = int(df.duplicated().sum())
    logger.info("Exact duplicate rows: %d", n_dupes)

# good fairness-oriented validation
# consider reporting subgroup sample sizes in percentages as well as counts
def check_subgroup_representation(
    y: pd.Series, sensitive: pd.DataFrame
) -> None:
    """Warn if any demographic subgroup is empty within the labels.

    Args:
        y: Target labels.
        sensitive: DataFrame of demographic columns aligned to ``y``.
    """
    # good implementation for fairness monitoring
    for col in sensitive.columns:
        group_counts = sensitive[col].value_counts()
        empty = group_counts[group_counts == 0]
        if not empty.empty:
            logger.warning("Empty subgroup(s) in %s: %s", col, list(empty.index))
        else:
            logger.info("All subgroups present in %s (%d groups).", col, group_counts.size)


# --- Pre-processing ----------------------------------------------------------
# Duplicate records are removed correctly
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and exact duplicate rows.

    Args:
        df: Raw clinical DataFrame.

    Returns:
        A cleaned copy of the DataFrame.
    """
    cleaned = df.copy()

    # Non-feature columns are removed before modelling,
    # Drop ID/placeholder columns if present (DoctorInCharge is a constant placeholder).
    to_drop = [c for c in ID_COLUMNS if c in cleaned.columns]
    if to_drop:
        cleaned = cleaned.drop(columns=to_drop)
        logger.info("Dropped non-feature columns: %s", to_drop)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d duplicate rows.", before - len(cleaned))

    return cleaned


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate features, target, and the demographic columns for fairness.

    Args:
        df: Cleaned clinical DataFrame.

    Returns:
        A tuple ``(X, y, sensitive)`` where ``X`` is the feature matrix,
        ``y`` is the target, and ``sensitive`` holds the demographic columns
        (a copy, also still present inside ``X``).

    Raises:
        KeyError: If the target column is missing.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in data.")

    y = df[TARGET_COLUMN].astype(int)# Consider validating that the target column contains
    x = df.drop(columns=[TARGET_COLUMN])

    available_demo = [c for c in DEMOGRAPHIC_COLUMNS if c in x.columns]
    sensitive = x[available_demo].copy()

    logger.info("Features: %d columns | Target: '%s'", x.shape[1], TARGET_COLUMN)
    return x, y, sensitive

# Scaling is correctly applied after train-test splitting
# Type hints are used consistently throughout the codebase which improves readability and compatibitliy
def scale_numeric_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standard-scale continuous numeric columns (fit on train only).

    Binary/categorical-coded columns are left unchanged. The scaler is fit on
    the training set only, to avoid data leakage into the test set.

    Args:
        x_train: Training feature matrix.
        x_test: Test feature matrix.

    Returns:
        Tuple ``(x_train_scaled, x_test_scaled, scaler)``.
    """
    # Treat columns with more than two unique values as continuous.
    numeric_cols = [
        c for c in x_train.columns
        if x_train[c].dtype != "object" and x_train[c].nunique() > 2
    ]

    scaler = StandardScaler()
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()

    # Good practice fitting the scaler only on training data which prevents data leakage
    x_train_scaled[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
    x_test_scaled[numeric_cols] = scaler.transform(x_test[numeric_cols])

    logger.info("Scaled %d continuous columns (fit on train only).", len(numeric_cols))
    return x_train_scaled, x_test_scaled, scaler

# the preprocessing pipeline is logically organised into loading, EDA, cleaning, splitting, and scaling stages
# this improves readability and maintainability
def preprocess_clinical(
    csv_path: str | Path, test_size: float = 0.20
) -> dict[str, object]:
    """Full tabular pipeline: load, EDA, clean, split, scale.

    Args:
        csv_path: Path to the clinical CSV.
        test_size: Fraction of data held out for testing.

    Returns:
        A dictionary with the processed splits and the fitted scaler.
    """
    df = load_clinical_data(csv_path)
    run_eda(df)

    cleaned = clean_data(df)
    x, y, sensitive = split_features_target(cleaned)
    check_subgroup_representation(y, sensitive)

    # Stratify on the target so both classes appear in train and test.
    # Stratified sampling is correctly implemented to preserve class distribution
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train: %d rows | Test: %d rows", len(x_train), len(x_test))

    x_train_scaled, x_test_scaled, scaler = scale_numeric_features(x_train, x_test)

    # the return function returns multiple outputs using a dictionary
    # consider using a dataclass or custom object to improve readabiltiy and reduce the risk of key-access errors
    return {
        "x_train": x_train_scaled,
        "x_test": x_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        # Demographic slice of the test set, aligned by index for fairness metrics.
        "sensitive_test": x_test[[c for c in DEMOGRAPHIC_COLUMNS if c in x_test.columns]],
    }

# consider adding unit tests for the data loading, cleaning, and scaling functions
# to align with the project test plan and improve code reliability 
if __name__ == "__main__":
    # Update this path to wherever the Kaggle CSV is downloaded.
    CSV_PATH = "data/raw/clinical/alzheimers_disease_data.csv"
    try:
        result = preprocess_clinical(CSV_PATH)
        logger.info("Clinical pre-processing complete. Train shape: %s",
                    result["x_train"].shape)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Pre-processing failed: %s", exc)

# The __main__ guard is implemented correctly,
# allowing the module to be imported without executing the pipeline
