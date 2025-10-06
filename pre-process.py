import pandas as pd
import numpy as np

def analyze_dataframe(
    df: pd.DataFrame,
    required_cols=None,
    max_nan_frac_per_col=0.05,
    min_unique_value_count=2,
    max_col_zero_frac=0.99,
    duplicate_value_to_nan=True,
    n_max_duplicates=144,
    value_to_replace=0,
    imputer_strategy="mean"
):
    """
    Analyze a DataFrame for preprocessing issues (without modifying it).

    Args:
        df (pd.DataFrame): Input dataframe.
        required_cols (list, optional): Fixed required columns.
        max_nan_frac_per_col (float): Max allowed NaN fraction per column.
        min_unique_value_count (int): Minimum unique values required in a column.
        max_col_zero_frac (float): Max allowed fraction of zeros in numeric columns.
        duplicate_value_to_nan (bool): Whether to check for long duplicate sequences.
        n_max_duplicates (int): Max allowed consecutive duplicate values.
        value_to_replace (int/float): Value considered for duplicate sequences.
        imputer_strategy (str): Default imputation strategy for numeric NaNs.

    Returns:
        dict: Report containing issues and suggestions.
    """

    if required_cols is None:
        required_cols = [
            "time_stamp", "asset_id", "train_test", "train_test_bool",
            "status_type_id", "status_type_bool"
        ]

    # Retain only required + *_avg columns
    cols_to_keep = required_cols + [c for c in df.columns if c.endswith("_avg")]
    df = df[cols_to_keep]

    report = {}

    # 1. NaN fraction check
    nan_fractions = df.isna().mean()
    nan_violations = nan_fractions[nan_fractions > max_nan_frac_per_col]
    report["high_nan_columns"] = nan_violations.to_dict()

    # 2. Unique value check
    low_unique = {}
    for col in df.columns:
        uniq_count = df[col].nunique(dropna=True)
        if uniq_count < min_unique_value_count:
            low_unique[col] = uniq_count
    report["low_unique_columns"] = low_unique

    # 3. Zero fraction check
    zero_fractions = (df == 0).mean(numeric_only=True)
    zero_violations = zero_fractions[zero_fractions > max_col_zero_frac]
    report["high_zero_columns"] = zero_violations.to_dict()

    # 4. Duplicate sequence check
    duplicate_report = {}
    if duplicate_value_to_nan:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                consecutive_dupes = (
                    (df[col] == value_to_replace)
                    .astype(int)
                    .groupby(df[col].ne(value_to_replace).cumsum())
                    .cumsum()
                )
                long_dupes = df[col][consecutive_dupes > n_max_duplicates]
                if not long_dupes.empty:
                    duplicate_report[col] = long_dupes.index.tolist()
    report["long_duplicate_sequences"] = duplicate_report

    # 5. Imputation strategy suggestion
    imputer_suggestions = {}
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                imputer_suggestions[col] = f"{imputer_strategy} imputation"
            else:
                imputer_suggestions[col] = "most_frequent imputation"
    report["imputation_suggestions"] = imputer_suggestions

    return report
