import pandas as pd
from sklearn.impute import SimpleImputer
from energy_fault_detector.quick_fault_detection import quick_fault_detector
from energy_fault_detector.config import Config
import tempfile
import os

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
            "time_stamp", "train_test", "train_test_bool",
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
    # report["long_duplicate_sequences"] = duplicate_report

    # 5. Imputation strategy suggestion
    imputer_suggestions = {}
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                imputer_suggestions[col] = f"{imputer_strategy} imputation"
            else:
                imputer_suggestions[col] = "most_frequent imputation"
    # report["imputation_suggestions"] = imputer_suggestions

    return df,report


farm_path = "/content/drive/MyDrive/Wind Turbine/Care Dataset/CARE_To_Compare/Wind Farm B/7.csv"
farm_path = r"D:\Personal\Ideas\Wind turbine\CARE_To_Compare\CARE_To_Compare\Wind Farm B\asset_files\train_0.csv"
test_file_path = r"D:\Personal\Ideas\Wind turbine\CARE_To_Compare\CARE_To_Compare\Wind Farm B\asset_files\predict_0.csv"
output_root_path = r"D:\Personal\Ideas\Wind turbine\CARE_To_Compare\CARE_To_Compare\Wind Farm B\models"
# Load the data using pandas
df = pd.read_csv(farm_path)

# df,report = analyze_dataframe(df)
# df.to_csv(r"D:\Personal\Ideas\Wind turbine\CARE_To_Compare\CARE_To_Compare\Wind Farm B\asset_files\train_0_processed.csv", index=False)
# import pprint
# pprint.pprint(report)

# Identify numeric columns
numeric_cols = df.select_dtypes(include=["number"]).columns
df_processed = df[["time_stamp"]].copy()

if len(numeric_cols) > 0:
    df_numeric = df[numeric_cols]

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    df_imputed_numeric = pd.DataFrame(
        imputer.fit_transform(df_numeric), columns=numeric_cols
    )

    # Re-add non-numeric columns (time_stamp and status_type_id_bool) to the imputed dataframe
    df_processed[numeric_cols] = df_imputed_numeric
else:
    # If there are no numeric columns, simply keep the original non-numeric data
    df_processed = df_processed.join(df.drop(columns=["time_stamp"]))

# Create a temporary CSV file
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
    temp_csv_path = tmp_file.name
    df_processed.to_csv(temp_csv_path, index=False)

# Pass the path of the temporary CSV file to the quick_fault_detector
quick_fault_detector, quick_fault_detector_df = quick_fault_detector(temp_csv_path, None, "train_test_bool", None, "time_stamp", "status_type_bool")

# from energy_fault_detector.quick_fault_detection import quick_fault_detector

prediction_results, events, metadata = quick_fault_detector(
    csv_data_path=temp_csv_path,
    csv_test_data_path=test_file_path,
    mode="train",
    time_column_name="time_stamp",          # optional, if you need timestamp parsing
    model_directory=output_root_path, # optional; defaults to the package setting
    model_subdir="asset_0",                 # optional; becomes a subfolder under model_directory
    model_name="farm_b"            # optional; final folder for saved artifacts
)


# Optionally, remove the temporary file after use
# os.remove(temp_csv_path)
