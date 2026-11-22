
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "isFraud"

EXCLUDED_COLUMNS = [
    "TransactionID",
    "isFraud",
    "period",
    "amount_band",
]


def prepare_features(
    data,
    include_identity=False,
    include_identity_indicator=False
):
    """
    Prepare feature matrix, target and preprocessing transformer.

    Parameters
    ----------
    data : pandas.DataFrame
        Merged transaction and identity dataset.
    include_identity : bool
        Whether to include identity features.
    include_identity_indicator : bool
        Whether to include the has_identity indicator.

    Returns
    -------
    X : pandas.DataFrame
        Selected model features.
    y : pandas.Series
        Fraud target.
    preprocessor : ColumnTransformer
        Unfitted preprocessing transformer.
    """

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    excluded = EXCLUDED_COLUMNS.copy()

    # Exclude time from the first baseline predictors.
    # Keep TransactionDT in the original data for chronological analysis.
    excluded.extend(["TransactionDT", "_merge", "has_identity"])

    # Exclude identity features when not requested.
    if not include_identity:
        excluded.extend([
            column for column in data.columns
            if column.startswith("id_")
            or column in ["DeviceType", "DeviceInfo"]
        ])

    X = data.drop(
        columns=[
            column for column in excluded
            if column in data.columns
        ]
    ).copy()

    y = data[TARGET_COLUMN].copy()

    # Add identity availability only when explicitly requested.
    if include_identity_indicator:
        if "has_identity" not in data.columns:
            raise ValueError(
                "has_identity column is required. "
                "Create it during the validated identity merge."
            )

        X["has_identity"] = data["has_identity"].astype(int)

    # Treat numeric-looking identifiers as categorical.
    categorical_identifiers = [
        column for column in [
            "card1", "card2", "card3", "card5",
            "addr1", "addr2"
        ]
        if column in X.columns
    ]

    for column in categorical_identifiers:
        X[column] = X[column].astype("string")

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Convert numerical features to float32
    X[numerical_columns] = X[numerical_columns].astype("float32")

    # Normalise missing values in categorical columns
    for column in categorical_columns:
        X[column] = X[column].astype(object)
        X[column] = X[column].where(X[column].notna(), np.nan)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="missing",
            missing_values=pd.NA
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=20,
            dtype=np.float32
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ], sparse_threshold=1.0
    )

    return X, y, preprocessor