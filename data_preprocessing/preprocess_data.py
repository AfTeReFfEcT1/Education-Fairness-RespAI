from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def preprocess_data(X: pd.DataFrame, y: pd.DataFrame, var_data: pd.DataFrame, seed: str):
    # Include only G3 as the target
    y_updated = y["G3"]

    # Copy X, exclude 'sex' attribute
    X_updated = X.copy()
    X_updated.drop("sex", axis=1)

    # Split data into a stratified train and test set
    # Stratify on target and sex
    stratify_groups = str(y) + "_" + X["sex"]
    idx_train, idx_test = train_test_split(X.index, test_size=0.2, stratify=stratify_groups, random_state=seed)

    X_train, X_test = X_updated.loc[idx_train], X_updated.loc[idx_test]
    y_train, y_test = y_updated.loc[idx_train], y_updated.loc[idx_test]

    # Transform data
    metadata = var_data
    categorical_cols = metadata.loc[
        (metadata["role"] == "Feature") &
        (metadata["type"] == "Categorical"),
        "name"
    ].tolist()

    binary_cols = metadata.loc[
        (metadata["role"] == "Feature") &
        (metadata["type"] == "Binary"),
        "name"
    ].tolist()

    integer_cols = metadata.loc[
        (metadata["role"] == "Feature") &
        (metadata["type"] == "Integer"),
        "name"
    ].tolist()


    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    binary_transformer = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False ,drop="if_binary"))
    ])

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_transformer, categorical_cols),
        ("binary", binary_transformer, binary_cols),
        ("numeric", numeric_transformer, integer_cols),
    ]).set_output(transform="pandas")

    X_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_preprocessed, X_test_preprocessed, y_test, y_train


if __name__ == "__main__":
    # fetch dataset
    student_performance = fetch_ucirepo(id=320)

    # data (as pandas dataframes)
    X = student_performance.data.features
    y = student_performance.data.targets
    var_data = student_performance.variables

    # preprocess data
    print(preprocess_data(X, y,var_data, seed=42))
