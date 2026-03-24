import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "SalePrice"
DEFAULT_FORM_FIELDS = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "YearBuilt",
    "Neighborhood",
]


def _to_python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _build_preprocessor(x_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x_frame.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features = x_frame.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _model_search_spaces() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "Linear Regression": (
            LinearRegression(),
            {"model__fit_intercept": [True, False]},
        ),
        "Ridge Regression": (
            Ridge(),
            {"model__alpha": [0.1, 1.0, 10.0, 50.0], "model__fit_intercept": [True, False]},
        ),
        "Lasso Regression": (
            Lasso(max_iter=100000),
            {"model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0], "model__fit_intercept": [True, False]},
        ),
    }


def _extract_feature_importance(best_estimator: Pipeline, top_n: int = 15) -> pd.DataFrame:
    model = best_estimator.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    preprocessor = best_estimator.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = np.abs(model.coef_)

    if coefficients.ndim > 1:
        coefficients = coefficients.mean(axis=0)

    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": coefficients}
    )
    feature_importance = feature_importance.sort_values(
        by="Importance", ascending=False
    ).head(top_n)

    return feature_importance.reset_index(drop=True)


def train_and_evaluate_models(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    random_state: int = 42,
    test_size: float = 0.2,
    cv: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    model_df = df.dropna(subset=[target_column]).copy()

    x = model_df.drop(columns=[target_column])
    y = model_df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    preprocessor = _build_preprocessor(x_train)
    model_spaces = _model_search_spaces()

    leaderboard_rows = []
    best_name = None
    best_estimator = None
    best_r2 = -np.inf

    for model_name, (estimator, search_space) in model_spaces.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=search_space,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(x_train, y_train)

        tuned_estimator = grid.best_estimator_
        predictions = tuned_estimator.predict(x_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        leaderboard_rows.append(
            {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Best Params": str(grid.best_params_),
                "CV RMSE": abs(grid.best_score_),
            }
        )

        if r2 > best_r2:
            best_r2 = r2
            best_name = model_name
            best_estimator = tuned_estimator

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by="R2", ascending=False
    ).reset_index(drop=True)

    if best_estimator is None or best_name is None:
        raise RuntimeError("Training failed to produce a best model")

    best_predictions = best_estimator.predict(x_test)
    prediction_vs_actual = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": best_predictions,
        }
    )

    feature_importance = _extract_feature_importance(best_estimator)

    defaults = {}
    feature_types = {}
    categorical_options = {}

    for col in x_train.columns:
        if pd.api.types.is_numeric_dtype(x_train[col]):
            defaults[col] = _to_python_scalar(x_train[col].median())
            feature_types[col] = "numeric"
        else:
            mode_values = x_train[col].mode(dropna=True)
            defaults[col] = _to_python_scalar(mode_values.iloc[0]) if not mode_values.empty else ""
            feature_types[col] = "categorical"
            unique_values = (
                x_train[col].dropna().astype(str).value_counts().head(50).index.tolist()
            )
            if defaults[col] not in unique_values and defaults[col] is not None:
                unique_values.insert(0, str(defaults[col]))
            categorical_options[col] = unique_values

    artifact = {
        "target_column": target_column,
        "best_model_name": best_name,
        "model": best_estimator,
        "feature_columns": x_train.columns.tolist(),
        "feature_types": feature_types,
        "defaults": defaults,
        "categorical_options": categorical_options,
        "form_fields": [f for f in DEFAULT_FORM_FIELDS if f in x_train.columns],
        "leaderboard": leaderboard.to_dict(orient="records"),
    }

    return leaderboard, prediction_vs_actual, feature_importance, artifact


def save_model_artifact(artifact: dict[str, Any], model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifact, model_path)


def load_model_artifact(model_path: str) -> dict[str, Any]:
    return joblib.load(model_path)


def get_feature_importance_from_artifact(
    artifact: dict[str, Any], top_n: int = 15
) -> pd.DataFrame:
    model = artifact.get("model")
    if model is None:
        return pd.DataFrame(columns=["Feature", "Importance"])
    return _extract_feature_importance(model, top_n=top_n)


def build_prediction_frame(
    user_inputs: dict[str, Any], artifact: dict[str, Any]
) -> pd.DataFrame:
    defaults = artifact["defaults"].copy()
    defaults.update(user_inputs)

    row = pd.DataFrame([defaults])
    ordered_columns = artifact["feature_columns"]

    for col in ordered_columns:
        if col not in row.columns:
            row[col] = artifact["defaults"].get(col)

    row = row[ordered_columns]
    return row
