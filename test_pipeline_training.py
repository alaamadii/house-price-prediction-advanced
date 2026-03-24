import os

import pandas as pd

from src.pipeline_training import (
    build_prediction_frame,
    get_feature_importance_from_artifact,
    load_model_artifact,
    save_model_artifact,
    train_and_evaluate_models,
)


def make_sample_df(rows: int = 80) -> pd.DataFrame:
    data = {
        "OverallQual": [5 + (i % 5) for i in range(rows)],
        "GrLivArea": [900 + i * 12 for i in range(rows)],
        "GarageCars": [1 + (i % 3) for i in range(rows)],
        "TotalBsmtSF": [700 + i * 6 for i in range(rows)],
        "1stFlrSF": [800 + i * 4 for i in range(rows)],
        "FullBath": [1 + (i % 2) for i in range(rows)],
        "YearBuilt": [1970 + (i % 40) for i in range(rows)],
        "Neighborhood": ["NAmes", "CollgCr", "OldTown", "Edwards"] * (rows // 4),
    }

    df = pd.DataFrame(data)
    df["SalePrice"] = (
        50000
        + df["OverallQual"] * 22000
        + df["GrLivArea"] * 75
        + df["GarageCars"] * 6000
        + df["TotalBsmtSF"] * 30
        + (df["YearBuilt"] - 1950) * 500
    )

    return df


def test_train_and_return_shapes():
    df = make_sample_df()
    leaderboard, pred_vs_actual, feature_importance, artifact = train_and_evaluate_models(
        df, cv=2, test_size=0.25
    )

    assert not leaderboard.empty
    assert {"Model", "MAE", "RMSE", "R2", "CV RMSE"}.issubset(leaderboard.columns)
    assert not pred_vs_actual.empty
    assert {"Actual", "Predicted"}.issubset(pred_vs_actual.columns)
    assert "model" in artifact
    assert "feature_columns" in artifact
    assert "defaults" in artifact
    assert isinstance(feature_importance, pd.DataFrame)


def test_save_load_and_predict_frame(tmp_path):
    df = make_sample_df()
    _, _, _, artifact = train_and_evaluate_models(df, cv=2, test_size=0.25)

    model_path = tmp_path / "best_model.joblib"
    save_model_artifact(artifact, str(model_path))
    assert os.path.exists(model_path)

    loaded = load_model_artifact(str(model_path))
    assert "model" in loaded

    sample_inputs = {
        "OverallQual": 7,
        "GrLivArea": 1500,
        "GarageCars": 2,
        "TotalBsmtSF": 900,
        "1stFlrSF": 1200,
        "FullBath": 2,
        "YearBuilt": 2001,
        "Neighborhood": "NAmes",
    }

    input_frame = build_prediction_frame(sample_inputs, loaded)
    pred = loaded["model"].predict(input_frame)

    assert input_frame.shape[0] == 1
    assert len(pred) == 1


def test_feature_importance_from_artifact():
    df = make_sample_df()
    _, _, _, artifact = train_and_evaluate_models(df, cv=2, test_size=0.25)

    fi = get_feature_importance_from_artifact(artifact, top_n=10)
    assert isinstance(fi, pd.DataFrame)
    assert {"Feature", "Importance"}.issubset(fi.columns)
