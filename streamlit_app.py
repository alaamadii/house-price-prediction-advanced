import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_loader import load_data
from pipeline_training import (
    build_prediction_frame,
    get_feature_importance_from_artifact,
    load_model_artifact,
    save_model_artifact,
    train_and_evaluate_models,
)

MODEL_PATH = os.path.join("models", "best_model.joblib")


def render_prediction_form(artifact: dict) -> None:
    st.subheader("Single House Price Prediction")

    form_fields = artifact.get("form_fields", [])
    defaults = artifact.get("defaults", {})
    feature_types = artifact.get("feature_types", {})
    categorical_options = artifact.get("categorical_options", {})

    if not form_fields:
        st.info("No compatible form fields found for this dataset.")
        return

    user_inputs = {}

    with st.form("prediction_form"):
        for field in form_fields:
            feature_type = feature_types.get(field)
            default_value = defaults.get(field)

            if feature_type == "numeric":
                safe_default = float(default_value) if default_value is not None else 0.0
                user_inputs[field] = st.number_input(field, value=safe_default)
            else:
                options = categorical_options.get(field, [])
                options = options if options else [str(default_value)]
                default_str = str(default_value) if default_value is not None else options[0]
                default_index = options.index(default_str) if default_str in options else 0
                user_inputs[field] = st.selectbox(field, options=options, index=default_index)

        predict_btn = st.form_submit_button("Predict Price")

    if predict_btn:
        input_frame = build_prediction_frame(user_inputs, artifact)
        prediction = artifact["model"].predict(input_frame)[0]
        st.success(f"Predicted House Price: ${prediction:,.2f}")


def render_model_download() -> None:
    if not os.path.exists(MODEL_PATH):
        return

    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    st.download_button(
        label="Download Saved Model",
        data=model_bytes,
        file_name="best_model.joblib",
        mime="application/octet-stream",
    )


def render_insights_page(artifact: dict | None) -> None:
    st.subheader("Model Insights")

    if artifact is None:
        st.info("Train a model first to see leaderboard and feature insights.")
        return

    leaderboard_records = artifact.get("leaderboard", [])
    if leaderboard_records:
        leaderboard_df = pd.DataFrame(leaderboard_records)
        st.markdown("### Leaderboard")
        st.dataframe(leaderboard_df, use_container_width=True)
        if "R2" in leaderboard_df.columns:
            st.bar_chart(leaderboard_df.set_index("Model")[["R2"]])

    feature_importance = get_feature_importance_from_artifact(artifact)
    st.markdown("### Feature Importance")
    if feature_importance.empty:
        st.info("Feature importance is not available for this model type.")
    else:
        st.dataframe(feature_importance, use_container_width=True)
        st.bar_chart(feature_importance.set_index("Feature")[["Importance"]])


def main() -> None:
    st.set_page_config(
        page_title="House Price Prediction Dashboard",
        page_icon=":house:",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #f9f5ef 0%, #e7efe9 45%, #dce6f2 100%);
        }
        .title-wrap {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.65);
            border: 1px solid rgba(60, 70, 90, 0.15);
        }
        .subtle {
            color: #2f3b52;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="title-wrap">
          <h1 style="margin-bottom: 0.2rem;">House Price Prediction App</h1>
          <p class="subtle">Pipeline + GridSearchCV + Saved model + Prediction form.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    page = st.sidebar.radio("Navigation", ["Train and Predict", "Insights"])

    left, right = st.columns([2, 1])

    df = None
    artifact = None
    with left:
        st.subheader("Dataset")
        uploaded_file = st.file_uploader("Upload train.csv", type=["csv"])
        use_default = st.checkbox(
            "Use default dataset path: data/train.csv", value=uploaded_file is None
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Uploaded dataset loaded successfully.")
        elif use_default:
            default_path = os.path.join("data", "train.csv")
            if os.path.exists(default_path):
                df = load_data(default_path)
                st.success(f"Loaded dataset from {default_path}")
            else:
                st.warning("Default dataset was not found. Upload a CSV file to continue.")

        if df is not None:
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.dataframe(df.head(10), use_container_width=True)

    if os.path.exists(MODEL_PATH):
        artifact = load_model_artifact(MODEL_PATH)

    with right:
        st.subheader("Model Status")
        if artifact is None:
            st.warning("No saved model found yet. Train once to create models/best_model.joblib")
        else:
            st.success(f"Loaded model: {artifact.get('best_model_name', 'Unknown')}")

    if page == "Train and Predict":
        st.write("")
        train_btn = st.button("Train, Tune, and Save Best Model", type="primary")

        if train_btn:
            if df is None:
                st.error("Please upload a dataset or enable the default dataset path.")
            elif "SalePrice" not in df.columns:
                st.error("Dataset must include the SalePrice target column.")
            else:
                with st.spinner("Training and tuning models with GridSearchCV..."):
                    leaderboard, pred_vs_actual, feature_importance, trained_artifact = train_and_evaluate_models(df)
                    save_model_artifact(trained_artifact, MODEL_PATH)

                artifact = trained_artifact
                st.session_state["pred_vs_actual"] = pred_vs_actual
                st.session_state["feature_importance"] = feature_importance
                st.success(f"Training complete. Best model saved to {MODEL_PATH}")

                st.subheader("Model Comparison")
                st.dataframe(leaderboard, use_container_width=True)
                st.bar_chart(leaderboard.set_index("Model")[["R2"]])

                st.subheader("Prediction vs Actual")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(pred_vs_actual["Actual"], pred_vs_actual["Predicted"], alpha=0.55)
                line_min = min(pred_vs_actual["Actual"].min(), pred_vs_actual["Predicted"].min())
                line_max = max(pred_vs_actual["Actual"].max(), pred_vs_actual["Predicted"].max())
                ax.plot([line_min, line_max], [line_min, line_max], color="red", linestyle="--")
                ax.set_xlabel("Actual Price")
                ax.set_ylabel("Predicted Price")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

                st.subheader("Top Feature Importance")
                if feature_importance.empty:
                    st.info("Feature importance is not available for this model type.")
                else:
                    st.dataframe(feature_importance, use_container_width=True)
                    st.bar_chart(feature_importance.set_index("Feature")[["Importance"]])

        st.write("")
        if artifact is not None:
            render_model_download()
            render_prediction_form(artifact)

    if page == "Insights":
        render_insights_page(artifact)
        pred_vs_actual = st.session_state.get("pred_vs_actual")
        if pred_vs_actual is not None and not pred_vs_actual.empty:
            st.markdown("### Prediction vs Actual")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(pred_vs_actual["Actual"], pred_vs_actual["Predicted"], alpha=0.55)
            line_min = min(pred_vs_actual["Actual"].min(), pred_vs_actual["Predicted"].min())
            line_max = max(pred_vs_actual["Actual"].max(), pred_vs_actual["Predicted"].max())
            ax.plot([line_min, line_max], [line_min, line_max], color="red", linestyle="--")
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
        else:
            st.info("Train in 'Train and Predict' page first to view prediction vs actual chart.")

    st.caption("Run from terminal: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
