
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocessing import load_data
from src.prediction import (
    load_model,
    predict_attrition,
    predict_single_employee
)


PROJECT_ROOT = Path(__file__).resolve().parent
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
THRESHOLD_PATH = METRICS_DIR / "thresholds.json"

MODEL_NAMES = [
    "Random Forest",
    "XGBoost",
    "Ensemble"
]

DROP_COLUMNS = [
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]

EDA_PLOTS = [
    "attrition_distribution_modular.png",
    "correlation_heatmap_modular.png",
    "attrition_by_department.png",
    "attrition_by_overtime.png",
    "attrition_by_age_group.png",
    "attrition_by_years_at_company.png"
]

SHAP_PLOTS = [
    "random_forest_shap_bar.png",
    "random_forest_shap_dot.png",
    "xgboost_shap_bar.png",
    "xgboost_shap_dot.png"
]


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# CACHED DATA
# --------------------------------------------------

@st.cache_data
def get_dataset():
    return load_data()


@st.cache_resource
def get_trained_model(model_name):
    return load_model(model_name)


@st.cache_data
def get_thresholds():
    if not THRESHOLD_PATH.exists():
        return None

    with open(THRESHOLD_PATH, "r") as file:
        return json.load(file)


def get_feature_columns(model):
    """Retrieve the original input feature names."""

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    preprocessor = model.named_steps["preprocessor"]

    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    raise ValueError("Model feature names are unavailable.")


def display_plots(plot_files):
    """Display available visualizations in two columns."""

    available_plots = [
        filename for filename in plot_files
        if (PLOT_DIR / filename).exists()
    ]

    if not available_plots:
        st.info(
            "No visualizations found. "
            "Run the training workflow first."
        )
        return

    for i in range(0, len(available_plots), 2):

        col1, col2 = st.columns(2)

        with col1:
            plot_path = PLOT_DIR / available_plots[i]

            st.image(
                str(plot_path),
                caption=plot_path.stem.replace(
                    "_", " "
                ).title(),
                width="stretch"
            )

        if i + 1 < len(available_plots):
            with col2:
                plot_path = PLOT_DIR / available_plots[i + 1]

                st.image(
                    str(plot_path),
                    caption=plot_path.stem.replace(
                        "_", " "
                    ).title(),
                    use_container_width=True
                )


# --------------------------------------------------
# LOAD PROJECT DATA
# --------------------------------------------------

st.title("Employee Attrition Prediction")

st.write(
    "An interactive machine learning application to explore "
    "employee attrition, compare predictive models, and estimate "
    "employee turnover."
)

try:
    df = get_dataset()

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


thresholds = get_thresholds()

if thresholds is None:
    st.warning(
        "Trained model thresholds are not available. "
        "Please run `python train.py` first."
    )
    st.stop()

missing_thresholds = [
    name for name in MODEL_NAMES
    if name not in thresholds
]

if missing_thresholds:
    st.error(
        "Thresholds are missing for: "
        + ", ".join(missing_thresholds)
        + ". Please retrain the models."
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Exploratory Data Analysis",
        "Model Performance",
        "Model Explainability",
        "Model Prediction"
    ]
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    total_employees = len(df)

    attrition_count = int(
        (df["Attrition"] == "Yes").sum()
    )

    retention_count = int(
        (df["Attrition"] == "No").sum()
    )

    attrition_rate = (
        attrition_count / total_employees * 100
        if total_employees else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Employees", total_employees)
    col2.metric("Employees Left", attrition_count)
    col3.metric("Employees Retained", retention_count)
    col4.metric("Attrition Rate", f"{attrition_rate:.2f}%")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Number of rows:** {df.shape[0]}")
        st.write(f"**Number of columns:** {df.shape[1]}")

    with col2:
        st.write(f"**Attrition cases:** {attrition_count}")
        st.write(f"**Retention cases:** {retention_count}")

    st.subheader("Project Models")

    st.write(
        "- Random Forest Classifier\n"
        "- XGBoost Classifier\n"
        "- Soft Voting Ensemble"
    )


# --------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

elif page == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    st.write(
        "Explore employee attrition patterns through "
        "dataset visualizations."
    )

    display_plots(EDA_PLOTS)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance Comparison")

    st.write(
        "Compare the evaluation metrics of the three "
        "trained machine learning models."
    )

    metrics_path = METRICS_DIR / "model_comparison.csv"

    if metrics_path.exists():

        comparison_df = pd.read_csv(metrics_path)

        st.subheader("Model Comparison Results")

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        st.download_button(
            label="Download Model Comparison",
            data=comparison_df.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "Model comparison results are not available yet. "
            "Run `python train.py` to generate them."
        )


# --------------------------------------------------
# MODEL EXPLAINABILITY
# --------------------------------------------------

elif page == "Model Explainability":

    st.header("Model Explainability")

    st.write(
        "SHAP (SHapley Additive exPlanations) helps explain "
        "how different employee features influence model predictions."
    )

    st.info(
        "These visualizations explain feature influence on model "
        "predictions. They do not establish cause and effect."
    )

    display_plots(SHAP_PLOTS)


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

elif page == "Model Prediction":

    st.header("Predict Employee Attrition")

    st.write(
        "Estimate employee attrition using an individual employee "
        "form or by uploading a CSV file containing multiple employees."
    )

    model_name = st.selectbox(
        "Select Machine Learning Model",
        MODEL_NAMES
    )

    threshold = float(thresholds[model_name])

    st.caption(
        f"Decision threshold for {model_name}: {threshold:.4f}"
    )

    try:
        model = get_trained_model(model_name)
        feature_columns = get_feature_columns(model)

    except Exception as error:
        st.error(f"Unable to load the selected model: {error}")
        st.stop()

    # ----------------------------------------------
    # INDIVIDUAL EMPLOYEE PREDICTION
    # ----------------------------------------------

    st.subheader("Individual Employee Prediction")

    st.write(
        "Enter the employee details to estimate their "
        "attrition probability."
    )

    categorical_columns = (
        df.drop(columns=["Attrition"], errors="ignore")
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )

    with st.form("individual_prediction_form"):

        employee_input = {}

        col1, col2 = st.columns(2)

        for index, feature in enumerate(feature_columns):

            target_column = col1 if index % 2 == 0 else col2

            with target_column:

                if feature in categorical_columns:

                    options = sorted(
                        df[feature].dropna().unique().tolist(),
                        key=str
                    )

                    if not options:
                        st.error(
                            f"No available values for {feature}."
                        )
                        continue

                    employee_input[feature] = st.selectbox(
                        feature,
                        options,
                        key=f"individual_{feature}"
                    )

                else:

                    median_value = float(df[feature].median())

                    if pd.api.types.is_integer_dtype(df[feature]):
                        employee_input[feature] = st.number_input(
                            feature,
                            value=int(median_value),
                            step=1,
                            key=f"individual_{feature}"
                        )
                    else:
                        employee_input[feature] = st.number_input(
                            feature,
                            value=median_value,
                            step=0.1,
                            key=f"individual_{feature}"
                        )

        individual_submit = st.form_submit_button(
            "Predict Employee Attrition"
        )

    if individual_submit:

        try:
            result = predict_single_employee(
                employee_input,
                model_name=model_name,
                threshold=threshold
            )

            probability = result["Attrition Probability"]
            prediction = result["Prediction"]
            outcome = result["Result"]

            st.subheader("Individual Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Attrition Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Prediction",
                outcome
            )

            if prediction == 1:
                st.warning(
                    "The model predicts that this employee "
                    "is likely to leave."
                )
            else:
                st.success(
                    "The model predicts that this employee "
                    "is likely to stay."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.divider()

    # ----------------------------------------------
    # BATCH CSV PREDICTION
    # ----------------------------------------------

    st.subheader("Batch Employee Prediction")

    st.write(
        "Upload a CSV containing multiple employee records. "
        "The CSV must contain all required model features."
    )

    template = pd.DataFrame(columns=feature_columns)

    st.download_button(
        label="Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="employee_input_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload Employee CSV",
        type=["csv"],
        key="batch_employee_upload"
    )

    if uploaded_file is not None:

        try:
            employee_data = pd.read_csv(uploaded_file)

            if employee_data.empty:
                st.error("The uploaded CSV file is empty.")

            else:

                st.write("**Uploaded Employee Data**")

                st.dataframe(
                    employee_data.head(),
                    use_container_width=True
                )

                if st.button("Predict Attrition for All Employees"):

                    prediction_results = predict_attrition(
                        employee_data,
                        model_name=model_name,
                        threshold=threshold
                    )

                    results = employee_data.copy()

                    results["Attrition Probability (%)"] = (
                        prediction_results[
                            "Attrition Probability"
                        ] * 100
                    ).round(2)

                    results["Predicted Attrition"] = (
                        prediction_results["Result"]
                    )

                    st.success(
                        "Predictions generated successfully!"
                    )

                    st.subheader("Batch Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    likely_to_leave = int(
                        (prediction_results["Prediction"] == 1).sum()
                    )

                    likely_to_stay = int(
                        (prediction_results["Prediction"] == 0).sum()
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Likely to Leave",
                        likely_to_leave
                    )

                    col2.metric(
                        "Likely to Stay",
                        likely_to_stay
                    )

                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_data,
                        file_name="employee_attrition_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as error:
            st.error(
                f"Batch prediction could not be completed: {error}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Employee Attrition Prediction Project"
)

st.sidebar.caption(
    "For educational and portfolio purposes."
)
