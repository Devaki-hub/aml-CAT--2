
from pathlib import Path
import glob

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishguard_model.joblib"
PLOT_DIR = ROOT / "outputs" / "plots"
REPORT_DIR = ROOT / "outputs" / "reports"

st.set_page_config(
    page_title="PhishGuard XAI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ PhishGuard XAI")
st.subheader("Explainable Machine Learning for Phishing Detection")
st.write(
    "Explore model performance, dataset examples, feature importance, "
    "and SHAP explanations."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train.py` first.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]
model_name = saved["best_model_name"]

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Project Overview",
        "Model Comparison",
        "Visualizations",
        "Sample Prediction"
    ]
)

# Project overview
if page == "Project Overview":
    st.header("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Selected Model", model_name)
        st.write("**Dataset:** UCI PhiUSIIL Phishing URL Dataset")
        st.write("**Target column:** `label`")
        st.write("**Approach:** Supervised binary classification")

    with col2:
        st.write("### Project workflow")
        st.markdown(
            """
            1. Dataset loading and preprocessing
            2. Logistic Regression baseline
            3. Random Forest and optional XGBoost
            4. Model comparison using classification metrics
            5. Feature importance and SHAP explanations
            6. Interactive dashboard
            """
        )

    st.info(
        "This dashboard demonstrates the trained model on dataset records. "
        "It does not yet extract every required feature from an arbitrary "
        "live URL."
    )

# Model comparison
elif page == "Model Comparison":
    st.header("📊 Model Performance Comparison")

    result_path = REPORT_DIR / "model_comparison.csv"

    if result_path.exists():
        results = pd.read_csv(result_path)
        st.dataframe(results, use_container_width=True)

        if "F1" in results.columns:
            st.bar_chart(results.set_index("Model")["F1"])

        st.caption(
            "Metrics are from the existing training script's held-out "
            "test split. They are not independently reproduced here."
        )
    else:
        st.warning("Run the training script to generate model_comparison.csv.")

# Visualization gallery
elif page == "Visualizations":
    st.header("🔍 Model Analysis and Explainability")

    plots = [
        ("01_label_distribution.png", "1. Dataset Label Distribution"),
        ("02_correlation_heatmap.png", "2. Feature Correlation Heatmap"),
        ("03_confusion_matrix.png", "3. Confusion Matrix"),
        ("04_roc_curves.png", "4. ROC Curve Comparison"),
        ("05_feature_importance.png", "5. Feature Importance"),
        ("06_shap_summary.png", "6. SHAP Summary Plot")
    ]

    for i in range(0, len(plots), 2):
        cols = st.columns(2)

        for col, (filename, title) in zip(cols, plots[i:i + 2]):
            with col:
                path = PLOT_DIR / filename
                st.subheader(title)

                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.warning(f"Plot not found: {filename}")

    shap_csv = REPORT_DIR / "shap_feature_importance.csv"
    if shap_csv.exists():
        st.subheader("Top SHAP Features")
        shap_df = pd.read_csv(shap_csv)
        st.dataframe(shap_df.head(10), use_container_width=True)

# Sample prediction
elif page == "Sample Prediction":
    st.header("🧪 Dataset Record Prediction")

    csv_files = glob.glob(
        str(ROOT / "data" / "raw" / "**" / "*.csv"),
        recursive=True
    )

    if not csv_files:
        st.error("Dataset CSV not found inside data/raw.")
        st.stop()

    @st.cache_data
    def load_data(path):
        return pd.read_csv(path, low_memory=False)

    df = load_data(csv_files[0])

    if "label" not in df.columns:
        st.error("The dataset must contain the 'label' target column.")
        st.stop()

    if len(df) == 0:
        st.error("The dataset contains no records.")
        st.stop()

    row_number = st.number_input(
        "Choose a dataset row number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    row = df.iloc[int(row_number)]

    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Required model features are missing: {missing}")
        st.stop()

    # Only model features are passed to the classifier.
    input_df = pd.DataFrame(
        [{feature: row[feature] for feature in features}],
        columns=features
    )

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    positive_index = classes.index(1)
    probability_positive = float(probabilities[positive_index])

    st.write(f"**Model:** {model_name}")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("Model prediction: Label 1")
        else:
            st.success("Model prediction: Label 0")

    with col2:
        st.metric(
            "Model probability for label 1",
            f"{probability_positive:.2%}"
        )

    st.caption(
        "Confirm the dataset's label mapping before interpreting label 1 "
        "as phishing. This display uses the model's predicted probability, "
        "not a guarantee that the website is malicious."
    )

    if "URL" in df.columns:
        st.write("**Dataset URL:**", str(row["URL"]))

    st.write("**Dataset's recorded label:**", int(row["label"]))

    with st.expander("Inspect the input features"):
        st.dataframe(input_df, use_container_width=True)

    st.warning(
        "Demonstration limitation: this row may have been included in the "
        "training split. Its prediction is not an independent test result. "
        "This page does not fetch or scan the website."
    )

st.divider()
st.caption(
    "PhishGuard XAI | Advanced Machine Learning Lab | "
    "Research and educational demonstration"
)
