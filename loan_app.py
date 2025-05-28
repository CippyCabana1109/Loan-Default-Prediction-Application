import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Load the trained model
model = load("loan_default_model.joblib")

st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title(" Loan Default Prediction App")

st.markdown(
    """
    Upload a CSV file containing loan applicant data, and this app will:
    - Predict likelihood of default using a trained machine learning model.
    - Visualize feature importance using SHAP values for interpretability.
    """
)

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Raw Uploaded Data")
    st.dataframe(input_df.head())

    # Drop target column if present
    if 'Default' in input_df.columns:
        input_df.drop('Default', axis=1, inplace=True)

    # Encode categorical features
    cat_cols = input_df.select_dtypes(include='object').columns
    for col in cat_cols:
        input_df[col] = pd.factorize(input_df[col])[0]

    # Fill missing values
    input_df.fillna(input_df.median(numeric_only=True), inplace=True)

    # Make predictions
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]

    # Add predictions to results
    results = input_df.copy()
    results['Predicted Default (0=No, 1=Yes)'] = preds
    results['Probability of Default'] = probs

    st.subheader(" Prediction Results")
    st.dataframe(results)

    # Downloadable CSV
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions", csv, "loan_predictions.csv", "text/csv")

    # SHAP Explanation
    st.subheader("📉 Feature Importance (SHAP Summary Plot)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    st.markdown("This plot shows which features impact the model the most.")
    shap.summary_plot(shap_values[1], input_df, show=False)
    st.pyplot(bbox_inches="tight")
else:
    st.info("Please upload a CSV file to begin.")
