# churn_outputs_from_test.py

import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("📊 Churn Prediction Results (from Test Data)")

# --- Upload Files ---
st.sidebar.header("Upload Model and Test Data")
model_file = st.sidebar.file_uploader("Trained Model (.pkl)", type="pkl")
test_features_file = st.sidebar.file_uploader("X_test.csv", type="csv")
test_labels_file = st.sidebar.file_uploader("y_test.csv", type="csv")

if model_file and test_features_file and test_labels_file:
    model = pickle.load(model_file)
    X_test = pd.read_csv(test_features_file)
    y_test = pd.read_csv(test_labels_file)

    st.success("Files loaded successfully!")

    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Combine for display
    result_df = X_test.copy()
    result_df['Actual'] = y_test
    result_df['Predicted'] = preds
    result_df['Churn_Probability'] = probs

    st.subheader("🔍 Sample Predictions")
    st.dataframe(result_df.head())

    # Metrics
    st.subheader("📈 Model Performance")
    report = classification_report(y_test, preds, output_dict=True)
    st.write("**Classification Report:**")
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt.gcf())
    plt.clf()

    roc = roc_auc_score(y_test, probs)
    st.metric("ROC AUC", f"{roc:.3f}")

    # Download option
    st.download_button("📥 Download Predictions", result_df.to_csv(index=False), "churn_predictions.csv")

else:
    st.info("Please upload the model and test datasets to begin.")
