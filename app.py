import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.title("Student Mental Health Classification")

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Map model names to their corresponding file names
model_map = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

uploaded_file = st.file_uploader("Upload Test CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Load the model, scaler, and label encoders
    model = joblib.load(f"model/{model_map[model_choice]}")
    scaler = joblib.load("model/scaler.pkl")
    le_gender = joblib.load("model/le_gender.pkl")
    le_dept = joblib.load("model/le_dept.pkl")

    # Preprocess data (encoding 'Gender' and 'Department' and scaling)
    df["Gender"] = le_gender.transform(df["Gender"])
    df["Department"] = le_dept.transform(df["Department"])

    # Drop non-predictive columns like 'Student_ID'
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

    # Separate features and target
    X = df.drop("Depression", axis=1)
    y = df["Depression"].astype(int)

    # Scale the features using the pre-fitted scaler
    X_scaled = scaler.transform(X)

    # Make predictions using the selected model
    y_pred = model.predict(X_scaled)

    # Display the classification report
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    # Create and display confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
