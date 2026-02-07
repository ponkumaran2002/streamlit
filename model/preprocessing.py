import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(df, save=True):
    # Drop non-numeric or irrelevant columns
    df = df.drop(columns=["Student_ID"])

    # Initialize label encoders
    le_gender = LabelEncoder()
    le_dept = LabelEncoder()

    # Encode 'Gender' and 'Department'
    df["Gender"] = le_gender.fit_transform(df["Gender"])
    df["Department"] = le_dept.fit_transform(df["Department"])
    df["Depression"] = df["Depression"].astype(int)

    # Separate features (X) and target (y)
    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    # Initialize and apply the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler and label encoders if needed
    if save:
        joblib.dump(le_gender, 'model/le_gender.pkl')
        joblib.dump(le_dept, 'model/le_dept.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')

    return X_scaled, y, scaler


# def preprocess_data(df):
#     df = df.drop(columns=["Student_ID"])

#     le_gender = LabelEncoder()
#     le_dept = LabelEncoder()

#     df["Gender"] = le_gender.fit_transform(df["Gender"])
#     df["Department"] = le_dept.fit_transform(df["Department"])
#     df["Depression"] = df["Depression"].astype(int)

#     X = df.drop("Depression", axis=1)
#     y = df["Depression"]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, y, scaler
