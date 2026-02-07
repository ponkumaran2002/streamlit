import pandas as pd
import joblib
from preprocessing import preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("data/student_data.csv")

X, y, scaler = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
models = {
    "logistic": LogisticRegression(max_iter=1000,class_weight="balanced"),
    "decision_tree": DecisionTreeClassifier(class_weight="balanced"),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200,class_weight="balanced"),
    "xgboost": XGBClassifier(eval_metric="logloss",  scale_pos_weight=scale_pos_weight)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")

joblib.dump(scaler, "model/scaler.pkl")
