
import os
import json
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification


# -------------------------------
# Configuration
# -------------------------------

DATA_PATH = "sample_input.csv"
REPORT_OUTPUT_PATH = "final_report.json"
MIN_ACCEPTABLE_ACCURACY = 0.90
RANDOM_STATE = 42


# -------------------------------
# Sample Data Generator
# -------------------------------

def generate_sample_data(path):
    """
    Ijad dataset sample baraye test
    """

    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=3.0,
        random_state=RANDOM_STATE
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["label"] = y

    df.to_csv(path, index=False)


# -------------------------------
# Data Loading
# -------------------------------

def load_data(path):
    """
    Load data, agar file nabood misazad
    """

    if not os.path.exists(path):
        generate_sample_data(path)

    return pd.read_csv(path)


# -------------------------------
# Model Pipeline
# -------------------------------

def build_model():
    """
    Sakhte ML pipeline
    """

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])


# -------------------------------
# Training & Prediction
# -------------------------------

def train_and_predict(data):
    """
    Train va predict rooye kol data
    """

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = build_model()
    model.fit(X, y)

    y_pred = model.predict(X)

    return y.values, y_pred


# -------------------------------
# Evaluation
# -------------------------------

def evaluate(y_true, y_pred):
    """
    Mohasebe accuracy
    """

    return accuracy_score(y_true, y_pred)


# -------------------------------
# Report Generation
# -------------------------------

def generate_report(y_true, y_pred, output_path):
    """
    Tolid report va zakhire dar file
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)


# -------------------------------
# End-to-End Run
# -------------------------------

def run_full_pipeline():
    """
    Ejraye kamel jaryan bedoon crash
    """

    data = load_data(DATA_PATH)

    y_true, y_pred = train_and_predict(data)

    accuracy = evaluate(y_true, y_pred)
    print(f"Model Accuracy: {accuracy:.2%}")

    if accuracy >= MIN_ACCEPTABLE_ACCURACY:
        print("Accuracy be had mojaz resid ")
    else:
        print(" Accuracy kamtar az had mojaz ast, vali program continue mishavad")

    generate_report(
        y_true,
        y_pred,
        REPORT_OUTPUT_PATH
    )

    print("Report ba movafaghiat zakhire shod.")


# -------------------------------
# Manual Run
# -------------------------------

if __name__ == "__main__":
    run_full_pipeline()
    print("End-to-End process tamam shod ")
