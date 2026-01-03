
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def validate_data(df):
    """
    Barrasi salamat dade ghabl az vared shodan be pipeline
    """
    required_columns = ["age", "income", "score"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column lazem vojood nadarad: {col}")

    if df.isnull().any().any():
        raise ValueError("Dade khali (None/NaN) dar dataset vojood darad")

    if not df["age"].between(0, 120).all():
        raise ValueError("Meghdar age kharej az bazeh mojaz ast")

    if (df["income"] < 0).any():
        raise ValueError("Income nemitavanad manfi bashad")

    if not df["score"].between(0, 1).all():
        raise ValueError("Score bayad beyn 0 ta 1 bashad")

    return df


def build_pipeline():
    """
    Sakhte pipeline amoozesh
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    return pipeline


def main():
    data = {
        "age": [25, -5, 40, 30],
        "income": [50000, 40000, None, 60000],
        "score": [0.8, 0.6, 0.9, 0.7],
        "label": [1, 0, 1, 0]
    }

    df = pd.DataFrame(data)

    print("Shoro barrasi salamat dade...")

    try:
        df = validate_data(df)
    except ValueError as e:
        print(f"Khata dar dade: {e}")
        print("Dade nasalem vared pipeline nashod.")
        return

    X = df[["age", "income", "score"]]
    y = df["label"]

    pipeline = build_pipeline()

    print("Train model ba dade salem shoro shod...")
    pipeline.fit(X, y)
    print("Train model ba movafaghiat anjam shod.")


if __name__ == "__main__":
    main()
