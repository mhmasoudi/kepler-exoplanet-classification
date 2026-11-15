
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "kepler_xgboost.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, data):
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    results = pd.DataFrame({
        "prediction": predictions,
        "probability_confirmed": probabilities
    })

    return results


def main():
    model = load_model()

    # Example observations from the dataset
    data_path = PROJECT_DIR / "data" / "kepler_binary.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["target", "kepid"])
    sample = X.head(5)

    results = predict(model, sample)

    print(results)


if __name__ == "__main__":
    main()
