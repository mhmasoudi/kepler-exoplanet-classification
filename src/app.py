
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "kepler_xgboost.joblib"

FEATURES = [
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

model = joblib.load(MODEL_PATH)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Kepler Exoplanet Classification API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object."}), 400

    missing_features = [feature for feature in FEATURES if feature not in data]

    if missing_features:
        return jsonify({
            "error": "Missing features",
            "missing_features": missing_features
        }), 400

    try:
        observation = pd.DataFrame([data], columns=FEATURES)
        prediction = int(model.predict(observation)[0])
        probability = float(model.predict_proba(observation)[0, 1])

        return jsonify({
            "prediction": prediction,
            "classification": (
                "Confirmed" if prediction == 1 else "False Positive"
            ),
            "probability_confirmed": probability
        })

    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
