from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from train_model import train_and_save, DATA_PATH, MODEL_DIR

app = Flask(__name__)
BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"

def get_bundle():
    if not BUNDLE_PATH.exists():
        train_and_save()
    return joblib.load(BUNDLE_PATH)

@app.route("/")
def home():
    bundle = get_bundle()
    metrics = bundle["metrics"]
    return render_template("index.html", metrics=metrics, labels=bundle["labels"])

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        attendance = float(payload["attendance"])
        hours = float(payload["hours_studied"])
        previous = float(payload["previous_scores"])
        if not (0 <= attendance <= 100):
            return jsonify(error="Attendance must be between 0 and 100."), 400
        if not (0 <= hours <= 24):
            return jsonify(error="Study hours must be between 0 and 24."), 400
        if not (0 <= previous <= 100):
            return jsonify(error="Previous score must be between 0 and 100."), 400

        bundle = get_bundle()
        row = pd.DataFrame([{
            "Attendance": attendance,
            "Hours_Studied": hours,
            "Previous_Scores": previous
        }])
        model_name = payload.get("model", "Random Forest")
        if model_name not in bundle["models"]:
            model_name = "Random Forest"
        model = bundle["models"][model_name]
        prediction = str(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        confidence = float(max(probabilities) * 100)
        return jsonify(
            prediction=prediction,
            confidence=round(confidence, 1),
            model=model_name,
            probabilities={str(k): round(float(v) * 100, 1)
                           for k, v in zip(model.classes_, probabilities)}
        )
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 500
    except Exception as exc:
        return jsonify(error=f"Prediction failed: {exc}"), 500

@app.route("/api/metrics")
def api_metrics():
    bundle = get_bundle()
    return jsonify(bundle["metrics"])

@app.route("/health")
def health():
    return jsonify(status="ok", message="Student Performance Classifier is running")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
