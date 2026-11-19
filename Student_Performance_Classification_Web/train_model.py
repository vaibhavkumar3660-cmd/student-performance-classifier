from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "StudentPerformanceFactors.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURES = ["Attendance", "Hours_Studied", "Previous_Scores"]

def score_to_class(score):
    # Transparent project-defined categories based on Exam_Score.
    if score < 65:
        return "Low"
    elif score <= 70:
        return "Medium"
    return "High"

def train_and_save():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python download_dataset.py "
            "or place StudentPerformanceFactors.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + ["Exam_Score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Exam_Score"])
    df["Performance_Class"] = df["Exam_Score"].apply(score_to_class)
    X = df[FEATURES]
    y = df["Performance_Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=5, min_samples_split=10, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_split=5,
                random_state=42, class_weight="balanced"
            ))
        ])
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, preds)) * 100, 2),
            "confusion_matrix": confusion_matrix(
                y_test, preds, labels=["Low", "Medium", "High"]
            ).tolist(),
            "classification_report": classification_report(
                y_test, preds, labels=["Low", "Medium", "High"],
                output_dict=True, zero_division=0
            )
        }

    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "models": models,
        "metrics": metrics,
        "labels": ["Low", "Medium", "High"],
        "features": FEATURES,
        "rows_used": int(len(df)),
        "class_thresholds": "Low: Exam_Score < 65; Medium: 65–70; High: > 70"
    }
    joblib.dump(bundle, MODEL_DIR / "model_bundle.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "models"}, f, indent=2)
    print("Training complete!")
    print(f"Records used: {len(df)}")
    for name, result in metrics.items():
        print(f"{name} accuracy: {result['accuracy']}%")
    return bundle

if __name__ == "__main__":
    train_and_save()
