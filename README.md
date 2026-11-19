# Student Performance Classification — Deployment Package

This project follows the supplied presentation: it compares a Decision Tree and a Random Forest classifier using Attendance, Hours_Studied and Previous_Scores.

## Dataset
Run `python download_dataset.py` to download `data/StudentPerformanceFactors.csv`.
Dataset mirror: https://github.com/grugg1233/StudentPerformanceFactors
The CSV has 6,607 records and 20 columns. It includes `Exam_Score` as the outcome.

## Class labels
The project derives a three-class target from `Exam_Score`:
- Low: below 65
- Medium: 65 through 70 inclusive
- High: above 70

These are project-defined thresholds, not official academic grading bands. The dataset is public and should be described as a sample dataset; do not treat predictions as definitive judgments about real students.

## Run locally (Windows / VS Code)
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python download_dataset.py
python train_model.py
python app.py
```
Open http://127.0.0.1:5000

## Deploy publicly on Render
1. Create a GitHub repository and upload the contents of this folder (including the `data` folder and downloaded CSV).
2. In Render, choose **New + → Web Service**, connect the GitHub repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy. Render will provide a public `onrender.com` URL.

`render.yaml` is also included for a Blueprint-style deployment. For reliable deployment, commit the CSV to the repository so the service can train on startup. Do not commit the `venv` folder.

## Endpoints
- `/` website
- `/health` service health check
- `/api/metrics` model evaluation metrics
- `/api/predict` POST JSON with `attendance`, `hours_studied`, `previous_scores`, and optional `model`

## Notes
- Models use an 80/20 stratified train/test split and a fixed random seed.
- Missing feature values are imputed with the median inside each model pipeline.
- Accuracy is calculated on the held-out test set; do not claim a result until the project has trained successfully.
- This is an educational demonstration, not a validated student assessment system.
