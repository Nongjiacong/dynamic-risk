@echo off
setlocal

echo [1/3] Regenerating public synthetic files...
python scripts\generate_sample_data.py
if errorlevel 1 exit /b 1

echo [2/3] Training bundled model...
python scripts\train_bundled_model.py
if errorlevel 1 exit /b 1

echo [3/3] Scoring sample landmark file...
python scripts\predict_from_csv.py --input data\sample_landmark_features.csv --output outputs\predictions.csv --summary-json outputs\summary.json
if errorlevel 1 exit /b 1

echo Completed. Outputs written to outputs\
