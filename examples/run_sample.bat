@echo off
setlocal

echo [1/1] Scoring sample landmark file...
python scripts\predict_from_csv.py --input data\sample_landmark_features.csv --output outputs\predictions.csv --summary-json outputs\summary.json
if errorlevel 1 exit /b 1

echo Completed. Outputs written to outputs\
