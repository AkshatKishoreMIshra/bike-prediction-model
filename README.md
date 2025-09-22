# Bike Rental Demand Prediction

This repo contains an interactive Streamlit demo that predicts hourly bike demand using historical features such as season, hour, weather, temperature, humidity, and other contextual features.

## Contents

- `app.py` - Streamlit app with improved UI and model-loading logic.
- `models/bike_model.pkl` - Trained model (not included). Place your trained scikit-learn Pipeline here.
- `requirements.txt` - Required Python packages.

## How to run

1. Create a virtual environment and activate it (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Put your trained model pipeline at `models/bike_model.pkl` (should accept the input columns used in the app).

3. Run the app:

```powershell
streamlit run app.py
```

## What to showcase on your resume

- Clean UI built with Streamlit and custom CSS.
- Cached model loading and structured code (production-friendly patterns).
- Interactive inputs, visualizations (matplotlib), and actionable insights.
- Describe metrics used during training (RMSE, R²) and model selection process in a short bullet list.

Include a link to the deployed demo (if deployed) and note the algorithm you used (RandomForest / XGBoost / Linear Regression) and evaluation scores.
