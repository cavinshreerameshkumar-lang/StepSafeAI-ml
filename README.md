# SolarShield ML API

FastAPI ensemble model for heat stroke risk prediction.

## Local setup

```bash
pip install -r requirements.txt

# Train with your CSV (run once, then commit the model)
python train.py --csv "path/to/your_data.csv"

# Start the API locally
uvicorn app.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

## Deploy to Railway

1. Train the model locally — this produces `models/ensemble.joblib`
2. Create a new GitHub repo and push this entire folder:
   ```bash
   git init
   git add .
   git commit -m "initial"
   gh repo create solarsheild-ml --public --push
   ```
3. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
4. Select your `solarsheild-ml` repo → Railway auto-detects Python and deploys
5. Copy the public URL (e.g. `https://solarsheild-ml.up.railway.app`)

## Add the URL to your React app

In your React project `.env`:
```
VITE_ML_API_URL=https://your-service.up.railway.app
```

## API endpoints

| Method | Path       | Description                    |
|--------|------------|--------------------------------|
| GET    | /health    | Health check + model info      |
| GET    | /features  | List of expected input fields  |
| POST   | /predict   | Run ensemble prediction        |
| GET    | /docs      | Interactive Swagger UI         |

## POST /predict — example

```json
{
  "env_temp": 39,
  "humidity": 0.7,
  "uv_index": 8.5,
  "age": 38,
  "weight": 72,
  "bmi": 24,
  "sex": 0,
  "cvd": 0,
  "sct": 0,
  "water_l": 1.5,
  "heart_rate": 95,
  "systolic": 125,
  "diastolic": 82,
  "exertional": 1,
  "exercise": 1,
  "sweating": 1,
  "hot_dry_skin": 0,
  "skin_color": 1.0,
  "city": "Chennai"
}
```

## Response

```json
{
  "heat_stroke_probability": 34.2,
  "adjusted_heat_stroke": 29.1,
  "dehydration_score": 61.4,
  "heat_stress_level": 2,
  "heat_stress_label": "High",
  "cooling_tier": "middle_income_urban",
  "cooling_factor": 0.85,
  "top_features": [...],
  "recommendations": [...]
}
```
