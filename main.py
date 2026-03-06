"""SolarShield ML API"""
import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

MODEL_PATH = os.getenv(
    'MODEL_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'ensemble.joblib')
)

BUNDLE = None

def get_bundle():
    global BUNDLE
    if BUNDLE is None:
        try:
            BUNDLE = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"✗ Model load failed: {e}")
    return BUNDLE

app = FastAPI(title="SolarShield ML API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FEATURES = [
    'water_L','month','cvd','HI','diastolic','env_temp','SCT',
    'systolic','weight','humidity','sun_exposure','BMI','exertional',
    'baro','HR','age','sweating','skin_color','exercise','sex',
    'hot_dry_skin','hour','dehydration',
]

COOLING_TIERS = {
    'high_income_urban': 0.70, 'middle_income_urban': 0.85,
    'low_income_urban': 1.00, 'slum_urban': 1.35,
    'rural_developed': 0.95, 'rural_developing': 1.20,
    'desert_remote': 1.30, 'coastal_humid': 1.10,
}

CITY_TO_TIER = {
    'chennai': 'middle_income_urban', 'mumbai': 'middle_income_urban',
    'delhi': 'middle_income_urban', 'bangalore': 'middle_income_urban',
    'hyderabad': 'middle_income_urban', 'kolkata': 'low_income_urban',
    'pune': 'middle_income_urban', 'ahmedabad': 'low_income_urban',
}


class PredictRequest(BaseModel):
    env_temp: float = 35.0
    humidity: float = 0.60
    heat_index: Optional[float] = None
    age: float = 30.0
    weight: float = 70.0
    bmi: Optional[float] = None
    sex: int = 1
    cvd: int = 0
    sct: int = 0
    water_l: float = 2.0
    heart_rate: float = 80.0
    systolic: float = 120.0
    diastolic: float = 80.0
    exertional: int = 0
    exercise: int = 0
    sweating: int = 1
    hot_dry_skin: int = 0
    skin_color: float = 1.0
    dehydration: int = 0
    city: Optional[str] = None
    cooling_tier: Optional[str] = None
    month: int = 6
    hour: float = 12.0
    sun_exposure: int = 1
    baro: float = 29.97


class FeatureContrib(BaseModel):
    feature: str
    value: float
    importance: float
    direction: str


class PredictResponse(BaseModel):
    heat_stroke_probability: float
    adjusted_heat_stroke: float
    dehydration_score: float
    heat_stress_level: int
    heat_stress_label: str
    cooling_tier: str
    cooling_factor: float
    top_features: List[FeatureContrib]
    recommendations: List[str]


@app.get("/health")
def health():
    bundle = get_bundle()
    return {"status": "ok", "model_loaded": bundle is not None}


@app.get("/features")
def features():
    return {"features": FEATURES}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    bundle = get_bundle()
    if not bundle:
        raise HTTPException(503, f"Model not loaded from {MODEL_PATH}")

    bmi = req.bmi if req.bmi else round(req.weight / (1.70 ** 2), 1)
    T_f = req.env_temp * 9 / 5 + 32
    RH  = req.humidity * 100
    HI  = req.heat_index if req.heat_index else max(
        -42.379 + 2.049*T_f + 10.143*RH - 0.225*T_f*RH
        - 0.00684*T_f**2 - 0.0548*RH**2 + 0.00122*T_f**2*RH
        + 0.000853*T_f*RH**2 - 0.00000199*T_f**2*RH**2, T_f)

    feat_vals = [
        req.water_l, req.month, req.cvd, HI, req.diastolic,
        req.env_temp, req.sct, req.systolic, req.weight,
        req.humidity, req.sun_exposure, bmi, req.exertional,
        req.baro, req.heart_rate, req.age, req.sweating,
        req.skin_color, req.exercise, req.sex, req.hot_dry_skin,
        req.hour, req.dehydration,
    ]

    X_raw    = np.array(feat_vals).reshape(1, -1)
    X_scaled = bundle['scaler'].transform(X_raw)

    hs       = bundle['heat_stroke']
    hs_probs = [m.predict_proba(X_raw)[0][1] for m in [hs['rf'], hs['gb'], hs['lr']]]
    hs_prob  = float(sum(w * p for w, p in zip(hs['weights'], hs_probs)))

    dh       = bundle['dehydration_score']
    dh_preds = [m.predict(X_raw)[0] for m in [dh['rf'], dh['gb'], dh['lr']]]
    dh_score = float(np.clip(sum(w * p for w, p in zip(dh['weights'], dh_preds)), 0, 100))

    hsl     = bundle['heat_stress_level']
    hsl_val = int(np.clip(
        round(hsl['weights'][0] * hsl['rf'].predict(X_raw)[0]
              + hsl['weights'][1] * hsl['gb'].predict(X_raw)[0]), 0, 3))
    if req.cvd or req.sct:           hsl_val = min(3, hsl_val + 1)
    if req.age > 65 or req.age < 12: hsl_val = min(3, hsl_val + 1)

    tier     = req.cooling_tier or CITY_TO_TIER.get((req.city or '').lower(), 'low_income_urban')
    factor   = COOLING_TIERS.get(tier, 1.0)
    adjusted = float(np.clip(hs_prob * factor, 0, 1))

    importances = hsl['rf'].feature_importances_
    top_feats   = sorted(
        [{'feature': FEATURES[i], 'value': float(feat_vals[i]),
          'importance': float(importances[i]),
          'direction': 'increases' if X_scaled[0][i] > 0 else 'decreases'}
         for i in range(len(FEATURES))],
        key=lambda x: x['importance'], reverse=True)[:6]

    recs = []
    if hs_prob > 0.7:        recs.append("🚨 Seek immediate shade or air-conditioned space")
    if hs_prob > 0.5:        recs.append("🌡 Monitor body temperature every 15 minutes")
    if dh_score > 60:        recs.append("💧 Drink 250ml of water immediately")
    if req.water_l < 2:      recs.append("💧 Increase water intake — aim for 3–4L today")
    if req.heart_rate > 110: recs.append("❤ Elevated heart rate — rest and cool down")
    if req.exertional:       recs.append("⚠ Reduce exertion intensity in current conditions")
    if hsl_val >= 2:         recs.append("🏥 Consider emergency services if symptoms worsen")
    if req.hot_dry_skin:     recs.append("🚿 Apply cool water to skin immediately")
    if not recs:             recs.append("✅ Conditions safe — stay hydrated and monitor")

    return PredictResponse(
        heat_stroke_probability=round(hs_prob * 100, 1),
        adjusted_heat_stroke=round(adjusted * 100, 1),
        dehydration_score=round(dh_score, 1),
        heat_stress_level=hsl_val,
        heat_stress_label=['Low','Moderate','High','Severe'][hsl_val],
        cooling_tier=tier, cooling_factor=factor,
        top_features=[FeatureContrib(**f) for f in top_feats],
        recommendations=recs,
    )
