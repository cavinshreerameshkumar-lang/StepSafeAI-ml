"""
SolarShield ML Training Script
================================
Run locally once:
    python train.py --csv path/to/your_data.csv

Outputs:
    models/ensemble.joblib   — trained ensemble models + scaler

Then commit models/ensemble.joblib and deploy to Railway.
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score

np.random.seed(42)

# ── Column mapping from your CSV headers → internal names ──
COL_MAP = {
    'Daily Ingested Water (L)':                              'water_L',
    'Time of year (month)':                                  'month',
    'Cardiovascular disease history':                        'cvd',
    'Dehydration':                                           'dehydration',
    'Heat Index (HI)':                                       'HI',
    'Diastolic BP':                                          'diastolic',
    'Environmental temperature (C)':                        'env_temp',
    'Sickle Cell Trait (SCT)':                              'SCT',
    'Systolic BP':                                           'systolic',
    'Weight (kg)':                                           'weight',
    'Patient temperature':                                   'pt_temp',
    'Rectal temperature (deg C)':                           'rectal_temp',
    'Relative Humidity':                                     'humidity',
    'Exposure to sun':                                       'sun_exposure',
    'BMI':                                                   'BMI',
    'Exertional (1) vs classic (0)':                        'exertional',
    'Barometric Pressure':                                   'baro',
    'Heart / Pulse rate (b/min)':                           'HR',
    'Age':                                                   'age',
    'Sweating':                                              'sweating',
    'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)': 'skin_color',
    'Strenuous exercise':                                    'exercise',
    'Nationality':                                           'nationality',
    'Sex':                                                   'sex',
    'Hot/dry skin':                                          'hot_dry_skin',
    'Time of day':                                           'hour',
    'Heat stroke':                                           'heat_stroke',
}

FEATURES = [
    'water_L', 'month', 'cvd', 'HI', 'diastolic', 'env_temp', 'SCT',
    'systolic', 'weight', 'humidity', 'sun_exposure', 'BMI', 'exertional',
    'baro', 'HR', 'age', 'sweating', 'skin_color', 'exercise', 'sex',
    'hot_dry_skin', 'hour', 'dehydration',
]

# ─────────────────────────────────────────────
#  Synthetic data generator (used when no CSV)
# ─────────────────────────────────────────────
def gen_synthetic(n=4000):
    rows = []
    COLUMNS = FEATURES + ['heat_stroke', 'pt_temp', 'rectal_temp']
    for _ in range(n):
        profile = np.random.choice(['low','moderate','high','extreme'], p=[0.35,0.30,0.25,0.10])
        month   = np.random.randint(1, 13)
        summer  = 1 if month in [4,5,6,7,8,9] else 0
        hour    = np.random.uniform(6, 20)
        midday  = 1 if 10 <= hour <= 16 else 0
        age     = np.random.randint(8, 85)
        weight  = np.random.uniform(35, 130)
        height  = np.random.uniform(1.45, 1.95)
        BMI     = weight / (height ** 2)
        cvd     = int(np.random.random() < (0.25 if age >= 65 else 0.05))
        SCT     = int(np.random.random() < 0.08)
        sex     = np.random.randint(0, 2)
        base_temp = {'low':25,'moderate':32,'high':38,'extreme':42}[profile]
        env_temp  = np.clip(np.random.normal(base_temp, 3), 15, 50)
        humidity  = np.clip(np.random.beta(2,3)*(0.9-0.2*summer)+0.1, 0.05, 0.95)
        baro      = np.random.normal(29.97, 0.2)
        sun       = int(midday and summer and np.random.random() < 0.7)
        T_f  = env_temp * 9/5 + 32
        RH   = humidity * 100
        HI   = max(-42.379 + 2.049*T_f + 10.143*RH - 0.225*T_f*RH
                   - 0.00684*T_f**2 - 0.0548*RH**2
                   + 0.00122*T_f**2*RH + 0.000853*T_f*RH**2
                   - 0.00000199*T_f**2*RH**2, T_f)
        exertional = int(np.random.random() < 0.4)
        exercise   = int(np.random.random() < (0.6 if exertional else 0.2))
        water_L    = np.clip(np.random.normal(3.0 if profile=='low' else 1.8, 0.8), 0.2, 8)
        dehydration = int(water_L < 1.5 or (water_L < 2.0 and profile in ['high','extreme']))
        HR         = np.clip(np.random.normal(80+20*(HI>100)+15*exertional+10*dehydration, 12), 50, 200)
        systolic   = np.clip(np.random.normal(120+15*cvd+5*(age>=65)-10*(profile=='low'), 12), 60, 200)
        diastolic  = np.clip(np.random.normal(80+10*cvd+3*(age>=65), 8), 40, 130)
        pt_temp    = np.clip(np.random.normal(37+0.8*(HI>105)+0.5*exertional+0.3*dehydration, 0.4), 36, 43)
        rectal_temp= np.clip(pt_temp + np.random.normal(0.2, 0.15), 36, 44)
        sweating   = int(pt_temp > 37.5 and dehydration == 0)
        hot_dry    = int(pt_temp > 39.5 and dehydration == 1)
        skin_color = 1.0 if sweating else (0.5 if hot_dry else 0.0)
        hs_score   = (0.25*(rectal_temp>40.0) + 0.20*(HI>105) + 0.15*exertional
                     + 0.10*dehydration + 0.08*cvd + 0.07*(age>65 or age<12)
                     + 0.05*hot_dry + 0.05*SCT + 0.05*(water_L<1.0))
        heat_stroke = int((hs_score + np.random.normal(0, 0.06)) > 0.45)
        rows.append([water_L, month, cvd, HI, diastolic, env_temp, SCT, systolic,
                     weight, humidity, sun, BMI, exertional, baro, HR, age,
                     sweating, skin_color, exercise, sex, hot_dry, hour,
                     dehydration, heat_stroke, pt_temp, rectal_temp])
    return pd.DataFrame(rows, columns=COLUMNS)

# ─────────────────────────────────────────────
#  Derived targets
# ─────────────────────────────────────────────
def add_targets(df):
    df['dehydration_score'] = np.clip(
        (2.5 - df['water_L']) * 20
        + (df['env_temp'] - 28).clip(0) * 1.5
        + (df['humidity'] * 100 - 50).clip(0) * 0.4
        + df['dehydration'] * 25
        + df['hot_dry_skin'] * 15
        + (1 - df['sweating']) * 5,
        0, 100
    ).round(1)

    def heat_stress(row):
        if row['HI'] > 130: s = 3
        elif row['HI'] > 115: s = 2
        elif row['HI'] > 103: s = 1
        else: s = 0
        if row.get('cvd', 0) or row.get('SCT', 0): s = min(3, s+1)
        if row['age'] > 65 or row['age'] < 12: s = min(3, s+1)
        return s

    df['heat_stress_level'] = df.apply(heat_stress, axis=1)
    return df

# ─────────────────────────────────────────────
#  Train ensemble per target
# ─────────────────────────────────────────────
def train_classifier(X_tr, X_te, y_tr, y_te, label):
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    rf.fit(X_tr, y_tr); gb.fit(X_tr, y_tr); lr.fit(X_tr, y_tr)

    aucs = [roc_auc_score(y_te, m.predict_proba(X_te)[:,1]) for m in [rf, gb, lr]]
    total = sum(aucs)
    weights = [a/total for a in aucs]

    ensemble = weights[0]*rf.predict_proba(X_te)[:,1] \
             + weights[1]*gb.predict_proba(X_te)[:,1] \
             + weights[2]*lr.predict_proba(X_te)[:,1]
    print(f"  [{label}] RF={aucs[0]:.3f}  GB={aucs[1]:.3f}  LR={aucs[2]:.3f}  "
          f"Ensemble AUC={roc_auc_score(y_te, ensemble):.3f}")
    return rf, gb, lr, weights

def train_regressor(X_tr, X_te, y_tr, y_te, label):
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    lr = Ridge(alpha=1.0)
    rf.fit(X_tr, y_tr); gb.fit(X_tr, y_tr); lr.fit(X_tr, y_tr)

    maes = [mean_absolute_error(y_te, m.predict(X_te)) for m in [rf, gb, lr]]
    inv   = [1/m for m in maes]
    total = sum(inv)
    weights = [v/total for v in inv]

    ensemble = weights[0]*rf.predict(X_te) \
             + weights[1]*gb.predict(X_te) \
             + weights[2]*lr.predict(X_te)
    print(f"  [{label}] RF={maes[0]:.2f}  GB={maes[1]:.2f}  LR={maes[2]:.2f}  "
          f"Ensemble MAE={mean_absolute_error(y_te, ensemble):.2f}")
    return rf, gb, lr, weights

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main(csv_path=None):
    print("SolarShield ML Training")
    print("═" * 40)

    if csv_path and os.path.exists(csv_path):
        print(f"Loading CSV: {csv_path}")
        raw = pd.read_csv(csv_path)
        raw = raw.rename(columns=COL_MAP)
        # Drop columns not in FEATURES
        available = [c for c in FEATURES if c in raw.columns]
        missing   = [c for c in FEATURES if c not in raw.columns]
        if missing:
            print(f"  ⚠  Missing columns (will be zero-filled): {missing}")
            for c in missing:
                raw[c] = 0
        df = raw[FEATURES + ['heat_stroke']].copy()
        print(f"  Loaded {len(df)} rows from CSV")
    else:
        print("No CSV provided — using synthetic data only")
        df = gen_synthetic(4000)[FEATURES + ['heat_stroke']].copy()

    # Always add 2000 synthetic rows to balance / augment
    print("Generating 2000 synthetic augmentation rows…")
    syn = gen_synthetic(2000)[FEATURES + ['heat_stroke']].copy()
    df  = pd.concat([df, syn], ignore_index=True)
    df  = add_targets(df)
    print(f"Total training rows: {len(df)}")
    print(f"Heat stroke prevalence: {df['heat_stroke'].mean():.2%}")

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)
    split_kwargs = dict(test_size=0.2, random_state=42)

    print("\nTraining ensembles…")

    # Heat stroke
    Xtr, Xte, ytr, yte = train_test_split(X, df['heat_stroke'], **split_kwargs, stratify=df['heat_stroke'])
    hs_rf, hs_gb, hs_lr, hs_w = train_classifier(Xtr, Xte, ytr, yte, 'Heat Stroke')

    # Dehydration score
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X, df['dehydration_score'], **split_kwargs)
    dh_rf, dh_gb, dh_lr, dh_w = train_regressor(Xtr2, Xte2, ytr2, yte2, 'Dehydration Score')

    # Heat stress level (0-3)
    Xtr3, Xte3, ytr3, yte3 = train_test_split(X, df['heat_stress_level'], **split_kwargs)
    hsl_rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    hsl_gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    hsl_rf.fit(Xtr3, ytr3); hsl_gb.fit(Xtr3, ytr3)
    hsl_w = [0.55, 0.45]
    ens3 = np.round(hsl_w[0]*hsl_rf.predict(Xte3) + hsl_w[1]*hsl_gb.predict(Xte3))
    print(f"  [Heat Stress Level] Ensemble Accuracy={accuracy_score(yte3, ens3):.3f}")

    # ── Bundle and save ──
    bundle = {
        'features': FEATURES,
        'scaler': scaler,
        'heat_stroke':       {'rf': hs_rf,  'gb': hs_gb,  'lr': hs_lr,  'weights': hs_w},
        'dehydration_score': {'rf': dh_rf,  'gb': dh_gb,  'lr': dh_lr,  'weights': dh_w},
        'heat_stress_level': {'rf': hsl_rf, 'gb': hsl_gb, 'weights': hsl_w},
        'meta': {
            'n_samples':   len(df),
            'features':    FEATURES,
            'prevalence':  float(df['heat_stroke'].mean()),
        }
    }

    os.makedirs('models', exist_ok=True)
    joblib.dump(bundle, 'models/ensemble.joblib', compress=3)
    size = os.path.getsize('models/ensemble.joblib') / 1024
    print(f"\n✓ Saved models/ensemble.joblib ({size:.0f} KB)")
    print("  Copy this file to your solarsheild-ml/ folder and deploy.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to your heatstroke CSV file')
    args = parser.parse_args()
    main(args.csv)
