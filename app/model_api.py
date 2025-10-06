# model_api.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib, numpy as np
import pandas as pd
from pathlib import Path

# ---------- model (unchanged) ----------
MODEL_PATH = "aqi_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="CleanAir Model API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(
    co: float = Query(...),
    o3: float = Query(...),
    no2: float = Query(...),
    pm25: float = Query(...),
    lat: float = Query(...),
    lng: float = Query(...),
):
    X = np.array([[co, o3, no2, pm25, lat, lng]], dtype=float)
    pred = model.predict(X)[0]
    return {"category": str(pred)}

# ---------- NEW: nearest city from your dataset ----------
# Robust path to CSV:  .../app/../data/AQI_and_LatLong.csv
BASE = Path(__file__).resolve().parent
DATA_CSV = (BASE.parent / "data" / "AQI_and_LatLong.csv")
try:
    _df = pd.read_csv(DATA_CSV).dropna(subset=["lat", "lng"]).copy()
except Exception as e:
    _df = pd.DataFrame()
    print(f"[nearest-city] failed to load dataset: {e}")

def _haversine_vec(lat: float, lng: float, arr_lat: np.ndarray, arr_lng: np.ndarray) -> np.ndarray:
    R = 6371.0
    phi1 = np.radians(lat)
    phi2 = np.radians(arr_lat)
    dphi = phi2 - phi1
    dlmb = np.radians(arr_lng - lng)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

@app.get("/nearest-city")
def nearest_city(lat: float = Query(...), lng: float = Query(...)):
    if _df.empty:
        raise HTTPException(status_code=500, detail="Dataset not available")
    dists = _haversine_vec(lat, lng, _df["lat"].to_numpy(), _df["lng"].to_numpy())
    idx = int(np.argmin(dists))
    row = _df.iloc[idx]
    return {
        "city": str(row.get("City", "")),
        "country": str(row.get("Country", "")),
        "aqi_value": int(row.get("AQI Value", 0)),
        "aqi_category": str(row.get("AQI Category", "")),
        "distance_km": float(dists[idx]),
    }
