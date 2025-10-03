# app/app.py
import math
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from utils import COLORS, health_tip

# -------------------- page config --------------------
st.set_page_config(page_title="CleanAir Compass", layout="wide")

# -------------------- styles --------------------
st.markdown(
    """
<style>
:root{
  --bg:#f6f7fb; --panel:#ffffff; --muted:#6b7280; --text:#111827; --border:#e5e7eb;
  --shadow:0 10px 28px rgba(17,24,39,.08); --radius:16px;
}
.result-card{
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  box-shadow:var(--shadow); padding:18px 18px; margin-top:14px;
}
.badge{
  display:inline-flex; gap:10px; align-items:center; font-weight:700; color:#fff;
  padding:8px 14px; border-radius:999px; font-size:0.95rem;
}
.sub{ font-size:.9rem; color:var(--muted); margin-top:6px; }
.kv{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.kv .chip{
  background:#f3f4f6; color:#111827; border:1px solid var(--border);
  border-radius:999px; padding:6px 12px; font-weight:600;
}
.info{
  background:#eef2ff; color:#1e3a8a; border:1px solid #c7d2fe;
  border-radius:12px; padding:12px 14px; margin:12px 0 2px;
}

/* nearest city card */
.city-card{
  background:#fff; border-radius:14px;
  padding:16px 18px; margin-top:10px;
  box-shadow:0 6px 20px rgba(0,0,0,.06);
}
.city-row{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
.pill{
  padding:6px 10px; border-radius:999px; font-weight:700; border:1px solid;
  background:rgba(0,0,0,.04);
}

/* threshold panel (progress to next category) */
.threshold{
  background:#fafafa; border:1px solid var(--border); border-radius:12px;
  padding:12px 14px; margin:10px 0; display:flex; gap:12px; align-items:center;
  box-shadow:0 6px 16px rgba(0,0,0,.04);
}
.th-icon{
  width:32px; height:32px; display:grid; place-items:center;
  border-radius:10px; background:#f1f5f9; color:#334155; font-size:18px;
}
.th-body{ flex:1; }
.mono{ font-variant-numeric: tabular-nums; }
.bar{
  height:8px; border-radius:999px; background:#e5e7eb; overflow:hidden; margin-top:8px;
}
.fill{ height:100%; background:var(--accent,#6366f1); }
.small-muted{ font-size:12px; color:var(--muted); margin-top:4px; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- helpers --------------------
AQI_RANGE = {
    "Good": (0, 50),
    "Moderate": (51, 100),
    "Unhealthy for Sensitive Groups": (101, 150),
    "Unhealthy": (151, 200),
    "Very Unhealthy": (201, 300),
    "Hazardous": (301, 500),
}

# PM2.5 cutoffs for the *start* of the next-worse category
NEXT_WORSE_PM25_START = {
    "Good": ("Moderate", 51),
    "Moderate": ("Unhealthy for Sensitive Groups", 101),
    "Unhealthy for Sensitive Groups": ("Unhealthy", 151),
    "Unhealthy": ("Very Unhealthy", 201),
    "Very Unhealthy": ("Hazardous", 301),
}

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two (lat,lon) in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def display_nearest_city(city_name: str, country: str, aqi_value: int, category: str, distance_km: float):
    """Pretty card for nearest city with category-colored border and pills."""
    color = COLORS.get(category, "#64748b")
    pill_bg = f"{color}20"  # translucent

    st.markdown(
        f"""
        <div class="city-card" style="border:2px solid {color}">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="font-size:20px">📍</div>
            <div style="font-weight:800; font-size:16px; color:#111827;">Nearest city in dataset</div>
          </div>
          <div class="city-row">
            <div style="font-size:16px;"><b>{city_name}</b>, {country}</div>
            <div class="pill" style="background:{pill_bg}; color:#111827; border-color:{color};">AQI {aqi_value}</div>
            <div class="pill" style="background:{pill_bg}; color:#111827; border-color:{color};">{category}</div>
            <div style="color:#6b7280; font-size:14px;">• Distance ≈ <b>{distance_km:.1f} km</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- cached loaders --------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/AQI_and_LatLong.csv")

@st.cache_resource
def load_model():
    return joblib.load("app/aqi_model.joblib")

df = load_data()
model = load_model()

# -------------------- layout --------------------
st.title("CleanAir Compass")
st.caption("Explore global air quality and predict AQI categories from pollutant levels.")

with st.sidebar:
    st.header("Filters")
    countries = sorted([c for c in df["Country"].dropna().unique()])
    sel_countries = st.multiselect("Country", countries)
    subset = df[df["Country"].isin(sel_countries)] if sel_countries else df

    st.divider()
    st.header("What-if Prediction")
    co    = st.number_input("CO AQI Value",    0.0, 500.0, 20.0, 1.0)
    ozone = st.number_input("Ozone AQI Value", 0.0, 500.0, 30.0, 1.0)
    no2   = st.number_input("NO2 AQI Value",   0.0, 500.0, 25.0, 1.0)
    pm25  = st.number_input("PM2.5 AQI Value", 0.0, 500.0, 35.0, 1.0)
    lat   = st.number_input("Latitude",  -90.0,  90.0,  float(subset["lat"].mean()))
    lng   = st.number_input("Longitude", -180.0, 180.0, float(subset["lng"].mean()))
    do_predict = st.button("Predict AQI Category", width="stretch")

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("City-level AQI Map")
    m = folium.Map(location=[subset["lat"].mean(), subset["lng"].mean()], zoom_start=2)
    for _, r in subset.iterrows():
        color = COLORS.get(r["AQI Category"], "#34495E")
        folium.CircleMarker(
            location=[r["lat"], r["lng"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=f"{r['City']} | {r['AQI Category']} (AQI {r['AQI Value']})",
        ).add_to(m)
    # Let Streamlit size the component to the column width
    st_folium(m, width=None, height=540)

with col2:
    st.subheader("Data (filtered)")
    st.dataframe(
        subset[
            [
                "Country",
                "City",
                "AQI Value",
                "AQI Category",
                "CO AQI Value",
                "Ozone AQI Value",
                "NO2 AQI Value",
                "PM2.5 AQI Value",
            ]
        ].head(500),
        width="stretch",
        height=540,
    )

# -------------------- prediction output (no probability) --------------------
if do_predict:
    X = np.array([[co, ozone, no2, pm25, lat, lng]], dtype=float)
    pred = model.predict(X)[0]

    # Colors / ranges
    badge_color = COLORS.get(pred, "#374151")
    lo, hi = AQI_RANGE.get(pred, (None, None))

    # Result card – header + chips
    st.markdown(
        f"""
        <div class="result-card">
          <div class="badge" style="background:{badge_color}">✅ Predicted: {pred}</div>
          <div class="sub">Model prediction for the inputs you entered.</div>

          <div class="kv">
            <span class="chip">Typical AQI range: <b>{lo if lo is not None else "?"}–{hi if hi is not None else "?"}</b></span>
            <span class="chip">CO: <b>{co:.0f}</b></span>
            <span class="chip">Ozone: <b>{ozone:.0f}</b></span>
            <span class="chip">NO₂: <b>{no2:.0f}</b></span>
            <span class="chip">PM2.5: <b>{pm25:.0f}</b></span>
            <span class="chip">Lat/Lng: <b>{lat:.2f}, {lng:.2f}</b></span>
          </div>
        """,
        unsafe_allow_html=True,
    )

    # Health advisory
    st.markdown(f"""<div class="info">{health_tip(pred)}</div>""", unsafe_allow_html=True)

    # Threshold panel (progress to next-worse category, based on PM2.5)
    if pred in NEXT_WORSE_PM25_START:
        nxt_cat, nxt_start = NEXT_WORSE_PM25_START[pred]
        remaining = max(0.0, nxt_start - pm25)

        # progress within current band
        lo_current = AQI_RANGE.get(pred, (0, 0))[0]
        denom = max(1, nxt_start - lo_current)
        pct = np.clip((pm25 - lo_current) / denom * 100, 0, 100)

        st.markdown(
            f"""
            <div class="threshold" style="--accent:{badge_color}">
              <div class="th-icon">🪜</div>
              <div class="th-body">
                <div>
                  You're <b class="mono">{remaining:.0f}</b> PM2.5 away from tipping into
                  <b>{nxt_cat}</b>.
                </div>
                <div class="bar"><div class="fill" style="width:{pct:.0f}%"></div></div>
                <div class="small-muted">Within current band (<b>{pred}</b>): {lo_current} → {nxt_start} · Position: {pct:.0f}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Nearest city (pretty card)
    try:
        _df = df.dropna(subset=["lat", "lng"]).copy()
        if not _df.empty:
            dists = _df.apply(lambda r: haversine_km(lat, lng, r["lat"], r["lng"]), axis=1)
            idx = int(dists.idxmin())
            row = _df.loc[idx]
            display_nearest_city(
                city_name=str(row["City"]),
                country=str(row["Country"]),
                aqi_value=int(row["AQI Value"]),
                category=str(row["AQI Category"]),
                distance_km=float(dists.min()),
            )
    except Exception:
        pass

    st.markdown("</div>", unsafe_allow_html=True)  # close .result-card
