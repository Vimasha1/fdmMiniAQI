# app/aqi_math.py
from __future__ import annotations
import math

# US EPA-style AQI breakpoints for sub-indices.
# NOTE: These expect already-normalized units:
# - pm25 in µg/m³ (24h or nowcast context)
# - o3 in ppb (8h)
# - no2 in ppb (1h)
# - co in ppm (8h)
BPS = {
    "pm25": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ],
    "o3": [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],  # ppb (8h)
    "no2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],  # ppb (1h)
    "co": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],  # ppm (8h)
}

def _linear(val, bp):
    Clow, Chigh, Ilow, Ihigh = bp
    return (Ihigh - Ilow) / (Chigh - Clow) * (val - Clow) + Ilow

def to_aqi(val: float | None, pollutant: str) -> float | None:
    """Convert a concentration (already in the expected unit) to a pollutant AQI sub-index."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    for bp in BPS[pollutant]:
        if bp[0] <= val <= bp[1]:
            return round(_linear(val, bp))
    return None  # out of supported range

def category(aqi: float | None) -> str:
    if aqi is None: return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"
