# app/live_sources.py
from __future__ import annotations
import requests
import pandas as pd
from typing import Any, Dict

OPENAQ_URL = "https://api.openaq.org/v2/measurements"

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["parameter", "value", "unit", "date.utc", "location", "latitude", "longitude"])

def fetch_openaq_nearby(lat: float, lng: float, radius_m: int = 100_000, limit: int = 400) -> pd.DataFrame:
    """
    Fetch recent OpenAQ measurements near (lat,lng). Resilient:
    - tries progressively larger radii (100 -> 200 -> 300 km)
    - catches all network/HTTP errors and returns an empty DataFrame
    """
    params_base: Dict[str, Any] = {
        "coordinates": f"{lat},{lng}",
        "limit": int(limit),
        "order_by": "datetime",
        "sort": "desc",
        "parameters": "pm25,o3,no2,co",
    }

    for rad in (radius_m, 200_000, 300_000):
        try:
            params = dict(params_base)
            params["radius"] = int(rad)
            r = requests.get(OPENAQ_URL, params=params, timeout=20)
            r.raise_for_status()
            js = r.json()
            rows = js.get("results", [])
            if not rows:
                continue

            # Normalize into a DataFrame
            data = []
            for it in rows:
                data.append({
                    "parameter": it.get("parameter"),
                    "value": it.get("value"),
                    "unit": it.get("unit"),
                    "date.utc": (it.get("date") or {}).get("utc"),
                    "location": it.get("location"),
                    "latitude": (it.get("coordinates") or {}).get("latitude"),
                    "longitude": (it.get("coordinates") or {}).get("longitude"),
                })
            df = pd.DataFrame(data)
            df.attrs["radius_used"] = rad
            return df

        except requests.exceptions.RequestException:
            # DNS/timeout/HTTP errors -> try next radius or return empty
            continue
        except Exception:
            # Any parsing issues -> return empty
            return _empty_df()

    return _empty_df()
