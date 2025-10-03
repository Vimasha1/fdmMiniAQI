
# app/utils.py
CATEGORIES = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

COLORS = {
    "Good": "#2ECC71",
    "Moderate": "#F1C40F",
    "Unhealthy for Sensitive Groups": "#E67E22",
    "Unhealthy": "#E74C3C",
    "Very Unhealthy": "#8E44AD",
    "Hazardous": "#7D3C98",
}

def health_tip(cat: str) -> str:
    tips = {
        "Good": "Air quality is satisfactory—enjoy outdoor activities.",
        "Moderate": "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Sensitive groups reduce prolonged outdoor exertion; consider a mask.",
        "Unhealthy": "Everyone limit prolonged outdoor exertion; mask recommended.",
        "Very Unhealthy": "Avoid outdoor activity; use high-quality masks indoors/outdoors.",
        "Hazardous": "Stay indoors; consider air purifiers; follow local health advisories.",
    }
    return tips.get(cat, "Check local guidance.")
