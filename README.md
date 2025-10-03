
# CleanAir Compass

A small web app to explore global AQI by city and predict AQI categories from pollutant inputs.

## Quickstart (local)

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Project Layout
```
cleanair-compass/
  data/AQI_and_LatLong.csv
  app/app.py
  app/utils.py
  app/aqi_model.joblib
  requirements.txt
  README.md
```

## Model
- Features: CO, Ozone, NO2, PM2.5 AQI values + lat, lng
- Target: AQI Category (multiclass)
- Classifier: Logistic Regression (balanced class weights)
- Metrics (hold-out): Accuracy=0.926, Macro F1=0.879
