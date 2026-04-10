import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load model and encoder (relative path works when running from root)
BASE_DIR = Path(__file__).parent.parent
model_path = BASE_DIR / "models" / "random_forest_property_model.pkl"
encoder_path = BASE_DIR / "models" / "neighborhood_encoder.pkl"

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

print(" Model and encoder loaded successfully from src/inference.py")


def predict_property_price(
    neighborhood_name: str,
    latitude: float,
    longitude: float,
    bedrooms: float,
    bathrooms: float,
    essential_utilities: int = 0,
    security_score: int = 1,
    access_score: int = 1,
    premium_features: int = 0,
    view_outdoor: int = 0,
    wellness_score: int = 0
) -> float:
    """
    Predict condominium price using the trained Random Forest model.
    Returns price in USD.
    """
    # Convert neighborhood name to encoded number
    neighborhood_encoded = encoder.transform([neighborhood_name])[0]

    # Prepare input for the model
    input_df = pd.DataFrame([{
        'Neighborhood': neighborhood_encoded,
        'Latitude': latitude,
        'Longitude': longitude,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Essential_Utilities_score': essential_utilities,
        'Security_score': security_score,
        'Access_score': access_score,
        'Premium_features_score': premium_features,
        'View_and _outdoor_score': view_outdoor,
        'Wellness_score': wellness_score
    }])

    # Make prediction (log scale → convert back to USD)
    pred_log = model.predict(input_df)
    pred_usd = np.expm1(pred_log)[0]

    return round(float(pred_usd), 2)
