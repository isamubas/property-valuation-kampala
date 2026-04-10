import joblib
import pandas as pd
import numpy as np

# Load the saved model
model = joblib.load('models/random_forest_property_model.pkl')

def predict_property_price(input_data):
    """
    Predict property price using the trained Random Forest model.
    input_data: dictionary with feature values
    """
    # Convert input to DataFrame (must match training features exactly)
    df_input = pd.DataFrame([input_data])
    
    # Make prediction (this is on log scale)
    pred_log = model.predict(df_input)
    
    # Convert back to actual USD
    pred_usd = np.expm1(pred_log)[0]
    
    return round(pred_usd, 2)

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example input - replace with your own values
    new_property = {
        'Neighborhood': 5,                    # encoded value from your LabelEncoder
        'Latitude': 0.3420,
        'Longitude': 32.6056,
        'Bedrooms': 3,
        'Bathrooms': 2.5,
        'Essential_Utilities_score': 0,
        'Security_score': 1,
        'Access_score': 1,
        'Premium_features_score': 0,
        'View_and _outdoor_score': 0,
        'Wellness_score': 0
    }

    price = predict_property_price(new_property)
    print(f"Predicted Property Price: ${price:,}")
