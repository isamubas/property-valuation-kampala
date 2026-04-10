---
title: Kampala Condominium Price Predictor
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# 🏠 Kampala Condominium Price Predictor

**Automated Valuation Model (AVM)** using Random Forest

Predicts residential condominium prices in Kampala, Uganda based on:
- Location (Latitude & Longitude)
- Bedrooms & Bathrooms
- Amenity scores (Utilities, Security, Access, etc.)

### Model Performance
- **Best Model**: Tuned Random Forest  
- **R² Score**: ≈ 0.457 (on log-transformed price)  
- Trained on 108 anonymized records

### Try the App
Just fill in the details and click **Predict Property Price**

---

**Note**: The full dataset is not shared publicly for privacy reasons. Only the trained model and encoder are provided.
#  Kampala Condominium Price Predictor

**Automated Valuation Model (AVM)** using Machine Learning (Random Forest)

This project predicts residential condominium prices in Kampala, Uganda using location (latitude/longitude), property features, and amenity scores.

## Key Results
- **Best Model**: Tuned Random Forest  
- **Performance**: R² ≈ 0.457 (log-transformed price)  
- **Dataset**: 108 observations (anonymized)  
- **Most Important Features**: Latitude, Longitude, Neighborhood, Bedrooms

## Project Structure
- `notebooks/` → Full analysis and model development  
- `models/` → Trained model (ready to use)  
- `src/` → Clean reusable Python code  
- `results/` → Visualizations

## How to Use

```bash
git clone https://github.com/isamubas/property-valuation-kampala.git
cd property-valuation-kampala
pip install -r requirements.txt
```

### Quick Prediction Example

```python
from src.inference import predict_property_price

price = predict_property_price(
    neighborhood_name="Kololo",
    latitude=0.3420,
    longitude=32.6056,
    bedrooms=3,
    bathrooms=2.5,
    essential_utilities=0,
    security_score=1,
    access_score=1,
    premium_features=0,
    view_outdoor=0,
    wellness_score=0
)
print(f"Predicted Price: ${price:,.2f}")
```

## Limitations
- Trained on a small dataset (108 records)  
- Full raw data is not shared for privacy reasons

## License
MIT License - See LICENSE file for details

Made  for academic and research purposes.
