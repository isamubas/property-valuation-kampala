# 🏠 Kampala Condominium Price Predictor

**Automated Valuation Model (AVM)** using Machine Learning (Random Forest)

This project predicts residential condominium prices in Kampala, Uganda using location (latitude/longitude), property features, and amenity scores.

## Key Results
- **Best Model**: Tuned Random Forest  
- **Performance**: R² ≈ 0.59 (log-transformed price)  
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

Made with ❤️ for academic and research purposes.