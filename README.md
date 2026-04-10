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

