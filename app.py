import gradio as gr
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load model and encoder
BASE_DIR = Path(__file__).parent
model = joblib.load(BASE_DIR / "models/random_forest_property_model.pkl")
encoder = joblib.load(BASE_DIR / "models/neighborhood_encoder.pkl")

neighborhood_names = encoder.classes_.tolist()

def predict_price(
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
):
    neighborhood_encoded = encoder.transform([neighborhood_name])[0]

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

    pred_log = model.predict(input_df)
    pred_usd = np.expm1(pred_log)[0]
    return f"${pred_usd:,.2f}"

# ==================== Gradio App ====================
with gr.Blocks(title="Kampala Property Valuation") as app:
    gr.Markdown("# 🏠 Kampala Condominium Price Predictor")
    gr.Markdown("### Random Forest Model (R² ≈ 0.59)")

    with gr.Row():
        with gr.Column():
            neighborhood = gr.Dropdown(choices=neighborhood_names, label="Neighborhood", value=neighborhood_names[0])
            lat = gr.Number(label="Latitude", value=0.3420, precision=6)
            lon = gr.Number(label="Longitude", value=32.6056, precision=6)
            bedrooms = gr.Number(label="Bedrooms", value=3, precision=1)
            bathrooms = gr.Number(label="Bathrooms", value=2.5, precision=1)

        with gr.Column():
            essential = gr.Slider(0, 3, value=0, step=1, label="Essential Utilities")
            security = gr.Slider(0, 3, value=1, step=1, label="Security")
            access = gr.Slider(0, 3, value=1, step=1, label="Access")
            premium = gr.Slider(0, 3, value=0, step=1, label="Premium Features")
            view = gr.Slider(0, 3, value=0, step=1, label="View & Outdoor")
            wellness = gr.Slider(0, 3, value=0, step=1, label="Wellness")

    btn = gr.Button("🔮 Predict Property Price", variant="primary", size="large")
    output = gr.Textbox(label="Predicted Price (USD)")

    btn.click(
        fn=predict_price,
        inputs=[neighborhood, lat, lon, bedrooms, bathrooms,
                essential, security, access, premium, view, wellness],
        outputs=output
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        show_error=True
    )
