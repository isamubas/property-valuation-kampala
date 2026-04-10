import gradio as gr
from inference import predict_property_price, encoder

# Get list of real neighborhood names for dropdown
neighborhood_names = encoder.classes_.tolist()

def predict(
    neighborhood_name: str,
    latitude: float,
    longitude: float,
    bedrooms: float,
    bathrooms: float,
    essential_utilities: int,
    security_score: int,
    access_score: int,
    premium_features: int,
    view_outdoor: int,
    wellness_score: int
):
    price = predict_property_price(
        neighborhood_name=neighborhood_name,
        latitude=latitude,
        longitude=longitude,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        essential_utilities=essential_utilities,
        security_score=security_score,
        access_score=access_score,
        premium_features=premium_features,
        view_outdoor=view_outdoor,
        wellness_score=wellness_score
    )
    return f"${price:,.2f}"


# ==================== Gradio Web App ====================
with gr.Blocks(title="Kampala Property Valuation") as app:
    gr.Markdown("#  Kampala Condominium Price Predictor")
    gr.Markdown("### Random Forest Model (R² ≈ 0.457)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Location")
            neighborhood_input = gr.Dropdown(
                choices=neighborhood_names,
                label="Neighborhood",
                value=neighborhood_names[0]
            )
            lat = gr.Number(label="Latitude", value=0.3420, precision=6)
            lon = gr.Number(label="Longitude", value=32.6056, precision=6)

        with gr.Column():
            gr.Markdown("### Property Details")
            bedrooms_input = gr.Number(label="Bedrooms", value=3, precision=1)
            bathrooms_input = gr.Number(label="Bathrooms", value=2.5, precision=1)

            gr.Markdown("### Amenity Scores (0-3)")
            essential = gr.Slider(0, 3, value=0, step=1, label="Essential Utilities")
            security = gr.Slider(0, 3, value=1, step=1, label="Security")
            access = gr.Slider(0, 3, value=1, step=1, label="Access")
            premium = gr.Slider(0, 3, value=0, step=1, label="Premium Features")
            view = gr.Slider(0, 3, value=0, step=1, label="View & Outdoor")
            wellness = gr.Slider(0, 3, value=0, step=1, label="Wellness")

    predict_btn = gr.Button("🔮 Predict Property Price", variant="primary", size="large")
    output = gr.Textbox(label="Predicted Price (USD)", placeholder="$xxx,xxx.xx")

    predict_btn.click(
        fn=predict,
        inputs=[
            neighborhood_input, lat, lon, bedrooms_input, bathrooms_input,
            essential, security, access, premium, view, wellness
        ],
        outputs=output
    )

if __name__ == "__main__":
    app.launch(share=True)
