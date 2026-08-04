"""Gradio app for the Kampala condominium valuation model.

Two deliberate differences from the previous version:

1. **It returns an interval, not a bare number.** The residual spread implies a
   95% interval of roughly x2.07 / /2.07 around any point estimate. Showing
   "$200,000" alone would imply a precision this model does not have.

2. **Coordinates are derived from the neighbourhood, not typed by the user.**
   The training data holds 16 neighbourhood centroids, not property locations,
   so a hand-entered coordinate would be answered by whichever centroid it fell
   nearest anyway. Selecting a neighbourhood is the honest interface.

The previous app also back-transformed with `np.expm1`, which is the inverse of
log1p; the model is trained on plain log, so `np.exp` is correct. Duan's
smearing factor is applied on top, since exp() of a log prediction returns the
conditional median and underestimates the mean.
"""
import json
import traceback
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent

model = joblib.load(BASE / "models" / "condominium_rf_model.pkl")
metadata = joblib.load(BASE / "models" / "model_metadata.pkl")
CENTROIDS = joblib.load(BASE / "models" / "neighbourhood_centroids.pkl")

FEATURES = metadata["features"]
SIGMA = metadata["sigma"]

report = json.loads((BASE / "results" / "final_model_report.json").read_text())
R2 = report["nested_cv"]["random_mean"]
R2_SD = report["nested_cv"]["random_std"]

# Duan's smearing estimator, computed from the training residuals.
SMEARING = 1.139

NEIGHBOURHOODS = sorted(CENTROIDS)


def predict(neighbourhood: str, bedrooms: float, security: int, access: int, view: int):
    # Errors are caught and returned as text. Without this a failure inside the
    # handler leaves the output blank, which looks like "the button does
    # nothing" and gives the user nothing to report.
    try:
        centroid = CENTROIDS[neighbourhood]
        row = pd.DataFrame([{
            "Bedrooms": float(bedrooms),
            "Latitude": centroid["Latitude"],
            "Longitude": centroid["Longitude"],
            "Security_score": float(security),
            "Access_score": float(access),
            "View_and _outdoor_score": float(view),
        }])[FEATURES]

        log_pred = float(model.predict(row)[0])
        point = np.exp(log_pred) * SMEARING
        low = np.exp(log_pred - 1.96 * SIGMA) * SMEARING
        high = np.exp(log_pred + 1.96 * SIGMA) * SMEARING

        return (
            f"## ${point:,.0f}\n\n"
            f"**95% interval: ${low:,.0f} — ${high:,.0f}**\n\n"
            f"The interval is wide because the model is typically wrong by about "
            f"40%. Treat the range as the answer, not the midpoint."
        )
    except Exception as exc:
        return (
            f"### ⚠️ Prediction failed\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            f"<details><summary>Details</summary>\n\n"
            f"```\n{traceback.format_exc()}\n```\n</details>"
        )


LIMITATIONS = """
This model is **deliberately honest about being limited**. Before using a
number from it:

- **Trained on 87 records.** Small enough that every estimate carries wide
  uncertainty.
- **Typical error is 40%** (MAPE). Only 52% of predictions land within 20% of
  the true price. Commercial AVMs target 10–15%.
- **Predictions are ~94% location.** Latitude alone carries most of the model;
  bedrooms and amenity scores together add very little.
- **Only valid for the 18 neighbourhoods listed.** Tested on held-out
  *neighbourhoods*, performance is negative — it cannot value a property
  somewhere it has never seen.
- **Floor area is absent.** The most important variable in property valuation
  was missing from 88% of the source data and had to be excluded.
- **Based on listing prices**, not achieved sale prices. Asking prices usually
  exceed what buyers actually pay.

Full methodology and limitations: `notes/model-documentation.md` in the repo.
"""

with gr.Blocks(title="Kampala Condominium Valuation") as app:
    gr.Markdown(
        f"# 🏠 Kampala Condominium Price Predictor\n"
        f"Random Forest hedonic model · nested-CV R² **{R2:.3f} ± {R2_SD:.3f}** "
        f"on log price (previous version: 0.457)"
    )

    with gr.Row():
        with gr.Column():
            neighbourhood = gr.Dropdown(
                choices=NEIGHBOURHOODS, value="Kololo", label="Neighbourhood",
                info="Coordinates are set from the neighbourhood centroid",
            )
            bedrooms = gr.Slider(1, 6, value=3, step=1, label="Bedrooms")
        with gr.Column():
            security = gr.Slider(0, 3, value=1, step=1, label="Security score")
            access = gr.Slider(0, 2, value=1, step=1, label="Access score")
            view = gr.Slider(0, 2, value=0, step=1, label="View & outdoor score")

    btn = gr.Button("Estimate value", variant="primary", size="lg")
    out = gr.Markdown()

    btn.click(predict, [neighbourhood, bedrooms, security, access, view], out)

    with gr.Accordion("⚠️ Limitations — read before relying on this", open=False):
        gr.Markdown(LIMITATIONS)

    gr.Markdown(
        "Amenity scores dropped from the previous version "
        "(`Essential_Utilities`, `Premium_features`, `Wellness`) were constant "
        "in 95–98% of the training data and carried no information. `Bathrooms` "
        "was dropped as insignificant once bedrooms is controlled for."
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
