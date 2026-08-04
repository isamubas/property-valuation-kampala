---
title: Kampala Condominium Price Predictor
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
---

# 🏠 Kampala Condominium Price Predictor

A Random Forest **hedonic valuation model** for Kampala condominiums, rebuilt
with a full audit trail: data quality diagnostics, regression diagnostics,
feature ablation, a nineteen-algorithm comparison, and nested cross-validation.

| Metric | This version | Previous |
|---|---|---|
| **Nested-CV R²** (log price) | **0.669 ± 0.165** | 0.457 |
| MAPE (USD) | 40.3% | not reported |
| Within 20% of actual | 51.7% | not reported |
| 95% prediction interval | ×2.07 / ÷2.07 | not reported |

**[Try the app](https://huggingface.co/spaces/The-Rookie/kampala-property-price-predictor)**

## Read this before using a number from it

The R² improved, but the model remains limited in ways worth stating plainly:

- **Typically wrong by 40%.** Only 52% of predictions land within 20% of the
  true price; commercial AVMs target 10–15%. A $200k estimate carries a 95%
  interval of roughly $97k–$414k. **The interval is the answer, not the midpoint.**
- **~94% of its power is location.** Permutation importance puts `Latitude` at
  0.694 against `Bedrooms` at 0.052. It is closer to a neighbourhood price
  lookup with a small structural adjustment than a full hedonic model.
- **Valid only for the 18 neighbourhoods it was trained on.** Held out whole
  *neighbourhoods*, R² is negative (−0.196). It cannot value a property
  somewhere unseen.
- **Trained on 87 records**, and on **listing** prices rather than achieved
  sale prices.

Full write-up: **[notes/model-documentation.md](notes/model-documentation.md)**

## What changed, and why

The previous model reported R² 0.457 on 108 records. Rebuilding surfaced
several issues:

**1. Only 87 of the 108 rows are usable.** `Price USD` is missing on 21. A
target cannot be imputed — filling it and then training to predict it means
learning the imputation rule, not the market. Those rows are kept as a demo
prediction set instead.

**2. Five features were removed, at no cost to accuracy.**

| Dropped | Reason |
|---|---|
| `final size` | 10 observations of 87 — floor area, normally the key variable, is effectively absent |
| `Bathrooms` | 59% missing, and insignificant once bedrooms is controlled for (p = 0.74) |
| `Essential_Utilities_score` | one value in 97.7% of rows |
| `Premium_features_score` | one value in 95.4% of rows |
| `Wellness_score` | one value in 95.4% of rows |

Ablation confirmed the cost: **0.005 R²**, against fold-to-fold noise of
±0.186. Median-filling `Bathrooms` had actively harmed the model, producing a
significant *negative* coefficient (−1.17, p < 0.001) — more bathrooms
predicting a lower price, which is an imputation artefact rather than a finding.

**3. Imputation moved inside the CV pipeline.** Filling missing values before
splitting leaks test-fold information into training.

**4. Two validation schemes are now reported.** Coordinates in this data are
neighbourhood centroids — 16 distinct locations across 87 rows — so random
folds let tree models recall a neighbourhood price level. That is legitimate
for valuing within known neighbourhoods, but it is not generalisation, and the
grouped score makes the limit visible.

**5. Back-transformation fixed.** The previous app used `np.expm1()`, the
inverse of `log1p`, on a model trained on plain `log`. Duan's smearing
correction is now applied too, since `exp()` of a log prediction returns the
conditional median and ran ~14% low.

## Diagnostics

| Test | Result |
|---|---|
| VIF | max 2.45 — no multicollinearity |
| Condition number | 48,237 raw → **3.4 standardised** (a scaling artefact, not collinearity) |
| Breusch–Pagan | p = 0.0198 — heteroskedasticity present |
| Jarque–Bera | p = 0.434 — residuals normal |
| Ramsey RESET | p = 0.0004 — non-linearity present |
| Cook's distance | 4 of 87 above 4/n, max 0.072 |

**No spatial statistics.** Moran's I and LISA were considered and excluded —
16 neighbourhood centroids across 87 rows cannot support spatial
autocorrelation inference. Location enters as fixed effects instead.

## Model comparison

Nineteen algorithms screened, then evaluated out-of-fold:

| Model | R² (log) | MAPE | Within 20% |
|---|---|---|---|
| **Random Forest** | **+0.683** | **40.3%** | **51.7%** |
| Extra Trees | +0.643 | 43.2% | 56.3% |
| Gradient Boosting | +0.634 | 44.5% | 51.7% |
| SVR (rbf) | +0.580 | 50.8% | 46.0% |
| Ridge (linear hedonic) | +0.187 | 98.3% | 14.9% |
| Neural net (64,32) | −1.220 | 1,832% | 12.6% |
| Neural net (16) | −7.864 | 28,178% | 0% |

All tree ensembles cluster at 0.65–0.68 — differences smaller than the ±0.17
spread, so statistically indistinguishable. Random Forest was chosen for
stability on small samples, not because it nominally edged the others.

**Neural networks fail structurally here.** Even with small architectures, L2
regularisation and early stopping, 87 rows cannot support thousands of weights;
they never converged and predicted prices in the millions.

## Reproducing

```bash
pip install -r requirements.txt

python src/audit_data.py         # data quality evidence
python src/diagnostics.py        # VIF, Breusch-Pagan, RESET, Cook's distance
python src/ablation.py           # does dropping features cost accuracy?
python src/compare_models.py     # 19-algorithm screen
python src/evaluate_all.py       # accuracy incl. neural nets, USD metrics
python src/train_final_model.py  # nested CV, tuning, saved artefacts

python app.py                    # run the Gradio app locally
```

The source dataset is not published (privacy); trained artefacts and all
reports are.

## What would improve it most

1. **Floor area for every record** — the single biggest gap
2. **More rows** — 87 is the binding constraint on every estimate
3. **True property coordinates** rather than neighbourhood centroids, which
   would also make spatial statistics viable
4. **Transaction prices** rather than listings
5. **Sale dates**, enabling temporal control and time-based validation
