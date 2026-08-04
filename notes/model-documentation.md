# Kampala Condominium Valuation Model — technical documentation

## Summary

A Random Forest hedonic model predicting condominium sale prices in Kampala
from location and structural attributes.

| Metric | Value |
|---|---|
| **Nested CV R²** (tuning inside folds — the honest figure) | **0.669 ± 0.165** |
| Simple out-of-fold R² | 0.683 |
| Previous model | 0.457 |
| MAPE (USD) | 40.3% |
| 95% prediction interval | ×2.07 / ÷2.07 |

That headline number needs two qualifications, both developed below:

- In USD terms the model is typically wrong by **40%**, and only **52% of
  predictions land within 20%** of the actual price. Commercial AVMs target
  10–15% MAPE.
- Roughly **94% of its predictive power comes from location alone.** It is
  closer to a neighbourhood price lookup with a small structural adjustment
  than a full hedonic valuation.

Everything here is reproducible:

```bash
python src/audit_data.py        # data quality evidence
python src/diagnostics.py       # VIF, Breusch-Pagan, RESET, Cook's distance
python src/ablation.py          # does dropping features cost accuracy?
python src/evaluate_all.py      # all model families incl. neural networks
python src/train_final_model.py # nested CV, tuning, saved artefacts
```

## 1. Data

108 condominium records from Kampala, with neighbourhood, coordinates,
bedrooms, bathrooms, floor area, and six amenity scores.

**87 rows are usable.** `Price USD` — the target — is missing on 21 rows. A
target cannot be imputed: filling it and then training to predict it means the
model learns the imputation rule rather than the market, and any resulting R²
partly measures the analyst's own guess. Those 21 rows are retained as a demo
prediction set (`results/demo_predictions.csv`), where the absence of a true
price is not a problem.

The missing prices are not spread evenly — 9 of 21 are Ntinda, 3 Muyenga — so
dropping them thins specific neighbourhoods.

## 2. Data quality findings

### 2.1 Floor area is unusable

`final size` has **10 non-missing values out of 87** (88.5% missing). Floor
area is normally the single most important variable in a hedonic price model.
Its effective absence is the central limitation of this dataset, and no
modelling choice compensates for it.

### 2.2 Three amenity scores are near-constant

| Feature | Modal value | Share of rows |
|---|---|---|
| `Essential_Utilities_score` | 0 | 97.7% |
| `Premium_features_score` | 0 | 95.4% |
| `Wellness_score` | 0 | 95.4% |

A variable that does not vary cannot explain variation in price. These add
noise and dilute tree splits.

### 2.3 Bathrooms carries no independent signal

This one is subtle and was initially misleading.

| Measure | Value |
|---|---|
| Correlation with log price, on the 36 observed rows | **+0.539** |
| Correlation after median-filling all 87 rows | +0.124 |
| Coefficient controlling for `Bedrooms`, observed rows only | −0.122, **p = 0.74** |

The strong raw correlation is confounded with bedrooms — larger units have
more of both. Once bedrooms is controlled for, bathrooms is insignificant.
Median-filling 59% of the column then produced a **negative and highly
significant** coefficient (−1.17, p < 0.001) in the full specification, which
is economically nonsensical and is an imputation artefact, not a finding.

Dropped.

### 2.4 Coordinates are neighbourhood centroids

14 unique latitudes, 11 longitudes, **16 distinct lat/long pairs across 87
rows** — an average of 5.4 properties sharing identical coordinates. These are
neighbourhood centroids, not property locations. They encode the same
information as `Neighborhood`.

This has a direct consequence for validation, covered in section 4.

## 3. Diagnostics

Run via `src/diagnostics.py` on the OLS hedonic specification.

| Test | Result | Interpretation |
|---|---|---|
| **VIF** | max 2.45 | No multicollinearity |
| **Condition number** | 48,237 raw → **3.4 standardised** | The large raw value is a *scaling* artefact (Longitude mean 32.6 / sd 0.026 alongside Latitude at 0.32), not collinearity. True collinearity does not disappear under rescaling |
| **Breusch–Pagan** | LM 16.66, p = 0.0198 | Heteroskedasticity present; inference requires HC3 robust standard errors |
| **Jarque–Bera** | JB 1.67, p = 0.434 | Residuals consistent with normality — the log transform worked |
| **Ramsey RESET** | F 13.81, p = 0.0004 | Functional form misspecified; genuine non-linearity, part of why trees beat linear models |
| **Cook's distance** | 4 of 87 above 4/n, max 0.072 | No single observation distorting the fit |

### No spatial statistics

Moran's I and LISA were considered and **deliberately excluded**. They require
spatial units; this data has 16 neighbourhood centroids across 87 rows.
Area-level spatial autocorrelation inference on n=16 would be badly
underpowered and would produce confident-looking maps unsupported by the data.
Location instead enters as fixed effects.

## 4. Validation design

Two schemes are reported, because they answer different questions.

| Scheme | Question | Use |
|---|---|---|
| **Random k-fold** | "Value a unit in a neighbourhood the model knows" | The stated use case |
| **Grouped by location** | "Value a unit in a neighbourhood never seen" | Honest limit statement |

With 16 locations and 5.4 rows each, a random split places the same coordinates
in train and test, letting tree models recall a neighbourhood price level. For
the stated use case that is legitimate — it is what neighbourhood fixed effects
do. It is *not* generalisation, and the grouped score makes the difference
visible:

| Feature set | Random CV | Grouped CV |
|---|---|---|
| Full | 0.676 | **−0.124** |
| Ridge, full | 0.191 | −1.539 |

**Every model tested is negative under grouped CV.** The model cannot value a
property in a neighbourhood absent from training.

## 5. Feature selection, and whether dropping cost anything

Five variables were dropped. Because "more determinants must mean more
accuracy" is a reasonable objection, this was measured rather than assumed
(`src/ablation.py`, repeated CV, 20 repeats):

| Specification | k | Random Forest R² |
|---|---|---|
| Everything kept | 11 | 0.669 ± 0.183 |
| **Audit recommendation** | 6 | **0.665 ± 0.188** |
| Core only (bedrooms + coordinates) | 3 | 0.654 ± 0.193 |
| **Location only** | 2 | **0.632 ± 0.207** |
| Bedrooms only | 1 | 0.048 ± 0.268 |

Dropping all five costs **0.005 R²** against a fold-to-fold spread of ±0.186 —
roughly 1/37th of the noise. Not a meaningful loss.

**The more important result in that table:** location alone reaches 0.632 of
the full model's 0.669. Bedrooms, bathrooms and all six amenity scores together
contribute about **0.037**. This is the empirical basis for describing the
model as predominantly locational.

## 6. Model comparison

Out-of-fold predictions, 5-fold CV, imputation inside the pipeline. USD metrics
use **Duan's smearing correction** — plain `exp()` back-transformation returns
the conditional median and underestimates the mean by roughly 14% here.

| Model | R² (log) | MAPE | Median error | Within 20% |
|---|---|---|---|---|
| **Random Forest** | **+0.683** | **40.3%** | **$27,178** | **51.7%** |
| Extra Trees | +0.643 | 43.2% | $37,378 | 56.3% |
| Gradient Boosting | +0.634 | 44.5% | $31,632 | 51.7% |
| SVR (rbf) | +0.580 | 50.8% | $24,071 | 46.0% |
| Ridge (linear hedonic) | +0.187 | 98.3% | $92,747 | 14.9% |
| Neural net (64,32), weak reg | −1.220 | 1,832% | $193,156 | 12.6% |
| Neural net (32,16) | −6.404 | 16,774% | $1,486,380 | 0% |
| Neural net (16) | −7.864 | 28,178% | $6,775,286 | 0% |

Nineteen algorithms were screened first (`results/model_comparison_condominiums.csv`),
including XGBoost, LightGBM, CatBoost, AdaBoost, KNN and robust regressors. All
tree ensembles clustered at 0.65–0.68 — differences far smaller than the ±0.17
spread, so they are statistically indistinguishable. Random Forest was selected
for stability on small samples and interpretability, not because 0.683 beats
0.678.

### Final tuned model

Hyperparameters selected by nested cross-validation, so the reported score does
not include the benefit of tuning on the folds it is scored against:

```
max_depth=4, max_features='sqrt', min_samples_leaf=1, n_estimators=500
```

| | Random CV | Grouped CV |
|---|---|---|
| Nested CV R² | **0.669 ± 0.165** | −0.196 ± 1.102 |

`max_depth=4` is shallow — with 87 rows, the tuner correctly chose a heavily
constrained forest.

**Permutation importance** (drop in R² when a feature is shuffled — unbiased,
unlike impurity importance which favours high-cardinality features):

| Feature | Importance |
|---|---|
| `Latitude` | **+0.694 ± 0.099** |
| `Longitude` | +0.120 ± 0.022 |
| `Bedrooms` | +0.052 ± 0.014 |
| `Access_score` | +0.046 ± 0.015 |
| `Security_score` | +0.009 ± 0.004 |
| `View_and _outdoor_score` | +0.003 ± 0.002 |

Latitude alone accounts for most of the model. North–south position in Kampala
separates the expensive southern neighbourhoods (Muyenga, Munyonyo, Kololo)
from cheaper northern ones — the model is largely reading that gradient.

### Prediction intervals

Residual standard deviation on the log scale is **0.371**, giving a 95%
interval of roughly **×2.07 / ÷2.07** around any point estimate.

In practice: a $200,000 valuation carries a 95% interval of about
**$97,000 – $414,000**. Any deployment must show this interval rather than a
bare point estimate, which would imply precision the model does not have.

### On neural networks

Included at request, and given a fair setup — small architectures, L2
regularisation, early stopping — rather than the untuned default that scored
−1.38 in the first sweep. They still fail catastrophically, predicting prices in
the millions for properties worth ~$180k, and never converged within 5,000
iterations.

This is not a tuning problem. With 87 rows and `early_stopping` reserving a
further 10% for validation, the networks fit ~70 examples across thousands of
weights. Neural networks require sample sizes in the thousands. This is a
structural mismatch between method and data.

## 7. Limitations

Ordered by how much they constrain the result.

1. **Sample size: 87 usable rows.** Every estimate carries wide uncertainty;
   fold-to-fold R² varies by ±0.18. Differences between models smaller than
   that are noise.

2. **Floor area effectively absent** (10 of 87 observations). The dominant
   structural variable in hedonic pricing is missing, and nothing recovers it.

3. **Practical accuracy is weak.** MAPE 40.3%; only 51.7% of predictions land
   within 20% of actual. Commercial AVMs target 10–15%. The 95% prediction
   interval spans ×2.07/÷2.07, so a $200k valuation means roughly
   **$97k–$414k** — wide enough that the interval, not the point estimate, is
   the honest output.

4. **Predictive power is overwhelmingly locational.** Location alone gives
   0.632 of 0.669. Structural and amenity attributes add ~0.037.

5. **No generalisation to unseen neighbourhoods.** Grouped CV is negative for
   every model tested. Valid only for the 18 neighbourhoods in training.

6. **Coordinates are centroids, not property locations**, so within-neighbourhood
   location effects (street, aspect, floor) cannot be captured at all.

7. **Target missing for 19% of rows**, non-randomly by neighbourhood, thinning
   Ntinda and Muyenga specifically.

8. **Three amenity scores near-constant** (95–98% single value), carrying
   almost no information.

9. **Heteroskedasticity present** (BP p = 0.0198) — OLS standard errors are
   unreliable without HC3 correction. Does not affect the tree models' point
   predictions.

10. **Functional form misspecified for the linear model** (RESET p = 0.0004).

11. **Listing prices, not transaction prices.** Asking prices typically exceed
    achieved prices, so the model predicts what sellers ask, not what buyers pay.

12. **No temporal dimension.** No sale dates, so no control for market movement
    and no time-based validation split.

## 8. What would improve it most

In order of expected impact:

1. **Floor area for every record.** The single biggest gap.
2. **More rows.** 87 is the binding constraint on every estimate.
3. **True property coordinates** rather than neighbourhood centroids — would
   also make the spatial statistics viable.
4. **Transaction prices** rather than listings.
5. **Sale dates**, enabling temporal control and time-based validation.
6. **Amenity variables with actual variation** — the current ones are
   near-constant and contribute nothing.
