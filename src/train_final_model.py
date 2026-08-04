"""Train, tune and save the final condominium valuation model.

Two things here are deliberate and worth reading before trusting the numbers:

**Nested cross-validation.** Hyperparameters are chosen in an inner loop that
never sees the outer test fold. Tuning and then reporting the best tuned score
on the same folds is a common way to report a number that cannot be reproduced
on new data; nested CV avoids it, at the cost of a lower and more honest score.

**Two validation schemes, reported side by side.** Coordinates in this dataset
are neighbourhood centroids (16 distinct locations across 87 rows). Random
folds place the same location in train and test, so tree models can recall a
neighbourhood price level. That is legitimate for the stated use case — valuing
a unit in a neighbourhood the model knows — but it is not generalisation, and
the grouped score is reported so the limit is visible.

Outputs: models/*.pkl, results/final_model_report.json, results/*.png
"""
import io
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    RepeatedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(
    os.environ.get(
        "CONDO_DATA",
        Path(__file__).resolve().parent.parent / "data" / "condominiums cleaned (1).xlsx",
    )
)
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Feature set established by src/audit_data.py and src/ablation.py.
FEATURES = [
    "Bedrooms",
    "Latitude",
    "Longitude",
    "Security_score",
    "Access_score",
    "View_and _outdoor_score",
]

DROPPED = {
    "final size": "10 observations of 87 — imputing would invent the variable",
    "Bathrooms": "59% missing and insignificant once Bedrooms is controlled for (p=0.74)",
    "Essential_Utilities_score": "one value in 97.7% of rows",
    "Premium_features_score": "one value in 95.4% of rows",
    "Wellness_score": "one value in 95.4% of rows",
}

PARAM_GRID = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [None, 4, 8],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 1.0],
}

RANDOM_STATE = 42


def load():
    df = pd.read_excel(RAW)
    train = df[df["Price USD"].notna()].copy()
    demo = df[df["Price USD"].isna()].copy()  # complete features, no target
    y = np.log(train["Price USD"])
    groups = train["Latitude"].astype(str) + "_" + train["Longitude"].astype(str)
    return train, demo, train[FEATURES], y, groups


def base_pipeline(model):
    """Impute and scale inside the pipeline so nothing leaks across folds."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def nested_cv_score(X, y, groups) -> dict:
    """Outer folds score a model whose hyperparameters were tuned inside."""
    inner = RepeatedKFold(n_splits=4, n_repeats=1, random_state=RANDOM_STATE)
    search = GridSearchCV(
        base_pipeline(RandomForestRegressor(random_state=RANDOM_STATE)),
        PARAM_GRID,
        cv=inner,
        scoring="r2",
        n_jobs=-1,
    )

    outer_random = RepeatedKFold(n_splits=5, n_repeats=4, random_state=RANDOM_STATE)
    random_scores = cross_val_score(search, X, y, cv=outer_random, scoring="r2", n_jobs=-1)

    outer_grouped = GroupShuffleSplit(n_splits=20, test_size=0.25, random_state=RANDOM_STATE)
    grouped_scores = cross_val_score(
        search, X, y, cv=outer_grouped, groups=groups, scoring="r2", n_jobs=-1
    )
    return {
        "random_mean": float(random_scores.mean()),
        "random_std": float(random_scores.std()),
        "grouped_mean": float(grouped_scores.mean()),
        "grouped_std": float(grouped_scores.std()),
    }


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    train, demo, X, y, groups = load()
    print(f"Training rows: {len(X)} | features: {len(FEATURES)} | "
          f"distinct locations: {groups.nunique()}")
    print(f"Held aside (no target, used as demo set): {len(demo)}\n")

    # --- Honest performance estimate --------------------------------------
    print("Nested CV (tuning happens inside each outer fold)...")
    nested = nested_cv_score(X, y, groups)
    print(f"  random  CV R2 = {nested['random_mean']:+.3f} +/- {nested['random_std']:.3f}")
    print(f"  grouped CV R2 = {nested['grouped_mean']:+.3f} +/- {nested['grouped_std']:.3f}\n")

    # --- Fit the deployable model -----------------------------------------
    print("Fitting final model on all training rows...")
    search = GridSearchCV(
        base_pipeline(RandomForestRegressor(random_state=RANDOM_STATE)),
        PARAM_GRID,
        cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE),
        scoring="r2",
        n_jobs=-1,
    ).fit(X, y)
    best = search.best_estimator_
    print(f"  best params: { {k.replace('model__',''): v for k,v in search.best_params_.items()} }")

    # --- Benchmarks for context -------------------------------------------
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    benchmarks = {}
    for label, model in [
        ("Ridge (linear hedonic)", RidgeCV()),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ("Location only (RF)", None),
    ]:
        if label.startswith("Location"):
            s = cross_val_score(
                base_pipeline(RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)),
                X[["Latitude", "Longitude"]], y, cv=cv, scoring="r2",
            )
        else:
            s = cross_val_score(base_pipeline(model), X, y, cv=cv, scoring="r2")
        benchmarks[label] = {"mean": float(s.mean()), "std": float(s.std())}
        print(f"  {label:<26} {s.mean():+.3f} +/- {s.std():.3f}")

    # --- Permutation importance (honest; impurity importance is biased) ----
    perm = permutation_importance(best, X, y, n_repeats=50, random_state=RANDOM_STATE)
    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean,
                      "std": perm.importances_std})
        .sort_values("importance", ascending=False)
    )
    print("\nPermutation importance:")
    print(importance.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # --- Prediction interval from residual spread -------------------------
    resid = y - best.predict(X)
    sigma = float(resid.std())
    print(f"\nResidual sigma (log scale): {sigma:.3f} -> a 95% interval spans roughly "
          f"x{np.exp(1.96*sigma):.2f} / /{np.exp(1.96*sigma):.2f} around the point estimate")

    # --- Demo predictions on the 21 priceless rows ------------------------
    demo_pred = np.exp(best.predict(demo[FEATURES]))
    demo_out = demo[["Neighborhood", "Bedrooms"]].copy()
    demo_out["predicted_usd"] = demo_pred.round(0)
    demo_out.to_csv(RESULTS / "demo_predictions.csv", index=False)

    # --- Save artefacts ----------------------------------------------------
    joblib.dump(best, MODELS / "condominium_rf_model.pkl")
    metadata = {
        "trained_rows": int(len(X)),
        "features": FEATURES,
        "dropped_features": DROPPED,
        "target": "log(Price USD)",
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "nested_cv": nested,
        "benchmarks": benchmarks,
        "permutation_importance": importance.to_dict("records"),
        "residual_sigma_log": sigma,
        "distinct_locations": int(groups.nunique()),
    }
    (RESULTS / "final_model_report.json").write_text(json.dumps(metadata, indent=2))
    joblib.dump({"features": FEATURES, "sigma": sigma}, MODELS / "model_metadata.pkl")

    # --- Figures -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].barh(importance["feature"][::-1], importance["importance"][::-1],
               xerr=importance["std"][::-1], color="#4a7c9b")
    ax[0].set(xlabel="Drop in R² when shuffled", title="Permutation importance")
    pred = best.predict(X)
    ax[1].scatter(np.exp(y), np.exp(pred), alpha=0.7, edgecolor="k", linewidth=0.4)
    lims = [np.exp(y).min(), np.exp(y).max()]
    ax[1].plot(lims, lims, "crimson", lw=1, ls="--")
    ax[1].set(xlabel="Actual USD", ylabel="Predicted USD",
              title="Actual vs predicted (in-sample)", xscale="log", yscale="log")
    fig.tight_layout()
    fig.savefig(RESULTS / "final_model.png", dpi=130)

    print(f"\nSaved model -> models/condominium_rf_model.pkl")
    print(f"Saved report -> results/final_model_report.json")


if __name__ == "__main__":
    main()
