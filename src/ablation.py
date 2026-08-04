"""Does dropping the sparse / near-constant features actually cost accuracy?

The audit recommended dropping five variables. That recommendation is only
worth acting on if removing them does not hurt out-of-sample performance — so
this measures it rather than assuming it.

Each specification is scored with repeated CV, imputation inside the pipeline,
and the spread reported. With n=87 the spread is the point: a difference much
smaller than the standard deviation is not a difference.

Writes results/ablation.csv.
"""
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
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
OUT = ROOT / "results" / "ablation.csv"

CORE = ["Bedrooms", "Latitude", "Longitude"]
WEAK_SCORES = ["Security_score", "Access_score", "View_and _outdoor_score"]
NEAR_CONSTANT = ["Essential_Utilities_score", "Premium_features_score", "Wellness_score"]
SPARSE = ["Bathrooms", "final size"]

SPECS = {
    "Everything (nothing dropped)": CORE + WEAK_SCORES + NEAR_CONSTANT + SPARSE,
    "Drop near-constant scores only": CORE + WEAK_SCORES + SPARSE,
    "Drop near-constant + 'final size'": CORE + WEAK_SCORES + ["Bathrooms"],
    "Audit recommendation (also drop Bathrooms)": CORE + WEAK_SCORES,
    "Core + Bathrooms only": CORE + ["Bathrooms"],
    "Core only (Bedrooms + coordinates)": CORE,
    "Bedrooms only (no location)": ["Bedrooms"],
    "Location only (no property attributes)": ["Latitude", "Longitude"],
}


def load():
    df = pd.read_excel(RAW)
    d = df[df["Price USD"].notna()].copy()
    return d, np.log(d["Price USD"])


def pipe(model):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def main() -> None:
    d, y = load()
    cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)

    rows = []
    print(f"{'specification':<45}{'k':>3}{'Ridge':>18}{'RandomForest':>20}")
    print("-" * 86)
    for name, cols in SPECS.items():
        X = d[cols]
        out = {"specification": name, "n_features": len(cols)}
        cells = []
        for label, model in [
            ("ridge", RidgeCV()),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]:
            s = cross_val_score(pipe(model), X, y, cv=cv, scoring="r2")
            out[f"{label}_mean"], out[f"{label}_std"] = s.mean(), s.std()
            cells.append(f"{s.mean():+.3f} +/- {s.std():.3f}")
        rows.append(out)
        print(f"{name:<45}{len(cols):>3}{cells[0]:>18}{cells[1]:>20}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    best = df.loc[df["rf_mean"].idxmax()]
    audit = df[df["specification"].str.startswith("Audit")].iloc[0]
    everything = df[df["specification"].str.startswith("Everything")].iloc[0]

    print("-" * 86)
    print(f"\nBest Random Forest spec : {best['specification']} "
          f"({best['rf_mean']:+.3f})")
    print(f"Audit recommendation    : {audit['rf_mean']:+.3f} +/- {audit['rf_std']:.3f} "
          f"using {audit['n_features']} features")
    print(f"Everything kept         : {everything['rf_mean']:+.3f} +/- {everything['rf_std']:.3f} "
          f"using {everything['n_features']} features")
    diff = audit["rf_mean"] - everything["rf_mean"]
    pooled = (audit["rf_std"] + everything["rf_std"]) / 2
    print(f"\nDifference: {diff:+.3f}, against a typical fold-to-fold spread of "
          f"+/-{pooled:.3f}")
    print("Dropping the five variables changes accuracy by far less than the noise"
          if abs(diff) < pooled else "The difference exceeds the typical spread")
    print(f"\nSaved -> results/{OUT.name}")


if __name__ == "__main__":
    main()
