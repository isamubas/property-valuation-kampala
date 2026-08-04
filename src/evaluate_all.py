"""Full accuracy comparison across model families, including neural networks.

Reports metrics in **USD**, not just log-scale R². For a valuation model
"typically wrong by 34%" is the number a user cares about; log-scale R² is not
interpretable to anyone outside the modelling.

Back-transforming a log prediction with exp() gives the *conditional median*,
which underestimates the mean. Duan's smearing estimator corrects for that, and
is applied before any USD-scale metric is computed.

Neural networks are included at the user's request. They are given a fair
setup — small architecture, L2 regularisation, early stopping — because the
default MLP that scored -1.38 in the first sweep was simply misconfigured for
87 rows. The result is reported honestly either way.

Writes results/accuracy_comparison.csv and results/actual_vs_predicted.png
"""
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(
    os.environ.get(
        "CONDO_DATA",
        Path(__file__).resolve().parent.parent / "data" / "condominiums cleaned (1).xlsx",
    )
)
RESULTS = ROOT / "results"

FEATURES = [
    "Bedrooms", "Latitude", "Longitude",
    "Security_score", "Access_score", "View_and _outdoor_score",
]
RANDOM_STATE = 42


def load():
    df = pd.read_excel(RAW)
    d = df[df["Price USD"].notna()].copy()
    return d[FEATURES], np.log(d["Price USD"]), d["Price USD"].values


def pipe(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def models() -> dict:
    rs = RANDOM_STATE
    return {
        "Random Forest": RandomForestRegressor(n_estimators=500, random_state=rs),
        "Gradient Boosting": GradientBoostingRegressor(random_state=rs),
        "Extra Trees": ExtraTreesRegressor(n_estimators=500, random_state=rs),
        "SVR (rbf)": SVR(C=10, gamma="scale"),
        "Ridge (linear hedonic)": RidgeCV(),
        # Neural networks, given a setup appropriate to n=87 rather than defaults.
        "Neural net (16)": MLPRegressor(
            hidden_layer_sizes=(16,), alpha=1.0, max_iter=5000,
            early_stopping=True, n_iter_no_change=50, random_state=rs),
        "Neural net (32,16)": MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=5000,
            early_stopping=True, n_iter_no_change=50, random_state=rs),
        "Neural net (64,32) weak reg": MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=5000,
            early_stopping=True, n_iter_no_change=50, random_state=rs),
    }


def smearing_factor(residuals: np.ndarray) -> float:
    """Duan's smearing estimator: corrects exp() back-transformation bias."""
    return float(np.mean(np.exp(residuals)))


def metrics(y_log_true, y_log_pred, y_usd_true) -> dict:
    resid = y_log_true - y_log_pred
    smear = smearing_factor(resid)
    usd_pred = np.exp(y_log_pred) * smear

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_log_true - y_log_true.mean()) ** 2)
    r2_log = 1 - ss_res / ss_tot

    err = y_usd_true - usd_pred
    return {
        "R2_log": r2_log,
        "RMSE_usd": float(np.sqrt(np.mean(err**2))),
        "MAE_usd": float(np.mean(np.abs(err))),
        "MedAE_usd": float(np.median(np.abs(err))),
        "MAPE_pct": float(np.mean(np.abs(err / y_usd_true)) * 100),
        "within_20pct": float(np.mean(np.abs(err / y_usd_true) <= 0.20) * 100),
        "smearing": smear,
        "_usd_pred": usd_pred,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    X, y_log, y_usd = load()
    print(f"{len(X)} rows, {len(FEATURES)} features. "
          "Predictions are out-of-fold (5-fold CV), never in-sample.\n")

    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, preds = [], {}

    for name, model in models().items():
        oof = cross_val_predict(pipe(model), X, y_log, cv=cv)
        m = metrics(y_log.values, oof, y_usd)
        preds[name] = m.pop("_usd_pred")
        rows.append({"model": name, **m})
        print(f"  {name:<28} R2={m['R2_log']:+.3f}  MAPE={m['MAPE_pct']:5.1f}%  "
              f"MedAE=${m['MedAE_usd']:>8,.0f}  within20%={m['within_20pct']:4.1f}%")

    df = pd.DataFrame(rows).sort_values("R2_log", ascending=False)
    df.to_csv(RESULTS / "accuracy_comparison.csv", index=False)

    print("\n" + "=" * 92)
    print("Ranked by out-of-fold R2 (log scale). MAPE/MedAE are in USD after "
          "smearing correction.")
    print("=" * 92)
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    # --- Actual vs predicted, every model ---------------------------------
    n = len(preds)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))
    lims = [y_usd.min() * 0.8, y_usd.max() * 1.2]
    for ax, (name, p) in zip(axes.ravel(), preds.items()):
        ax.scatter(y_usd, p, alpha=0.65, edgecolor="k", linewidth=0.3, s=28)
        ax.plot(lims, lims, "crimson", ls="--", lw=1)
        r2 = df.loc[df.model == name, "R2_log"].iloc[0]
        ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
               title=f"{name}\nR²={r2:+.3f}", xlabel="Actual USD", ylabel="Predicted USD")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Actual vs predicted — out-of-fold predictions (dashed = perfect)", y=1.0)
    fig.tight_layout()
    fig.savefig(RESULTS / "actual_vs_predicted.png", dpi=125, bbox_inches="tight")
    print(f"\nSaved -> results/accuracy_comparison.csv, results/actual_vs_predicted.png")


if __name__ == "__main__":
    main()
