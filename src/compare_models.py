"""Compare every practical regression algorithm on the condominium dataset.

Three things this does that a naive sweep would not:

1. **Imputation lives inside the CV pipeline.** Filling missing values before
   splitting leaks test-fold information into training. With 50% of bathrooms
   missing, that is not a small effect.

2. **Repeated CV, with the spread reported.** n=87 is small enough that a
   single 5-fold split is mostly noise. Each model is scored over many splits
   and reported as mean +/- std, so it is visible when two models are not
   meaningfully different.

3. **Two validation schemes.** Coordinates in this data are neighbourhood
   centroids: 16 unique locations across 87 rows. A random split puts the same
   location in train and test, letting tree models recall a neighbourhood
   average rather than value a property. Grouped splits hold out whole
   locations and answer a different, harder question.

Read both columns. Random-CV answers "how well does this value a property in a
neighbourhood I already know?"; grouped-CV answers "...in a neighbourhood I
have never seen?".
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.model_selection import GroupShuffleSplit, RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(
    os.environ.get(
        "CONDO_DATA",
        Path(__file__).resolve().parent.parent / "data" / "condominiums cleaned (1).xlsx",
    )
)
OUT = ROOT / "results" / "model_comparison_condominiums.csv"

RANDOM_STATE = 42


def load() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_excel(DATA)
    # Rows with no target teach nothing and must never be imputed — the target
    # is what we are trying to learn.
    d = df[df["Price USD"].notna()].copy()

    y = np.log(d["Price USD"])
    features = ["Bedrooms", "Bathrooms", "Latitude", "Longitude"] + [
        c for c in d.columns if "score" in c.lower()
    ]
    X = d[features]
    groups = d["Latitude"].astype(str) + "_" + d["Longitude"].astype(str)
    return X, y, groups


def models() -> dict:
    rs = RANDOM_STATE
    return {
        "Baseline (predict mean)": DummyRegressor(strategy="mean"),
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(),
        "Lasso": LassoCV(random_state=rs, max_iter=5000),
        "ElasticNet": ElasticNetCV(random_state=rs, max_iter=5000),
        "BayesianRidge": BayesianRidge(),
        "Huber (robust)": HuberRegressor(max_iter=1000),
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
        "SVR (rbf)": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=rs),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=rs),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=rs),
        "Gradient Boosting": GradientBoostingRegressor(random_state=rs),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=rs),
        "AdaBoost": AdaBoostRegressor(random_state=rs),
        "XGBoost": _xgb(rs),
        "LightGBM": _lgbm(rs),
        "CatBoost": _catboost(rs),
        "MLP (neural net)": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=3000, random_state=rs
        ),
    }


def _xgb(rs):
    from xgboost import XGBRegressor

    return XGBRegressor(n_estimators=300, random_state=rs, verbosity=0)


def _lgbm(rs):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=300, random_state=rs, verbose=-1)


def _catboost(rs):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(iterations=300, random_seed=rs, verbose=0)


def make_pipeline(model) -> Pipeline:
    """Impute then scale, both fitted per fold so nothing leaks across splits."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def score(model, X, y, groups) -> dict:
    pipe = make_pipeline(model)

    random_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    random_scores = cross_val_score(pipe, X, y, cv=random_cv, scoring="r2")

    grouped_cv = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=RANDOM_STATE)
    grouped_scores = cross_val_score(
        pipe, X, y, cv=grouped_cv, groups=groups, scoring="r2"
    )

    return {
        "random_mean": random_scores.mean(),
        "random_std": random_scores.std(),
        "grouped_mean": grouped_scores.mean(),
        "grouped_std": grouped_scores.std(),
    }


def main() -> None:
    X, y, groups = load()
    print(f"Condominium dataset: {len(X)} usable rows, {X.shape[1]} features, "
          f"{groups.nunique()} distinct locations\n")

    rows = []
    for name, model in models().items():
        try:
            result = score(model, X, y, groups)
        except Exception as exc:  # keep the sweep going if one model fails
            print(f"  {name:<24} FAILED: {type(exc).__name__}")
            continue
        rows.append({"model": name, **result})
        print(
            f"  {name:<24} random {result['random_mean']:+.3f}+/-{result['random_std']:.3f}"
            f"   grouped {result['grouped_mean']:+.3f}+/-{result['grouped_std']:.3f}"
        )

    df = pd.DataFrame(rows).sort_values("grouped_mean", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    print("Ranked by GROUPED CV (generalising to unseen locations)")
    print("=" * 78)
    print(
        df.to_string(
            index=False,
            formatters={c: "{:+.3f}".format for c in df.columns if c != "model"},
        )
    )
    print(f"\nSaved -> results/{OUT.name}")


if __name__ == "__main__":
    main()
