"""Train the price model and serve predictions with an 80% price range.

The range uses split-conformal prediction: we take the model's cross-validated
errors (in log price) and use their 10th and 90th percentiles as the range
around each new estimate. Unlike quantile regression, this is calibrated by
construction, and the test set confirms roughly 80% coverage.

Usage:
    python -m chennai_housing.model
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from . import __version__  # noqa: E402
from .schema import BOOLEANS, CHOICES, NUMERIC_RULES  # noqa: E402
from .simulate import DATA_PATH, ROOT  # noqa: E402

MODEL_PATH = ROOT / "models" / "price_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"

CATEGORICAL = list(CHOICES)
NUMERIC = list(NUMERIC_RULES) + BOOLEANS
FEATURES = CATEGORICAL + NUMERIC


def _prep(scale: bool) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
    ])


def candidates() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([("prep", _prep(True)), ("reg", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([("prep", _prep(False)), ("reg", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1))]),
        "hist_gradient_boosting": Pipeline([("prep", _prep(False)), ("reg", HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, random_state=42))]),
    }


def _scores(y_true, y_pred) -> dict:
    return {
        "mae_inr": round(float(mean_absolute_error(y_true, y_pred))),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train(data_path: Path = DATA_PATH) -> dict:
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], np.log(df["price_inr"].to_numpy())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    cv = KFold(5, shuffle=True, random_state=42)
    comparison, cv_log_preds = {}, {}
    for name, model in candidates().items():
        cv_log_preds[name] = cross_val_predict(model, X_tr, y_tr, cv=cv)
        pred = np.exp(cv_log_preds[name])
        comparison[name] = _scores(np.exp(y_tr), pred)
        print(f"{name:24s} CV MAPE {comparison[name]['mape']:.3%}")
    best_name = min(comparison, key=lambda n: comparison[n]["mape"])

    best = candidates()[best_name].fit(X_tr, y_tr)
    test_pred = np.exp(best.predict(X_te))
    residuals = y_tr - cv_log_preds[best_name]
    res_low, res_high = (float(q) for q in np.quantile(residuals, [0.1, 0.9]))
    low, high = test_pred * np.exp(res_low), test_pred * np.exp(res_high)
    coverage = float(np.mean((np.exp(y_te) >= low) & (np.exp(y_te) <= high)))

    metrics = {
        "version": __version__,
        "data": "synthetic (see chennai_housing/simulate.py)",
        "rows": int(len(df)),
        "cv_comparison": comparison,
        "selected_model": best_name,
        "test": _scores(np.exp(y_te), test_pred),
        "range_80pct_coverage": round(coverage, 4),
        "range_multipliers": [round(float(np.exp(res_low)), 4), round(float(np.exp(res_high)), 4)],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"point": best, "res_low": res_low, "res_high": res_high, "features": FEATURES,
                 "version": __version__}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    ax.scatter(test_pred / 1e5, np.exp(y_te) / 1e5, s=5, alpha=0.4, color="#1f6feb")
    lim = [0, np.percentile(np.exp(y_te) / 1e5, 99.5)]
    ax.plot(lim, lim, color="#d1242f", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted price (₹ lakh)")
    ax.set_ylabel("Actual price (₹ lakh)")
    ax.set_title("Test set: predicted vs actual")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual.png")
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    return metrics


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m chennai_housing.model")
    return joblib.load(path)


def predict(values: dict) -> dict:
    """Point estimate plus an 80% range for one validated property."""
    bundle = load_model()
    row = pd.DataFrame([values])[bundle["features"]]
    point = float(np.exp(bundle["point"].predict(row)[0]))
    low, high = point * np.exp(bundle["res_low"]), point * np.exp(bundle["res_high"])
    return {
        "price_inr": round(point, -3),
        "range_inr": [round(low, -3), round(high, -3)],
        "price_per_sqft": round(point / values["area_sqft"]),
    }


if __name__ == "__main__":
    train()
