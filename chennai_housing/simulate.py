"""Generate a synthetic Chennai housing dataset.

IMPORTANT: this is simulated data. The base rates and premiums below are
illustrative assumptions chosen to produce realistic-looking relationships
(prime central areas cost more per square foot than the southern suburbs,
new buildings cost more than old ones, and so on). They are not market data
and must not be used to value real property.

The point of the project is the engineering around a price model: validation,
training, uncertainty ranges and serving through a web app and JSON API.
Replace `data/chennai_housing.csv` with real listings that have the same
columns and retrain to use it for real.

Usage:
    python -m chennai_housing.simulate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "chennai_housing.csv"

# Illustrative base rate in rupees per square foot for a new apartment.
LOCALITY_RATE = {
    "T Nagar": 13_500,
    "Adyar": 13_000,
    "Anna Nagar": 12_000,
    "KK Nagar": 9_500,
    "Velachery": 8_500,
    "Sholinganallur": 7_000,
    "Chrompet": 6_000,
    "Karapakkam": 5_500,
}
PROPERTY_TYPES = {"Apartment": 1.00, "Independent House": 1.18, "Villa": 1.32}
FURNISHING = {"Unfurnished": 1.00, "Semi-furnished": 1.04, "Furnished": 1.09}


def simulate(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locality = rng.choice(list(LOCALITY_RATE), size=n, p=[0.1, 0.11, 0.13, 0.12, 0.16, 0.14, 0.12, 0.12])
    ptype = rng.choice(list(PROPERTY_TYPES), size=n, p=[0.68, 0.24, 0.08])

    size_mean = np.select([ptype == "Apartment", ptype == "Independent House"], [1150, 1700], 2600)
    sqft = np.clip(rng.lognormal(np.log(size_mean), 0.32), 400, 6000).round(-1)
    bedrooms = np.clip(np.round(sqft / 520 + rng.normal(0, 0.4, n)), 1, 5).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5).astype(int)
    age = np.clip(rng.gamma(2.0, 6.0, n), 0, 45).round().astype(int)
    floor = np.where(ptype == "Apartment", rng.integers(0, 15, n), 0)
    parking = rng.random(n) < np.where(ptype == "Apartment", 0.7, 0.92)
    near_metro = rng.random(n) < np.where(np.isin(locality, ["T Nagar", "Anna Nagar", "KK Nagar", "Velachery", "Chrompet"]), 0.55, 0.2)
    furnishing = rng.choice(list(FURNISHING), size=n, p=[0.55, 0.32, 0.13])
    dist_main_road_m = np.clip(rng.exponential(350, n), 10, 2500).round()

    rate = np.array([LOCALITY_RATE[x] for x in locality], dtype=float)
    price = (
        sqft
        * rate
        * np.array([PROPERTY_TYPES[x] for x in ptype])
        * np.array([FURNISHING[x] for x in furnishing])
        * (1 - 0.009 * age)                       # about 0.9% depreciation per year
        * np.where(near_metro, 1.07, 1.0)         # metro access premium
        * np.where(parking, 1.04, 1.0)
        * (1 + 0.004 * np.minimum(floor, 10))     # higher floors, slightly more
        * (1 - 0.00004 * dist_main_road_m)        # farther from the main road, slightly less
        * rng.lognormal(0, 0.09, n)               # everything the data doesn't capture
    )

    return pd.DataFrame({
        "locality": locality,
        "property_type": ptype,
        "area_sqft": sqft.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "floor": floor,
        "parking": parking,
        "near_metro": near_metro,
        "furnishing": furnishing,
        "dist_main_road_m": dist_main_road_m.astype(int),
        "price_inr": (price / 1000).round() * 1000,
    })


def main() -> None:
    df = simulate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
