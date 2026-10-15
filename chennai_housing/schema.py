"""Input validation and Indian-style price formatting, shared by the web form and the API."""

from __future__ import annotations

from .simulate import FURNISHING, LOCALITY_RATE, PROPERTY_TYPES

NUMERIC_RULES = {
    "area_sqft": (300, 10_000, int),
    "bedrooms": (1, 8, int),
    "bathrooms": (1, 8, int),
    "age_years": (0, 80, int),
    "floor": (0, 60, int),
    "dist_main_road_m": (0, 5_000, int),
}
CHOICES = {
    "locality": list(LOCALITY_RATE),
    "property_type": list(PROPERTY_TYPES),
    "furnishing": list(FURNISHING),
}
BOOLEANS = ["parking", "near_metro"]
FIELDS = list(CHOICES) + list(NUMERIC_RULES) + BOOLEANS


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def validate(payload: dict) -> tuple[dict, dict]:
    """Return (clean_values, errors). Errors map field name to a readable message."""
    clean, errors = {}, {}
    for field, options in CHOICES.items():
        value = payload.get(field)
        if value not in options:
            errors[field] = f"Choose one of: {', '.join(options)}."
        else:
            clean[field] = value
    for field, (low, high, cast) in NUMERIC_RULES.items():
        raw = payload.get(field)
        try:
            value = cast(float(raw))
        except (TypeError, ValueError):
            errors[field] = "Enter a number."
            continue
        if not low <= value <= high:
            errors[field] = f"Must be between {low:,} and {high:,}."
        else:
            clean[field] = value
    for field in BOOLEANS:
        clean[field] = _to_bool(payload.get(field, False))
    if "bathrooms" in clean and "bedrooms" in clean and clean["bathrooms"] > clean["bedrooms"] + 2:
        errors["bathrooms"] = "That's unusually many bathrooms for the number of bedrooms."
    return clean, errors


def format_inr(amount: float) -> str:
    """Format rupees the Indian way: 85.4 L or 1.25 Cr."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    return f"₹{amount / 1e5:.1f} L"
