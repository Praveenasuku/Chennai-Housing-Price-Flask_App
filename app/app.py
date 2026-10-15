"""Flask web app and JSON API for Chennai house price estimates.

Run locally:
    flask --app app.app run --debug
Production (see Dockerfile):
    gunicorn -b 0.0.0.0:8000 "app.app:create_app()"
"""

from __future__ import annotations

import json

from flask import Flask, jsonify, render_template, request

from chennai_housing import __version__
from chennai_housing.model import METRICS_PATH, load_model, predict
from chennai_housing.schema import CHOICES, FIELDS, format_inr, validate

DEFAULTS = {
    "locality": "Velachery", "property_type": "Apartment", "furnishing": "Semi-furnished",
    "area_sqft": 1200, "bedrooms": 2, "bathrooms": 2, "age_years": 5, "floor": 3,
    "dist_main_road_m": 300, "parking": True, "near_metro": True,
}


def create_app() -> Flask:
    app = Flask(__name__)
    load_model()  # fail fast at startup if the model file is missing

    @app.get("/")
    def form():
        return render_template("index.html", values=DEFAULTS, choices=CHOICES, errors={}, result=None)

    @app.post("/")
    def form_submit():
        raw = {f: request.form.get(f) for f in FIELDS}
        values, errors = validate(raw)
        if errors:
            return render_template("index.html", values={**DEFAULTS, **raw}, choices=CHOICES, errors=errors, result=None), 400
        result = predict(values)
        result["display"] = {
            "price": format_inr(result["price_inr"]),
            "low": format_inr(result["range_inr"][0]),
            "high": format_inr(result["range_inr"][1]),
            "per_sqft": f"₹{result['price_per_sqft']:,}",
        }
        return render_template("index.html", values=values, choices=CHOICES, errors={}, result=result)

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object with the property details."}), 400
        values, errors = validate(payload)
        if errors:
            return jsonify({"error": "Invalid input", "fields": errors}), 400
        result = predict(values)
        result["formatted"] = format_inr(result["price_inr"])
        return jsonify(result)

    @app.get("/api/options")
    def api_options():
        return jsonify(CHOICES)

    @app.get("/health")
    def health():
        metrics = json.loads(METRICS_PATH.read_text())
        return jsonify({"status": "ok", "version": __version__, "model": metrics["selected_model"],
                        "test_mape": metrics["test"]["mape"]})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
