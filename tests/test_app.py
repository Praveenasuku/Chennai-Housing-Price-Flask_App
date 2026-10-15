import pytest

from app.app import create_app
from chennai_housing.schema import format_inr, validate
from chennai_housing.simulate import simulate

VALID = {
    "locality": "Adyar", "property_type": "Apartment", "furnishing": "Furnished",
    "area_sqft": 1500, "bedrooms": 3, "bathrooms": 2, "age_years": 4, "floor": 6,
    "dist_main_road_m": 200, "parking": True, "near_metro": False,
}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_simulator_is_reproducible():
    a, b = simulate(n=200, seed=1), simulate(n=200, seed=1)
    assert a.equals(b)
    assert (a["price_inr"] > 0).all()


def test_format_inr():
    assert format_inr(8_540_000) == "₹85.4 L"
    assert format_inr(12_500_000) == "₹1.25 Cr"


def test_validate_reports_each_bad_field():
    _, errors = validate({**VALID, "locality": "Mumbai", "area_sqft": "big", "bedrooms": 0})
    assert set(errors) == {"locality", "area_sqft", "bedrooms"}


def test_api_predict_returns_price_and_range(client):
    body = client.post("/api/predict", json=VALID).get_json()
    low, high = body["range_inr"]
    assert low < body["price_inr"] < high
    assert body["formatted"].startswith("₹")


def test_prime_locality_costs_more_than_suburb(client):
    adyar = client.post("/api/predict", json=VALID).get_json()["price_inr"]
    karapakkam = client.post("/api/predict", json={**VALID, "locality": "Karapakkam"}).get_json()["price_inr"]
    assert adyar > karapakkam * 1.5


def test_bigger_home_costs_more(client):
    small = client.post("/api/predict", json={**VALID, "area_sqft": 900}).get_json()["price_inr"]
    large = client.post("/api/predict", json={**VALID, "area_sqft": 1800}).get_json()["price_inr"]
    assert large > small


def test_api_rejects_bad_input(client):
    response = client.post("/api/predict", json={**VALID, "furnishing": "Luxury"})
    assert response.status_code == 400
    assert "furnishing" in response.get_json()["fields"]
    assert client.post("/api/predict", data="not json").status_code == 400


def test_form_page_and_submission(client):
    assert client.get("/").status_code == 200
    form = {k: ("1" if v is True else v) for k, v in VALID.items() if v is not False}
    html = client.post("/", data=form).get_data(as_text=True)
    assert "Estimated price" in html and "Likely range" in html


def test_form_shows_validation_errors(client):
    response = client.post("/", data={**{k: str(v) for k, v in VALID.items()}, "area_sqft": "50"})
    assert response.status_code == 400
    assert "Must be between" in response.get_data(as_text=True)


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
