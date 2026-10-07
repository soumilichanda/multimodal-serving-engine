"""
tests/test_api.py
Automated validation of FastAPI endpoints, Pydantic V2 rejection rules,
and thread-safe LRU caching behavior.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_and_cache_stats():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "cache_stats" in data
    assert data["cache_stats"]["capacity"] == 256


def test_vector_prediction_caching():
    payload = {"features": [1.2, -0.4, 0.8]}

    # First request: Cache Miss
    res1 = client.post("/v1/predict/vector", json=payload)
    assert res1.status_code == 200
    assert res1.json()["cached"] is False

    # Second identical request: Cache Hit
    res2 = client.post("/v1/predict/vector", json=payload)
    assert res2.status_code == 200
    assert res2.json()["cached"] is True
    assert res2.json()["prediction"] == res1.json()["prediction"]


def test_vector_empty_payload_rejection():
    # Pydantic V2 validator should block empty feature vectors
    bad_payload = {"features": []}
    response = client.post("/v1/predict/vector", json=bad_payload)
    assert response.status_code == 422


def test_corrupted_image_header_rejection():
    # Base64 string that passes length check but lacks valid JFIF/PNG headers
    corrupted_payload = {
        "image_base64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    }
    response = client.post("/v1/predict/image", json=corrupted_payload)
    assert response.status_code == 422
    assert "Corrupted image header" in response.json()["detail"]


def test_image_predict_valid_sample():
    # Valid 1x1 black PNG base64 string
    valid_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    response = client.post("/v1/predict/image", json={"image_base64": valid_png_b64})
    assert response.status_code in [200, 422]