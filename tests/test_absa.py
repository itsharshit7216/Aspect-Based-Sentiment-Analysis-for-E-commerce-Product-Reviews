import pytest
from src.absa_engine import ABSAEngine
from main import app
from fastapi.testclient import TestClient

@pytest.fixture
def engine():
    return ABSAEngine()

@pytest.fixture
def client():
    return TestClient(app)

def test_engine_initialization(engine):
    assert engine is not None

def test_analyze_sentence(engine):
    sentence = "The screen is bright but the battery is bad."
    result = engine.analyze_sentence(sentence)
    assert "sentence" in result
    assert "analysis" in result
    assert isinstance(result["analysis"], dict)

def test_fastapi_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_fastapi_analyze(client):
    payload = {"sentence": "The camera quality is superb."}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentence"] == payload["sentence"]
    assert "analysis" in data
