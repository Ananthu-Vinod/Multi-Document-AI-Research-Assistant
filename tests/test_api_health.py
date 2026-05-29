"""FastAPI health endpoint smoke test."""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Ask My Docs" in r.json()["name"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "chunk_count" in data
