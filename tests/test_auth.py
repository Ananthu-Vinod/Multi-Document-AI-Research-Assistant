"""Authentication API endpoints test suite."""

import uuid
import pytest
from fastapi.testclient import TestClient
from app import app
from database import Base, engine, init_db


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure clean database tables before running tests."""
    init_db()


def test_auth_registration_and_login():
    with TestClient(app) as client:
        unique_id = str(uuid.uuid4())[:8]
        email = f"user_{unique_id}@example.com"
        password = "secretpassword123"

        # Register
        r = client.post("/auth/register", json={"email": email, "password": password, "full_name": "Test User"})
        assert r.status_code == 201
        data = r.json()
        assert "access_token" in data
        token = data["access_token"]

        # Login
        r_login = client.post("/auth/login", data={"username": email, "password": password})
        assert r_login.status_code == 200
        assert "access_token" in r_login.json()

        # Get profile
        r_me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r_me.status_code == 200
        profile = r_me.json()
        assert profile["email"] == email
        assert profile["full_name"] == "Test User"


def test_auth_unauthorized():
    with TestClient(app) as client:
        r = client.get("/auth/me")
        assert r.status_code == 401
