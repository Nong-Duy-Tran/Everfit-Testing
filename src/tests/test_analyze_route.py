"""Request-handling for /analyze that doesn't require the LLM.

The 400/404 paths short-circuit before any model call, and empty history raises
in the deterministic layer before generation — so these run offline.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_rejects_both_user_id_and_workouts(client):
    r = client.post(
        "/analyze",
        json={"user_id": "user_a", "workouts": [], "question": "my trends?"},
    )
    assert r.status_code == 400


def test_rejects_neither_user_id_nor_workouts(client):
    r = client.post("/analyze", json={"question": "my trends?"})
    assert r.status_code == 400


def test_unknown_user_is_404(client):
    r = client.post("/analyze", json={"user_id": "nobody", "question": "my trends?"})
    assert r.status_code == 404


def test_empty_inline_history_is_insufficient_data(client):
    r = client.post("/analyze", json={"workouts": [], "question": "my trends?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "insufficient_data"
    assert body["summary"] is None
