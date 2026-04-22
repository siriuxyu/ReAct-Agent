import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


def test_auth_google_redirect():
    """GET /auth/google 应重定向到 Google 授权页。"""
    with patch("auth.google_oauth.get_auth_url", return_value=("https://accounts.google.com/fake", "state123")):
        from server import api
        client = TestClient(api, follow_redirects=False)
        resp = client.get("/auth/google")
        assert resp.status_code == 307
        assert "accounts.google.com" in resp.headers["location"]


def test_auth_google_callback_success():
    """GET /auth/google/callback?code=X&state=Y 成功时返回 200。"""
    from unittest.mock import MagicMock
    fake_creds = MagicMock()
    with patch("auth.google_oauth.exchange_code", return_value=fake_creds):
        from server import api
        client = TestClient(api)
        resp = client.get("/auth/google/callback?code=fake_code&state=fake_state")
        assert resp.status_code == 200
        assert "success" in resp.json()["message"].lower()


def test_auth_google_callback_invalid_state():
    """state 不匹配时返回 400。"""
    with patch("auth.google_oauth.exchange_code", side_effect=ValueError("Unknown OAuth state")):
        from server import api
        client = TestClient(api)
        resp = client.get("/auth/google/callback?code=x&state=bad")
        assert resp.status_code == 400
