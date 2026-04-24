import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_needs_authorization_error_raised_when_no_token(tmp_path):
    """无 token 文件时应抛出 NeedsAuthorizationError。"""
    with patch.dict(os.environ, {"GOOGLE_TOKEN_FILE": str(tmp_path / "token.json")}):
        from auth.google_oauth import get_credentials, NeedsAuthorizationError
        with pytest.raises(NeedsAuthorizationError):
            get_credentials()


def test_get_auth_url_returns_url_and_state(tmp_path):
    """get_auth_url 应返回 (url, state) 元组，url 包含 accounts.google.com。"""
    creds_content = {
        "installed": {
            "client_id": "fake-id.apps.googleusercontent.com",
            "client_secret": "fake-secret",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    creds_file = tmp_path / "credentials.json"
    import json
    creds_file.write_text(json.dumps(creds_content))

    with patch.dict(os.environ, {
        "GOOGLE_CREDENTIALS_FILE": str(creds_file),
        "OAUTH_REDIRECT_URI": "http://localhost:8000/auth/google/callback",
    }):
        import importlib
        import auth.google_oauth as mod
        importlib.reload(mod)
        url, state = mod.get_auth_url()
        assert "accounts.google.com" in url
        assert isinstance(state, str) and len(state) > 0


def test_save_and_load_token(tmp_path):
    """token 写入后能用 Credentials.from_authorized_user_file 读取。"""
    from unittest.mock import MagicMock
    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"token": "fake", "refresh_token": "r", "token_uri": "https://oauth2.googleapis.com/token", "client_id": "c", "client_secret": "s", "scopes": []}'

    token_path = tmp_path / "token.json"
    with patch.dict(os.environ, {"GOOGLE_TOKEN_FILE": str(token_path)}):
        import importlib
        import auth.google_oauth as mod
        importlib.reload(mod)
        mod._save_token(fake_creds)
        assert token_path.exists()
