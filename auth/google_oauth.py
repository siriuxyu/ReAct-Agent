import os
import secrets
import time
from pathlib import Path

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import Flow
except ImportError:
    Credentials = None
    Request = None
    Flow = None

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/tasks",
]

_CREDENTIALS_FILE = lambda: os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
_TOKEN_FILE = lambda: os.environ.get("GOOGLE_TOKEN_FILE", "auth/token.json")
_REDIRECT_URI = lambda: os.environ.get(
    "OAUTH_REDIRECT_URI", "http://localhost:8000/auth/google/callback"
)

_pending_flows: dict[str, tuple] = {}  # state -> (flow, created_at)


class NeedsAuthorizationError(Exception):
    pass


def get_credentials() -> Credentials:
    if Credentials is None or Request is None:
        raise NeedsAuthorizationError(
            "Google OAuth dependencies are not installed. Please install the Google auth libraries."
        )
    token_path = Path(_TOKEN_FILE())
    if not token_path.exists():
        raise NeedsAuthorizationError("No token found. Please authorize via /auth/google")
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)
        return creds
    raise NeedsAuthorizationError("Token invalid. Please re-authorize via /auth/google")


def get_auth_url() -> tuple[str, str]:
    if Flow is None:
        raise RuntimeError("google-auth-oauthlib is not installed")
    now = time.time()
    expired = [s for s, (_, t) in _pending_flows.items() if now - t > 600]
    for s in expired:
        _pending_flows.pop(s, None)
    flow = Flow.from_client_secrets_file(
        _CREDENTIALS_FILE(), scopes=SCOPES, redirect_uri=_REDIRECT_URI()
    )
    state = secrets.token_urlsafe(16)
    auth_url, _ = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", state=state, prompt="consent"
    )
    _pending_flows[state] = (flow, now)
    return auth_url, state


def exchange_code(code: str, state: str) -> Credentials:
    flow, _ = _pending_flows.pop(state, (None, None))
    if flow is None:
        raise ValueError(f"Unknown OAuth state: {state}")
    flow.fetch_token(code=code)
    creds = flow.credentials
    _save_token(creds)
    return creds


def _save_token(creds: Credentials) -> None:
    token_path = Path(_TOKEN_FILE())
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
