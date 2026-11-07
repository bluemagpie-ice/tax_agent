"""Issue short-lived Bearer sessions for the existing operator-managed identities."""

import hmac
import os
import re
import secrets
import time

SESSION_TTL = 8 * 60 * 60
IDENTITIES = {"user": ("user_001", "张三"), "admin": ("user_000", "开发者")}
_SESSION_FORMAT = re.compile(r"(v1\.(user|admin)\.([0-9]{1,12})\.[A-Za-z0-9_-]{22})\.([a-f0-9]{64})")


def _credentials():
    user = os.getenv("AUTH_USER_TOKEN", "")
    admin = os.getenv("DEVELOPER_TOKEN") or os.getenv("AUTH_ADMIN_TOKEN", "")
    return {"user": user, "admin": admin} if user != admin else {}


def _signature(payload, credential, signing_key):
    # Rotating either the login credential or the server key revokes existing sessions.
    key = hmac.digest(signing_key.encode(), credential.encode(), "sha256")
    return hmac.digest(key, payload.encode(), "sha256").hex()


def issue_session(credential):
    signing_key = os.getenv("AUTH_SESSION_SECRET", "")
    if not signing_key:
        raise RuntimeError("未配置会话签发密钥，请联系服务管理员。")
    for role, expected in _credentials().items():
        if expected and hmac.compare_digest(credential.encode(), expected.encode()):
            expires = int(time.time()) + SESSION_TTL
            payload = f"v1.{role}.{expires}.{secrets.token_urlsafe(16)}"
            return f"{payload}.{_signature(payload, expected, signing_key)}", expires
    raise ValueError("登录凭证无效，请核对服务端配置后重试。")


def session_role(token):
    match = _SESSION_FORMAT.fullmatch(token)
    if not match:
        return None
    payload, role, expires, signature = match.groups()
    credential = _credentials().get(role)
    signing_key = os.getenv("AUTH_SESSION_SECRET", "")
    if credential and signing_key and int(expires) > time.time():
        if hmac.compare_digest(signature, _signature(payload, credential, signing_key)):
            return role
    return None
