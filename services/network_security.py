"""Outbound URL policy. Private model services require an operator allowlist."""
import ipaddress
import os
import socket
from urllib.parse import urlsplit


def validate_outbound_url(value: str, *, model=False) -> str:
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.fragment:
        raise ValueError("需要不含凭证的 HTTP(S) 地址")
    try:
        port = url.port or (443 if url.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("端口不合法") from exc
    origin = f"{url.scheme}://{url.hostname.lower()}:{port}"
    allowed = set()
    for entry in os.getenv("TAX_AGENT_MODEL_ALLOWED_ORIGINS", "").split(","):
        if entry.strip():
            item = urlsplit(entry.strip())
            allowed.add(f"{item.scheme}://{(item.hostname or '').lower()}:{item.port or (443 if item.scheme == 'https' else 80)}")
    if model and origin in allowed:
        return value
    if model and url.scheme != "https":
        raise ValueError("模型服务需使用 HTTPS；本地服务须由管理员在环境变量中显式允许该 origin")
    try:
        addresses = [ipaddress.ip_address(url.hostname)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(row[4][0]) for row in socket.getaddrinfo(url.hostname, port, type=socket.SOCK_STREAM)]
        except OSError as exc:
            raise ValueError("无法解析服务地址") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("禁止访问内网、环回、链路本地及保留地址")
    return value


def fetch_policy_response(session, url, timeout=30, limit=2 * 1024 * 1024):
    """Reject redirects and bound decoded bytes, including responses without length."""
    validate_outbound_url(url)
    session.trust_env = False
    response = session.get(url, timeout=timeout, allow_redirects=False, stream=True)
    try:
        if 300 <= response.status_code < 400:
            raise ValueError("政策源重定向被拒绝，请填写最终公开地址")
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > limit:
                raise ValueError("政策页面超过 2MB 上限")
            chunks.append(chunk)
        response._content = b"".join(chunks)
        response._content_consumed = True
        return response
    finally:
        response.close()
