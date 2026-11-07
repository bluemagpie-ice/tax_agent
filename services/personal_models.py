"""Tenant-scoped model storage and safe outbound clients for personal endpoints."""

from contextlib import contextmanager

import httpx
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from services.config_manager import validate_api_url
from services.network_security import resolve_model_address
from services.security import require_tenant


def normalize_model_url(value):
    url = validate_api_url(value.strip())
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise ValueError("请填写有效的 API 基础地址。") from exc
    if (not parsed.host or any(c.isspace() or c == "\\" for c in url)
            or parsed.path.rstrip("/").endswith(("/chat/completions", "/models", "/responses"))):
        raise ValueError("请填写有效的 API 基础地址。")
    return str(parsed)


def get_personal_model():
    from database.models import PersonalModelConfig as Config, SessionLocal

    with SessionLocal() as db:
        row = db.get(Config, require_tenant())
        return {"model": row.model, "base_url": row.base_url, "api_key": row.api_key} if row else None


def personal_model_key(base_url, supplied_key):
    if supplied_key:
        return supplied_key
    saved = get_personal_model()
    if saved and saved["base_url"] == base_url:
        return saved["api_key"]
    return ""


def save_personal_model(model, base_url, api_key):
    from database.models import PersonalModelConfig as Config, SessionLocal

    owner = require_tenant()
    statement = update(Config).where(Config.owner_id == owner)
    if api_key:
        values = {"model": model, "base_url": base_url, "api_key": api_key}
        try:
            with SessionLocal.begin() as db:
                if not db.execute(statement.values(**values)).rowcount:
                    db.add(Config(owner_id=owner, **values))
        except IntegrityError:
            # A concurrent first save may have created the row.
            with SessionLocal.begin() as db:
                if not db.execute(statement.values(**values)).rowcount:
                    raise
    else:
        with SessionLocal.begin() as db:
            updated = db.execute(statement.where(Config.base_url == base_url).values(model=model)).rowcount
            if not updated:
                raise ValueError("请填写该 API 地址对应的 Key。")


@contextmanager
def personal_model_client(base_url, timeout):
    """Pin each connection to a checked IP while retaining the original TLS name and Host."""
    origin = httpx.URL(base_url)

    def pin(request):
        if (request.url.scheme, request.url.host, request.url.port) != (origin.scheme, origin.host, origin.port):
            raise ValueError("模型请求超出配置的 API 来源。")
        address = resolve_model_address(str(request.url))
        request.extensions["sni_hostname"] = origin.raw_host.decode("ascii")
        request.url = request.url.copy_with(host=address)

    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False,
                      event_hooks={"request": [pin]}) as client:
        yield client
