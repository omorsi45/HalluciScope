import io
import json
import logging
import re

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel
from starlette.requests import Request

from backend.api.app import create_app
from backend.api.errors import ApiError
from backend.config import Settings
from backend.obs.logging import JsonFormatter
from backend.obs.metrics import HTTP_REQUESTS

ORIGIN = "http://localhost:5173"
JSON_LIMIT = 1_048_576


class _Typed(BaseModel):
    n: int


@pytest.fixture
def app():
    application = create_app(Settings(cors_origins=[ORIGIN]))

    @application.get("/api/v1/_apierr")
    async def apierr():
        raise ApiError(409, "conflict_x", "Conflict")

    @application.get("/api/v1/_boom")
    async def boom():
        raise RuntimeError("SECRET-BOOM")

    @application.post("/api/v1/_typed")
    async def typed(body: _Typed):
        return {"n": body.n}

    @application.post("/api/v1/_echo")
    async def echo(request: Request):
        return {"len": len(await request.body())}

    return application


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_problem_json_shape(client):
    response = await client.get("/api/v1/_apierr")
    assert response.status_code == 409
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "/errors/conflict_x",
        "title": "Conflict",
        "status": 409,
        "detail": None,
        "code": "conflict_x",
    }


async def test_unhandled_error_is_generic(client):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        response = await client.get("/api/v1/_boom")
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["title"] == "Internal server error"
    assert "SECRET-BOOM" not in response.text
    logged = stream.getvalue()
    assert "unhandled error" in logged
    assert "SECRET-BOOM" not in logged
    assert response.headers["x-request-id"]


async def test_validation_error_hides_input(client):
    response = await client.post("/api/v1/_typed", json={"n": "SECRET-INPUT"})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert "SECRET-INPUT" not in response.text


async def test_unknown_route_is_problem_not_found(client):
    response = await client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "not_found"


async def test_request_id_echo_and_replace(client):
    echoed = await client.get("/api/v1/_apierr", headers={"X-Request-ID": "abc-1"})
    assert echoed.headers["x-request-id"] == "abc-1"
    replaced = await client.get("/api/v1/_apierr", headers={"X-Request-ID": "bad id!"})
    assert re.fullmatch(r"[0-9a-f]{32}", replaced.headers["x-request-id"])


async def test_request_updates_metrics_with_route_template(client):
    key = ("/api/v1/_apierr", "GET", "409")
    before = HTTP_REQUESTS.labels(route=key[0], method=key[1], status=key[2]).value
    await client.get("/api/v1/_apierr")
    after = HTTP_REQUESTS.labels(route=key[0], method=key[1], status=key[2]).value
    assert after == before + 1


async def test_security_headers_on_api(client):
    response = await client.get("/api/v1/_apierr")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"
    assert (
        response.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    )


async def test_csp_only_on_api_paths(client):
    response = await client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in response.headers


async def test_cors_allows_listed_origin_only(client):
    preflight = {"Access-Control-Request-Method": "POST"}
    allowed = await client.options("/api/v1/_echo", headers={"Origin": ORIGIN, **preflight})
    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert allowed.headers["access-control-allow-credentials"] == "true"
    denied = await client.options(
        "/api/v1/_echo", headers={"Origin": "http://evil.example", **preflight}
    )
    assert "access-control-allow-origin" not in denied.headers


async def test_body_limit_by_content_length(client):
    response = await client.post("/api/v1/_echo", content=b"x" * (JSON_LIMIT + 1))
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


async def test_body_limit_chunked(client):
    chunk = b"x" * 65_536

    async def body():
        for _ in range(32):  # 2 MB total, no Content-Length
            yield chunk

    request = client.build_request("POST", "/api/v1/_echo", content=body())
    assert "content-length" not in request.headers
    response = await client.send(request)
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


async def test_chunked_overflow_hides_app_error_behind_413(client):
    """The typed route would turn a failed body read into 400; the limit must win."""

    async def body():
        for _ in range(32):
            yield b"x" * 65_536

    request = client.build_request("POST", "/api/v1/_typed", content=body())
    response = await client.send(request)
    assert response.status_code == 413


async def test_small_body_passes(client):
    response = await client.post("/api/v1/_echo", content=b"0123456789")
    assert response.status_code == 200
    assert response.json() == {"len": 10}


async def test_multipart_uses_upload_limit(client):
    payload = b"x" * (JSON_LIMIT + 1)
    response = await client.post(
        "/api/v1/_echo",
        content=payload,
        headers={"Content-Type": "multipart/form-data; boundary=b"},
    )
    assert response.status_code == 200


async def test_legacy_health_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert json.loads(response.text) == {"status": "ok"}


async def test_overflow_after_response_started_closes_by_raising():
    from backend.api.middleware import BodySizeLimitMiddleware, _BodyTooLarge

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        while True:
            await receive()

    sent = []
    messages = iter([{"type": "http.request", "body": b"xxxx", "more_body": True}] * 3)

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message["type"])

    middleware = BodySizeLimitMiddleware(app, json_limit=10)
    scope = {"type": "http", "method": "POST", "path": "/x", "headers": []}
    with pytest.raises(_BodyTooLarge):
        await middleware(scope, receive, send)
    assert sent == ["http.response.start"]


async def test_unhandled_error_carries_security_and_cors_headers(client):
    response = await client.get("/api/v1/_boom", headers={"Origin": ORIGIN})
    assert response.status_code == 500
    assert response.headers["x-request-id"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"
    assert (
        response.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    )
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "x-request-id" in response.headers["access-control-expose-headers"].lower()


def _request_metric_methods() -> set[str]:
    return {key[1] for key in HTTP_REQUESTS._children}


async def test_unknown_methods_share_one_metric_label(client):
    for index in range(5):
        await client.request(f"BREW{index}", "/api/v1/_apierr")
    methods = _request_metric_methods()
    assert not {m for m in methods if m.startswith("BREW")}
    assert "OTHER" in methods
    assert methods <= {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "OTHER"}


async def _asgi_get(app, headers):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/v1/_apierr",
        "raw_path": b"/api/v1/_apierr",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    await app(scope, receive, send)
    return sent


async def test_request_id_with_trailing_newline_is_replaced(app):
    sent = await _asgi_get(app, [(b"x-request-id", b"abc-1\n")])
    start = next(m for m in sent if m["type"] == "http.response.start")
    value = dict(start["headers"])[b"x-request-id"].decode()
    assert re.fullmatch(r"[0-9a-f]{32}", value)


@pytest.mark.parametrize("declared", [b"\xb2", b"1" * 20, b"-5", b"1 2", b""])
async def test_odd_content_length_is_ignored_not_fatal(declared):
    from backend.api.middleware import BodySizeLimitMiddleware

    calls = []

    async def inner(scope, receive, send):
        calls.append(scope["path"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/x",
        "headers": [(b"content-length", declared)],
    }
    await BodySizeLimitMiddleware(inner, json_limit=10)(scope, receive, send)
    assert calls == ["/x"]
    assert sent[0]["status"] == 204


def test_create_app_rejects_wildcard_cors_origin():
    with pytest.raises(ValueError, match="cors"):
        create_app(Settings(cors_origins=["*"]))
    with pytest.raises(ValueError, match="cors"):
        create_app(Settings(cors_origins=[ORIGIN, "*"]))
