"""Pure ASGI middleware.

BaseHTTPMiddleware is deliberately not used: it buffers streaming responses and does not
propagate contextvars reliably.
"""

import logging
import re
import time
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.api.errors import problem_response, unhandled_error_response
from backend.obs.logging import bind
from backend.obs.metrics import HTTP_DURATION, HTTP_REQUESTS

logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
_MAX_CONTENT_LENGTH_DIGITS = 19


def _method_label(method: str) -> str:
    """Bound the metric label: the method is client-controlled, so unknown ones share a value."""
    return method if method in _KNOWN_METHODS else "OTHER"


class _BodyTooLarge(Exception):
    """Raised inside the wrapped receive to abort the app once the limit is crossed."""


class UnhandledErrorMiddleware:
    """Innermost: turn an unhandled exception into the generic 500 problem response.

    Sitting inside SecurityHeaders and CORS means the 500 passes back out through them and
    carries their headers (and the request id), so a cross-origin SPA can read it. The
    framework's own outer error handler would bypass all of them.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except _BodyTooLarge:
            raise
        except Exception:
            if started:
                raise
            await unhandled_error_response()(scope, receive, send_wrapper)


class RequestContextMiddleware:
    """Request id, structured access log and HTTP metrics. Outermost middleware."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).get("x-request-id")
        request_id = supplied if supplied and _REQUEST_ID_RE.fullmatch(supplied) else uuid4().hex
        status = 500
        started = False
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        with bind(request_id=request_id):
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception:
                if started:
                    raise
                # Fallback only: UnhandledErrorMiddleware normally answers first, inside CORS
                # and SecurityHeaders. This 500 carries the request id but not their headers.
                await unhandled_error_response()(scope, receive, send_wrapper)
            finally:
                duration = time.perf_counter() - start
                route_obj = scope.get("route")
                route = getattr(route_obj, "path", None) or "unmatched"
                method = _method_label(scope["method"])
                HTTP_REQUESTS.labels(route=route, method=method, status=str(status)).inc()
                HTTP_DURATION.labels(route=route, method=method).observe(duration)
                logger.info(
                    "request",
                    extra={
                        "route": route,
                        "method": method,
                        "status": status,
                        "duration_ms": round(duration * 1000, 3),
                    },
                )


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        is_api = scope["path"].startswith("/api/")

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                headers["X-Frame-Options"] = "DENY"
                if is_api:
                    headers["Content-Security-Policy"] = (
                        "default-src 'none'; frame-ancestors 'none'"
                    )
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodySizeLimitMiddleware:
    """Reject oversized bodies without buffering them.

    A declared Content-Length over the limit is refused before the app runs. Otherwise the
    bytes of ``http.request`` messages are counted as the app reads them; crossing the limit
    aborts the app's read, discards whatever response the app then tries to send (a
    framework may turn the aborted read into its own error), and answers 413 instead.
    """

    def __init__(
        self,
        app: ASGIApp,
        json_limit: int = 1_048_576,
        upload_limit: int = 20_971_520 + 65_536,
    ) -> None:
        self.app = app
        self.json_limit = json_limit
        self.upload_limit = upload_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        is_multipart = headers.get("content-type", "").lower().startswith("multipart/form-data")
        limit = self.upload_limit if is_multipart else self.json_limit

        declared = headers.get("content-length")
        # Only plain ASCII digits (isdigit alone accepts characters int() rejects); anything
        # else is treated as absent and the streamed byte count still applies.
        if (
            declared is not None
            and declared.isascii()
            and declared.isdigit()
            and len(declared) <= _MAX_CONTENT_LENGTH_DIGITS
            and int(declared) > limit
        ):
            await self._reject(scope, receive, send)
            return

        received = 0
        exceeded = False
        started = False

        async def receive_wrapper() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if exceeded and not started:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except _BodyTooLarge:
            pass
        except Exception:
            if not exceeded:
                raise
        if exceeded:
            if started:
                # The response is already on the wire; the only way to stop is to close.
                raise _BodyTooLarge
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = problem_response(
            413, "payload_too_large", "Payload too large", "Request body exceeds the size limit."
        )
        await response(scope, receive, send)
