"""Tiny authenticated HTTP server exposing the metrics registry for the worker process."""

import hmac
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from backend.obs.metrics import REGISTRY, Registry

CONTENT_TYPE = "text/plain; version=0.0.4"


def start_metrics_server(
    host: str, port: int, token: str | None, registry: Registry = REGISTRY
) -> ThreadingHTTPServer:
    expected = f"Bearer {token}".encode() if token is not None else None

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes = b"", extra: dict[str, str] | None = None):
            self.send_response(status)
            self.send_header("Content-Type", CONTENT_TYPE if status == 200 else "text/plain")
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if urlsplit(self.path).path != "/metrics" or expected is None:
                self._reply(404)
                return
            supplied = self.headers.get("Authorization", "").encode("utf-8", "replace")
            if not hmac.compare_digest(supplied, expected):
                self._reply(401, extra={"WWW-Authenticate": "Bearer"})
                return
            self._reply(200, registry.render().encode("utf-8"))

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="metrics-server").start()
    return server
