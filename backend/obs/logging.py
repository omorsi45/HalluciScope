"""Structured JSON logging with bound context and a field whitelist.

Each record becomes one JSON object per line. Only the fields in ALLOWED_EXTRA are
copied from ``extra=...``; everything else is dropped so user text cannot leak into logs.
Exceptions are reduced to the class name and stack frames, never the message.

Log messages must be constant strings. A string ``msg`` is ``record.getMessage()``, so any
``%s`` argument a caller passes ends up in the log line; callers control that, which is
why dynamic values belong in whitelisted ``extra`` fields instead. A non-string message
object (for example an exception) is reduced to its type name, and whitelisted values that
are not str, int, float, bool or None are emitted as their type name too.
"""

import json
import logging
import math
import os
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)

_CONTEXT_VARS: dict[str, ContextVar[str | None]] = {
    "request_id": request_id_var,
    "tenant_id": tenant_id_var,
    "user_id": user_id_var,
    "job_id": job_id_var,
}

ALLOWED_EXTRA: frozenset[str] = frozenset(
    {
        "route",
        "method",
        "status",
        "duration_ms",
        "kind",
        "outcome",
        "error_code",
        "attempt",
        "worker_id",
        "n_passages",
        "n_claims",
        "n_reused",
        "n_new",
        "bytes",
        "version_number",
        "stage",
    }
)


@contextmanager
def bind(**values: str | None) -> Iterator[None]:
    unknown = set(values) - set(_CONTEXT_VARS)
    if unknown:
        raise KeyError(sorted(unknown)[0])
    tokens = [(_CONTEXT_VARS[key], _CONTEXT_VARS[key].set(value)) for key, value in values.items()]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _safe_value(value: object) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return type(value).__name__


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage()
            if isinstance(record.msg, str)
            else type(record.msg).__name__,
        }
        for key, var in _CONTEXT_VARS.items():
            value = var.get()
            if value is not None:
                payload[key] = value
        for key in ALLOWED_EXTRA:
            if key in record.__dict__:
                payload[key] = _safe_value(record.__dict__[key])
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc_frames"] = [
                f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"
                for frame in traceback.extract_tb(record.exc_info[2])
            ]
        return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    uvicorn = logging.getLogger("uvicorn")
    uvicorn.handlers = [handler]
    uvicorn.propagate = False
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
    access.disabled = True
