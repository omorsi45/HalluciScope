"""RFC 7807 style problem responses and the exception handlers that produce them.

Handlers never echo request input or exception messages: validation details are built
from ``loc`` and ``msg`` only, and unhandled errors become a constant 500 body.
"""

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.auth.service import AuthError

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"

_HTTP_CODES = {
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
}

_AUTH_TITLES = {
    "invalid_credentials": "Invalid credentials",
    "too_many_attempts": "Too many attempts",
    "email_taken": "Email already registered",
    "weak_password": "Weak password",
    "invalid_email": "Invalid email",
    "invalid_display_name": "Invalid display name",
    "registration_disabled": "Registration disabled",
    "not_a_member": "Not a member of this workspace",
    "api_key_tenant_fixed": "API keys cannot switch workspace",
    "not_found": "Not found",
}


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        title: str,
        detail: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.headers = headers


def problem_response(
    status: int,
    code: str,
    title: str,
    detail: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {
            "type": f"/errors/{code}",
            "title": title,
            "status": status,
            "detail": detail,
            "code": code,
        },
        status_code=status,
        headers=headers,
        media_type=PROBLEM_MEDIA_TYPE,
    )


def unhandled_error_response() -> JSONResponse:
    """Log the exception being handled (class and frames only) and return a generic 500.

    Must be called while the exception is being handled so ``logger.exception`` sees it.
    """
    logger.exception("unhandled error")
    return problem_response(500, "internal_error", "Internal server error")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return problem_response(exc.status, exc.code, exc.title, exc.detail, exc.headers)

    @app.exception_handler(AuthError)
    async def _auth_error(request: Request, exc: AuthError) -> JSONResponse:
        title = _AUTH_TITLES.get(exc.code, "Authentication error")
        return problem_response(exc.status, exc.code, title)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return problem_response(422, "validation_error", "Validation error", detail)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        try:
            title = HTTPStatus(exc.status_code).phrase
        except ValueError:
            title = "HTTP error"
        detail = exc.detail if isinstance(exc.detail, str) else None
        return problem_response(exc.status_code, code, title, detail, exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        return unhandled_error_response()
