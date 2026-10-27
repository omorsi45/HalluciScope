from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.errors import install_error_handlers
from backend.api.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    UnhandledErrorMiddleware,
)
from backend.api.routes.analysis import router as analysis_router
from backend.api.routes.api_keys import router as api_keys_router
from backend.api.routes.auth import router as auth_router
from backend.api.routes.collections import router as collections_router
from backend.api.routes.feedback import router as feedback_router
from backend.config import Settings


def create_app(settings: Settings | None = None, lifespan=None) -> FastAPI:
    settings = settings or Settings()
    if "*" in settings.cors_origins:
        raise ValueError("cors_origins must not contain '*' because credentials are allowed")
    app = FastAPI(
        title="HalluciScope",
        description="Multi-signal hallucination detection for RAG QA",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    install_error_handlers(app)

    # add_middleware prepends, so the last one added is the outermost. The error middleware
    # is innermost so its 500 passes back out through CORS and the security headers.
    app.add_middleware(UnhandledErrorMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-CSRF-Token", "X-Request-ID", "Authorization"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Later tasks include their routers into v1 before it is mounted.
    v1 = APIRouter()
    v1.include_router(auth_router, prefix="/auth")
    v1.include_router(api_keys_router, prefix="/api-keys")
    v1.include_router(collections_router, prefix="/collections")
    app.include_router(v1, prefix="/api/v1")

    # Legacy routes, replaced in Plan C.
    app.include_router(analysis_router, prefix="/api")
    app.include_router(feedback_router, prefix="/api")

    return app
