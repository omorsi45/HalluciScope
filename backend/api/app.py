from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.analysis import router as analysis_router
from backend.api.routes.feedback import router as feedback_router


def create_app(lifespan=None) -> FastAPI:
    app = FastAPI(
        title="HalluciScope",
        description="Multi-signal hallucination detection for RAG QA",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.include_router(analysis_router, prefix="/api")
    app.include_router(feedback_router, prefix="/api")

    return app
