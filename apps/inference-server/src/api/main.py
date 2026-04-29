from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.errors import register_exception_handlers
from src.api.health import build_health_payload, build_liveness_payload
from src.api.middleware import RequestContextLoggingMiddleware
from src.api.tasks import router as task_router
from src.core.logging import configure_logging
from src.models.captioning import list_available_caption_models


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="smart-glass-inference-server",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(RequestContextLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(task_router)

    @app.get("/health/live")
    async def liveness_check() -> JSONResponse:
        status_code, payload = build_liveness_payload()
        return JSONResponse(status_code=status_code, content=payload)

    @app.get("/health/ready")
    async def readiness_check() -> JSONResponse:
        status_code, payload = build_health_payload()
        return JSONResponse(status_code=status_code, content=payload)

    @app.get("/health")
    async def health_check() -> JSONResponse:
        status_code, payload = build_health_payload()
        return JSONResponse(status_code=status_code, content=payload)

    @app.get("/models/captioning")
    async def get_caption_models() -> dict[str, list[dict[str, str]]]:
        models = [
            {
                "key": spec.key,
                "model_id": spec.model_id,
                "family": spec.family,
                "recommended_quantization": spec.recommended_quantization,
                "notes": spec.notes,
            }
            for spec in list_available_caption_models()
        ]
        return {"models": models}

    return app


app = create_app()
