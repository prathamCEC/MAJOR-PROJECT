from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .core.config import settings
from .core.logging_config import logger
from .db.init_db import init_db
from .api.routes import auth, health, inference, patients, analyses, report


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject defensive HTTP security headers into every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Target PyTorch Device: {settings.DEVICE}")
    logger.info(f"API Docs available at: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 60)
    
    # Initialize relational database tables and default administrator
    try:
        await init_db()
    except Exception as e:
        logger.error(f"Database initialization warning / error: {e}", exc_info=True)

    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Research REST API for Multimodal Stroke and Alzheimer's Disease Detection "
        "using OCT-A, OCT-B, Fundus Retinal Images and Tabular Clinical Data. "
        "Features Relational SQL Database, Argon2id Authentication, and Patient Management."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Attach Security Headers Middleware
app.add_middleware(SecurityHeadersMiddleware)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Safe Error Handlers (Requirement 25: Never expose stack traces or internal secrets)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Sanitize validation errors to prevent leaking internal schemas or file paths."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append({"field": field, "message": err.get("msg", "Invalid input value")})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error in request parameters.", "errors": errors},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for uncaught 500 errors. Logs details server-side only; returns sanitized response."""
    logger.error(f"Unhandled system error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact system support."},
    )


# Include API Routers
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(analyses.router)
app.include_router(health.router)
app.include_router(inference.router)
app.include_router(report.router)

# Mount Static Visualizations for frontend rendering
static_dir = settings.PROJECT_ROOT / "phase_10_explainability" / "outputs"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/explainability", StaticFiles(directory=str(static_dir)), name="explainability_static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
