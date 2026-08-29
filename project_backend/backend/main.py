from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .core.logging_config import logger
from .api.routes import health, inference, report


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Target PyTorch Device: {settings.DEVICE}")
    logger.info(f"API Docs available at: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Research REST API for Multimodal Stroke and Alzheimer's Disease Detection "
        "using OCT-A, OCT-B, Fundus Retinal Images and Tabular Clinical Data."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
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
