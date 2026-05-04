import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from db.database import init_db
from routes import diagnoses, chat, conditions, dashboard, auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("dermai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DermAI Pro — LangChain Multi-Agent System v2.0")
    init_db()
    yield
    logger.info("Shutting down DermAI Pro")


app = FastAPI(
    title="DermAI Pro",
    description="Production-grade AI Skincare Diagnosis — LangChain Multi-Agent RAG System",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(diagnoses.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(conditions.router, prefix="/api")


@app.get("/api/healthz")
async def health():
    from agents.rag_agent import rag_agent
    return {
        "status": "ok",
        "service": "DermAI Pro",
        "version": "2.0.0",
        "agents": {
            "vision": "ready",
            "rag": {
                "ready": rag_agent.vectorstore is not None or rag_agent.fallback_retriever is not None,
                "embed_type": rag_agent.embed_type or "not_initialized",
                "document_count": len(rag_agent.documents),
            },
            "routine": "ready",
            "chat": "ready"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": str(request.url)}
    )
