import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.db import get_collection
from src.query_parser import mmm_retriever
from pipeline import run as run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

_collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _collection
    logger.info("Loading vector store…")
    _collection = get_collection()
    logger.info("Vector store ready")
    yield


app = FastAPI(title="AutoAnalyst API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question:  str
    n_results: int = 10


class AnalyzeRequest(BaseModel):
    question: str


def _check_collection():
    if _collection is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2026.1"}


@app.get("/")
async def root():
    return {"message": "AutoAnalyst MMM Copilot API"}


@app.post("/query")
async def query(request: QueryRequest):
    """Raw semantic retrieval — returns matching MMM chunks."""
    _check_collection()
    logger.info("/query: %r (n_results=%d)", request.question, request.n_results)
    result = mmm_retriever(request.question, _collection, request.n_results)
    return {"question": request.question, "result": result}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Full pipeline: plan → execute tools → synthesize answer."""
    _check_collection()
    logger.info("/analyze: %r", request.question)
    answer = run_pipeline(request.question, _collection)
    return answer
