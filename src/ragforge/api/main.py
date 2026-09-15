from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ragforge.api.application import (
    configure_production_services,
)
from ragforge.api.config import APIConfig
from ragforge.api.dependencies import (
    get_container,
)
from ragforge.api.schemas import (
    AgentRequest,
    AgentResponseSchema,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryRequest,
    MemoryResponse,
    RAGRequest,
    RAGResponseSchema,
)
from ragforge.api.services import APIService


config = APIConfig()


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Initialize production services when the
    FastAPI application starts.
    """

    configure_production_services()

    yield


app = FastAPI(
    title=config.title,
    version=config.version,
    description=config.description,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=config.title,
        version=config.version,
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(
    request: GenerateRequest,
) -> GenerateResponse:
    service = APIService(
        get_container()
    )

    try:
        return service.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post(
    "/rag",
    response_model=RAGResponseSchema,
)
def rag(
    request: RAGRequest,
) -> RAGResponseSchema:
    service = APIService(
        get_container()
    )

    try:
        return service.rag(
            request.question
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post(
    "/agent",
    response_model=AgentResponseSchema,
)
def agent(
    request: AgentRequest,
) -> AgentResponseSchema:
    service = APIService(
        get_container()
    )

    try:
        return service.agent(
            request.query
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post(
    "/memory",
    response_model=MemoryResponse,
)
def remember(
    request: MemoryRequest,
) -> MemoryResponse:
    service = APIService(
        get_container()
    )

    return service.remember(
        content=request.content,
        memory_type=request.memory_type,
        metadata=request.metadata,
    )


@app.post(
    "/memory/recall",
    response_model=MemoryRecallResponse,
)
def recall(
    request: MemoryRecallRequest,
) -> MemoryRecallResponse:
    service = APIService(
        get_container()
    )

    return service.recall(
        query=request.query,
        top_k=request.top_k,
    )


@app.delete("/memory")
def clear_memory() -> dict[str, str]:
    service = APIService(
        get_container()
    )

    service.clear_memory()

    return {
        "status": "cleared"
    }