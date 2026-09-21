"""FastAPI REST layer for the research-paper recommender."""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from recommender import (
    ArxivClient,
    PaperRecommender,
    SemanticEncoder,
    demo_papers,
)
from research_workspace import ResearchWorkspace


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://research_user:research_password@127.0.0.1:5432/research_papers",
)
DEFAULT_PROFILE = os.getenv("PROFILE", "neuro")
DEFAULT_INTERESTS = os.getenv(
    "INTERESTS",
    "Machine learning for schizophrenia diagnosis, functional MRI, "
    "brain connectivity, and multimodal clinical data",
)
PAPERS_PER_QUERY = int(os.getenv("PAPERS_PER_QUERY", "75"))
DEFAULT_QUERIES = json.loads(
    os.getenv(
        "QUERIES_JSON",
        json.dumps(
            [
                'all:schizophrenia AND (all:"machine learning" OR all:fMRI)',
                'all:"brain connectivity" AND (all:graph OR all:learning)',
                'all:multimodal AND (all:clinical OR all:neuroscience)',
            ]
        ),
    )
)


_services = {}
_service_lock = threading.Lock()


def get_services():
    if _services:
        return _services["recommender"], _services["workspace"]

    with _service_lock:
        if _services:
            return _services["recommender"], _services["workspace"]

        encoder = SemanticEncoder()
        recommender = PaperRecommender(DATABASE_URL, encoder)

        if recommender.count == 0:
            client = ArxivClient()
            collected = []
            try:
                for query in DEFAULT_QUERIES:
                    collected.extend(
                        client.fetch(
                            query,
                            max_results=PAPERS_PER_QUERY,
                        )
                    )
            except Exception as error:
                print(
                    "Live arXiv collection unavailable; "
                    "loading clearly labeled demo papers:",
                    type(error).__name__,
                    error,
                )
                collected = demo_papers()

            if not collected:
                collected = demo_papers()

            recommender.add_papers(collected)

        recommender.set_profile(
            DEFAULT_PROFILE,
            DEFAULT_INTERESTS,
        )
        workspace = ResearchWorkspace(
            recommender,
            DEFAULT_PROFILE,
        )

        _services["recommender"] = recommender
        _services["workspace"] = workspace
        return recommender, workspace


@asynccontextmanager
async def lifespan(app):
    get_services()
    yield
    recommender = _services.get("recommender")
    if recommender is not None:
        recommender.close()


app = FastAPI(
    title="Research Paper Recommender API",
    version="0.2.0",
    description=(
        "Semantic paper retrieval, personalized ranking, "
        "topic maps, feedback, and PDF imports."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TopicRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=250)


class ProfileRequest(BaseModel):
    profile: str = Field(min_length=1, max_length=200)


class PaperImportRequest(BaseModel):
    profile: str = Field(min_length=1, max_length=200)
    mode: str = "text"
    arxiv: str = ""
    title: str = ""
    abstract: str = ""
    url: str = ""
    filename: str = ""
    pdf: str = ""
    new_tab: bool = False


class VisitRequest(BaseModel):
    profile: str
    paper_id: str


class FeedbackRequest(BaseModel):
    profile: str
    paper_id: str
    rating: int


class HistoryRequest(BaseModel):
    profile: str
    visits: list[dict]


def as_http_error(error):
    if isinstance(error, ValueError):
        return HTTPException(
            status_code=400,
            detail=str(error),
        )
    return HTTPException(
        status_code=500,
        detail=f"{type(error).__name__}: {error}",
    )


@app.get("/", response_class=HTMLResponse)
async def index(profile: str | None = None):
    try:
        _, workspace = get_services()
        profile = profile or DEFAULT_PROFILE
        return HTMLResponse(workspace.html(profile))
    except Exception as error:
        raise as_http_error(error)


@app.get("/api/health")
async def health():
    recommender, _ = get_services()
    return {
        "status": "ok",
        "papers": recommender.count,
        "database": "PostgreSQL",
        "vector_extension": "pgvector",
        "embedding_model": recommender.encoder.identity,
    }


@app.get("/api/workspace")
async def workspace_payload(
    profile: str = Query(default=DEFAULT_PROFILE),
):
    try:
        _, workspace = get_services()
        return {"payload": workspace.payload(profile)}
    except Exception as error:
        raise as_http_error(error)


@app.post("/api/topics")
async def create_topic(request: TopicRequest):
    try:
        _, workspace = get_services()
        profile = workspace.new_topic(request.topic)
        return {"payload": workspace.payload(profile)}
    except Exception as error:
        raise as_http_error(error)


@app.post("/api/topics/switch")
async def switch_topic(request: ProfileRequest):
    try:
        _, workspace = get_services()
        return {
            "payload": workspace.payload(
                request.profile,
            )
        }
    except Exception as error:
        raise as_http_error(error)


@app.post("/api/papers/import")
async def import_paper(request: PaperImportRequest):
    try:
        _, workspace = get_services()
        data = request.model_dump()
        profile = workspace.import_paper(
            request.profile,
            data,
        )
        return {"payload": workspace.payload(profile)}
    except Exception as error:
        raise as_http_error(error)


@app.post("/api/visits")
async def record_visit(request: VisitRequest):
    try:
        _, workspace = get_services()
        return workspace.visit(
            request.profile,
            request.paper_id,
        )
    except Exception as error:
        raise as_http_error(error)


@app.post("/api/feedback")
async def save_feedback(request: FeedbackRequest):
    try:
        _, workspace = get_services()
        return workspace.rate(
            request.profile,
            request.paper_id,
            request.rating,
        )
    except Exception as error:
        raise as_http_error(error)


@app.post("/api/history/restore")
async def restore_history(request: HistoryRequest):
    try:
        _, workspace = get_services()
        return workspace.restore(
            request.profile,
            request.visits,
        )
    except Exception as error:
        raise as_http_error(error)


@app.get("/api/recommendations")
async def recommendations(
    profile: str = Query(default=DEFAULT_PROFILE),
    top_k: int = Query(default=10, ge=1, le=50),
    exploration: float = Query(
        default=0.2,
        ge=0.0,
        le=1.0,
    ),
):
    try:
        recommender, _ = get_services()
        return {
            "profile": profile,
            "recommendations": recommender.recommend(
                profile,
                top_k=top_k,
                exploration=exploration,
            ),
        }
    except Exception as error:
        raise as_http_error(error)


@app.get("/api/pdf")
async def uploaded_pdf(paper_id: str):
    try:
        _, workspace = get_services()
        return Response(
            content=workspace.get_pdf(paper_id),
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    'inline; filename="uploaded-paper.pdf"'
                )
            },
        )
    except Exception as error:
        raise as_http_error(error)
