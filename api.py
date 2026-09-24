import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from crew import run_research, REPORTS_DIR

app = FastAPI(
    title="Agentic AI Academic Researcher API",
    description="Asynchronous REST API for autonomous academic literature discovery, hybrid RAG ingestion, and comparative review.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=3)

# In-memory job registry for tracking asynchronous execution
JOBS: Dict[str, Dict[str, Any]] = {}


class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research topic or scientific inquiry")
    session_id: Optional[str] = Field(default=None, description="Optional custom session identifier")


class ResearchJobResponse(BaseModel):
    job_id: str
    status: str
    query: str
    created_at: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    query: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


def _execute_research_job(job_id: str, query: str, session_id: Optional[str]):
    """Target worker executed inside thread pool."""
    JOBS[job_id]["status"] = "running"
    try:
        result_text = run_research(query=query, save_report=True, session_id=session_id)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result_text
        JOBS[job_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["completed_at"] = datetime.now().isoformat()


@app.get("/health", tags=["System"])
def health_check():
    """System health check and environmental status."""
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status": "healthy",
        "service": "Agentic AI Academic Researcher",
        "version": "2.0.0",
        "groq_configured": has_groq,
        "openai_configured": has_openai,
        "reports_available": len(os.listdir(REPORTS_DIR)) if os.path.exists(REPORTS_DIR) else 0
    }


@app.post("/api/research", response_model=ResearchJobResponse, tags=["Research Engine"])
def create_research_job(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Submits an academic research task for asynchronous multi-agent execution.
    Returns immediately with a job_id to poll status.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Research query cannot be empty.")

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    now = datetime.now().isoformat()

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "query": request.query.strip(),
        "created_at": now,
        "completed_at": None,
        "result": None,
        "error": None
    }

    # Dispatch to background thread pool
    background_tasks.add_task(_execute_research_job, job_id, request.query.strip(), request.session_id)

    return ResearchJobResponse(
        job_id=job_id,
        status="queued",
        query=request.query.strip(),
        created_at=now,
        message="Research job queued successfully. Check /api/research/{job_id} for progress."
    )


@app.get("/api/research/{job_id}", response_model=JobStatusResponse, tags=["Research Engine"])
def get_job_status(job_id: str):
    """Retrieves the live status and completed results of a research task."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return JobStatusResponse(**job)


@app.get("/api/reports", tags=["Reports"])
def list_reports() -> List[Dict[str, Any]]:
    """Lists all stored research dossiers."""
    if not os.path.exists(REPORTS_DIR):
        return []

    reports = []
    for fname in os.listdir(REPORTS_DIR):
        if fname.endswith(".md"):
            fpath = os.path.join(REPORTS_DIR, fname)
            stat = os.stat(fpath)
            reports.append({
                "filename": fname,
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    return sorted(reports, key=lambda x: x["modified"], reverse=True)


@app.get("/api/reports/{filename}", tags=["Reports"])
def get_report_content(filename: str):
    """Retrieves the Markdown content of a specific stored report."""
    safe_name = os.path.basename(filename)
    fpath = os.path.join(REPORTS_DIR, safe_name)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Report not found.")

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    return {"filename": safe_name, "content": content}
