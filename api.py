import os
import gc
import json
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

app = FastAPI(
    title="Agentic AI Academic Researcher API",
    description="Asynchronous REST API for autonomous academic literature discovery, hybrid RAG ingestion, and comparative review.",
    version="2.0.0"
)

# Configure CORS Whitelist via ALLOWED_ORIGINS environment variable
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_raw.strip() == "*":
    cors_origins = ["*"]
else:
    cors_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=3)

# In-memory job registry for tracking asynchronous execution
JOBS: Dict[str, Dict[str, Any]] = {}


class ConfigRequest(BaseModel):
    groq_api_key: str = Field(..., description="Groq API Key to configure in runtime")


class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research topic or scientific inquiry")
    session_id: Optional[str] = Field(default=None, description="Optional custom session identifier")
    api_key: Optional[str] = Field(default=None, description="Optional Groq API Key override")


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


JOBS_FILE = os.path.join(REPORTS_DIR, "jobs_registry.json")


def _load_persisted_jobs() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_persisted_job(job_id: str, data: Dict[str, Any]):
    try:
        jobs = _load_persisted_jobs()
        jobs[job_id] = data
        with open(JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2)
    except Exception:
        pass


def _execute_research_job(job_id: str, query: str, session_id: Optional[str], api_key: Optional[str] = None):
    """Target worker executed inside thread pool."""
    JOBS[job_id]["status"] = "running"
    _save_persisted_job(job_id, JOBS[job_id])
    try:
        gc.collect()
        if api_key and api_key.strip():
            os.environ["GROQ_API_KEY"] = api_key.strip()
        from crew import run_research
        result_text = run_research(query=query, save_report=True, session_id=session_id)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result_text
        JOBS[job_id]["completed_at"] = datetime.now().isoformat()
        _save_persisted_job(job_id, JOBS[job_id])
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["completed_at"] = datetime.now().isoformat()
        _save_persisted_job(job_id, JOBS[job_id])
    finally:
        gc.collect()


@app.get("/health", tags=["System"])
def health_check():
    """System health check and environmental status."""
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    report_count = len([f for f in os.listdir(REPORTS_DIR) if f.endswith(".md")]) if os.path.exists(REPORTS_DIR) else 0
    return {
        "status": "healthy",
        "service": "Agentic AI Academic Researcher",
        "version": "2.0.0",
        "groq_configured": has_groq,
        "openai_configured": has_openai,
        "reports_available": report_count
    }


@app.post("/api/config", tags=["System"])
def set_config(config: ConfigRequest):
    """Configures or updates runtime credentials."""
    key = config.groq_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    os.environ["GROQ_API_KEY"] = key
    return {"status": "success", "message": "Groq API Key successfully configured in runtime."}


@app.post("/api/research", response_model=ResearchJobResponse, tags=["Research Engine"])
def create_research_job(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Submits an academic research task for asynchronous multi-agent execution.
    Returns immediately with a job_id to poll status.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Research query cannot be empty.")

    if request.api_key and request.api_key.strip():
        os.environ["GROQ_API_KEY"] = request.api_key.strip()

    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is not set. Please provide it in the UI settings or configure it in your environment."
        )

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
    _save_persisted_job(job_id, JOBS[job_id])

    # Dispatch to background thread pool
    background_tasks.add_task(_execute_research_job, job_id, request.query.strip(), request.session_id, request.api_key)

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
        persisted = _load_persisted_jobs()
        job = persisted.get(job_id)
        # If job was persisted as running but missing from active memory, it was interrupted by server restart
        if job and job.get("status") == "running":
            job["status"] = "failed"
            job["error"] = "Research execution was interrupted by cloud container recycling. Please retry."
            _save_persisted_job(job_id, job)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
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


# Mount Static Frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serves the Single Page Application interface."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Agentic AI Academic Researcher API is online. Frontend static files not found."}


@app.get("/{filename}.js", include_in_schema=False)
def serve_root_js(filename: str):
    fpath = os.path.join(STATIC_DIR, f"{filename}.js")
    if os.path.exists(fpath):
        return FileResponse(fpath, media_type="application/javascript")
    raise HTTPException(status_code=404)


@app.get("/{filename}.css", include_in_schema=False)
def serve_root_css(filename: str):
    fpath = os.path.join(STATIC_DIR, f"{filename}.css")
    if os.path.exists(fpath):
        return FileResponse(fpath, media_type="text/css")
    raise HTTPException(status_code=404)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
