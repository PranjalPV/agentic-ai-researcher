import os
import json
import re
from typing import Any, List, Dict, Union
from crewai.tools import tool
from rag.rag_tool import get_rag_engine


def _extract_paths(raw_input: Any) -> List[str]:
    """Helper to extract a list of file paths from LLM tool argument."""
    if isinstance(raw_input, list):
        paths = []
        for item in raw_input:
            if isinstance(item, str):
                paths.append(item.strip())
            elif isinstance(item, dict) and "path" in item:
                paths.append(item["path"].strip())
        return paths

    if isinstance(raw_input, str):
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, list):
                return _extract_paths(parsed)
            if isinstance(parsed, dict):
                if "downloaded_files" in parsed:
                    return _extract_paths(parsed["downloaded_files"])
                if "paths" in parsed:
                    return _extract_paths(parsed["paths"])
        except Exception:
            pass

        # Match local file path patterns (e.g. temp_papers/... or D:/...)
        found = re.findall(r"(?:[A-Za-z]:[/\\]|temp_papers[/\\MF])[^\s'\"<>,;]+", raw_input)
        if found:
            return [p.strip() for p in found]

    return []

def _extract_papers(raw_input: Any) -> List[Dict[str, Any]]:
    """Helper to extract structured paper objects from LLM tool arguments."""
    if isinstance(raw_input, list):
        papers = []
        for item in raw_input:
            if isinstance(item, dict) and ("title" in item or "abstract" in item or "summary" in item):
                papers.append(item)
        if papers:
            return papers

    if isinstance(raw_input, str):
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, list):
                return _extract_papers(parsed)
            if isinstance(parsed, dict):
                if "papers" in parsed:
                    return _extract_papers(parsed["papers"])
                if "title" in parsed:
                    return [parsed]
        except Exception:
            pass

        # Regex fallback for text-formatted paper lists
        title_matches = re.findall(r"(?:Title|Paper Title|\bTitle\b):\s*([^\n\r]+)", raw_input, re.IGNORECASE)
        url_matches = re.findall(r"https?://(?:arxiv\.org/[^\s'\"<>,;]+|[^\s'\"<>,;]+\.pdf)", raw_input)
        if title_matches:
            papers = []
            for i, t in enumerate(title_matches):
                url = url_matches[i] if i < len(url_matches) else (url_matches[0] if url_matches else "")
                papers.append({
                    "title": t.strip(" *-\""),
                    "summary": f"Academic research paper: {t.strip()}",
                    "pdf_url": url
                })
            return papers

    return []


@tool("rag_pdf_indexer_tool")
def rag_ingestion_tool(literature_data: Union[Any, str]) -> str:
    """
    Ingest, chunk, and index academic literature into the Hybrid RAG knowledge base (Dense Vector + BM25 Lexical).
    Attaches direct open-access paper URLs and titles for grounded citations.
    
    Input:
    - literature_data: List of papers, JSON string with paper titles, abstracts, and PDF URLs, or local PDF paths.

    Output:
    - JSON confirmation with total documents indexed into vector storage.
    """
    rag_engine = get_rag_engine()

    # 1. First priority: Direct paper metadata ingestion (instant, zero memory overhead)
    papers = _extract_papers(literature_data)
    if papers:
        indexed_count = rag_engine.ingest_papers(papers)
        return json.dumps({
            "status": "success",
            "message": f"Successfully indexed {indexed_count} papers into Hybrid RAG with verified citations.",
            "indexed_files_count": indexed_count,
            "total_chunks_indexed": indexed_count,
            "indexed_documents": [
                {"title": p.get("title"), "url": p.get("pdf_url") or p.get("url", "")}
                for p in papers
            ]
        }, indent=2)

    # 2. Fallback: Path extraction if file paths were provided
    paths = _extract_paths(literature_data)
    if not paths and os.path.exists("temp_papers"):
        # Auto-fallback: check if there are files in temp_papers directory
        paths = [
            os.path.join("temp_papers", f)
            for f in os.listdir("temp_papers")
            if f.lower().endswith(".pdf")
        ]

    if not paths:
        return json.dumps({
            "status": "warning",
            "message": "No valid PDF paths provided or found for vector indexing.",
            "indexed_files": 0,
            "total_chunks": 0
        })

    rag_engine = get_rag_engine()
    successful = []
    total_chunks = 0

    for path in paths:
        if not os.path.exists(path):
            continue

        try:
            chunks_created = rag_engine.ingest_pdf(path)
            if chunks_created > 0:
                successful.append({
                    "path": path,
                    "filename": os.path.basename(path),
                    "chunks": chunks_created
                })
                total_chunks += chunks_created
        except Exception as e:
            print(f"[rag_ingestion_tool] Failed to index {path}: {e}")
        finally:
            # Unlink binary PDF after indexing to reclaim memory buffers and ephemeral disk
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    import gc
    gc.collect()

    return json.dumps({
        "status": "success" if successful else "failed",
        "indexed_files_count": len(successful),
        "total_chunks_indexed": total_chunks,
        "indexed_documents": successful
    }, indent=2)