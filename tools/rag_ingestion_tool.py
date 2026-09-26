import os
import json
import re
from typing import Any, List, Union
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


@tool("rag_pdf_indexer_tool")
def rag_ingestion_tool(pdf_paths: Union[List[str], str]) -> str:
    """
    Ingest, chunk, and index downloaded academic PDFs into the Hybrid RAG knowledge base.
    Creates dense semantic embeddings in ChromaDB and sparse lexical tokens in BM25 with layout-aware tracking.

    Input:
    - pdf_paths: List of local PDF file paths, or a JSON string from the PDF download tool.

    Output:
    - JSON confirmation with total documents indexed and number of text chunks created.
    """
    paths = _extract_paths(pdf_paths)
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