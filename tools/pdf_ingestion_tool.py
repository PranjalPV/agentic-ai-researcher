import os
import re
import json
import uuid
from typing import Any, List, Union
import requests
from crewai.tools import tool

TEMP_DIR = "temp_papers"


def _extract_urls(raw_input: Any) -> List[str]:
    """Helper to extract a list of URLs from diverse LLM tool call argument types."""
    if isinstance(raw_input, list):
        urls = []
        for item in raw_input:
            if isinstance(item, str):
                urls.append(item.strip())
            elif isinstance(item, dict) and "pdf_url" in item:
                urls.append(item["pdf_url"].strip())
        return urls

    if isinstance(raw_input, str):
        # Attempt to parse as JSON first
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, list):
                return _extract_urls(parsed)
            if isinstance(parsed, dict) and "pdf_urls" in parsed:
                return _extract_urls(parsed["pdf_urls"])
        except Exception:
            pass

        # Fallback to regex extraction of http/https URLs
        found = re.findall(r"https?://[^\s'\"<>,;]+", raw_input)
        return [u.strip() for u in found]

    return []


@tool("pdf_download_tool")
def pdf_ingestion_tool(pdf_urls: str) -> str:
    """
    Download academic PDF research papers from a list of web URLs to local temporary storage.

    Input:
    - pdf_urls: List of direct PDF URLs, or a comma-separated string/JSON containing URLs.

    Output:
    - JSON string reporting successfully downloaded file paths and file sizes.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    urls = _extract_urls(pdf_urls)

    if not urls:
        return json.dumps({
            "status": "error",
            "message": "No valid PDF URLs detected in input argument.",
            "downloaded_files": []
        })

    # Strictly cap to top 2 papers to avoid memory exhaustion on 512MB cloud instances
    urls = urls[:2]

    saved_files = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for url in urls:
        path = None
        try:
            # Ensure URL points to PDF endpoint for arXiv
            if "arxiv.org/abs/" in url:
                url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"

            response = requests.get(url, headers=headers, timeout=20, stream=True, allow_redirects=True)
            if response.status_code != 200:
                continue

            filename = f"paper_{uuid.uuid4().hex[:8]}.pdf"
            path = os.path.join(TEMP_DIR, filename)

            total_bytes = 0
            max_bytes = 10 * 1024 * 1024  # 10MB ceiling
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            break
                        f.write(chunk)

            if total_bytes < 100:
                if os.path.exists(path):
                    os.remove(path)
                continue

            # Verify PDF magic byte signature (%PDF-)
            with open(path, "rb") as f:
                header = f.read(5)
            if not header.startswith(b"%PDF"):
                if os.path.exists(path):
                    os.remove(path)
                continue

            saved_files.append({
                "path": path,
                "size_kb": round(total_bytes / 1024, 1),
                "source_url": url
            })

        except Exception as e:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
            continue

    return json.dumps({
        "status": "success" if saved_files else "failed",
        "total_downloaded": len(saved_files),
        "downloaded_files": [f["path"] for f in saved_files],
        "details": saved_files
    }, indent=2)
