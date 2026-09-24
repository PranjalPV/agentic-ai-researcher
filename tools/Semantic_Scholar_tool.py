import json
import time
import random
import requests
from crewai.tools import tool


@tool("semantic_scholar_academic_search_tool")
def semantic_scholar_tool(query: str, limit: int = 3) -> str:
    """
    Search Semantic Scholar for academic papers with citation counts and open-access PDFs.

    Input:
    - query: Academic research query string
    - limit: Number of papers to retrieve (default 3)

    Output:
    - JSON-formatted string with titles, authors, year, citation count, and open-access PDF links.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {
        "User-Agent": "AgenticResearchAssistant/2.0 (Academic Research; mailto:research@agentic.ai)"
    }
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,abstract,year,citationCount,openAccessPdf"
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 429:
                sleep_sec = (2 ** attempt) + random.uniform(1.0, 2.5)
                time.sleep(sleep_sec)
                continue

            if response.status_code != 200:
                break

            data = response.json()
            papers = []

            for item in data.get("data", []):
                pdf_info = item.get("openAccessPdf")
                pdf_url = pdf_info.get("url") if pdf_info else None
                authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]

                if pdf_url:
                    abstract = item.get("abstract") or ""
                    papers.append({
                        "title": item.get("title", "Untitled").strip(),
                        "authors": authors[:5],
                        "published_year": item.get("year"),
                        "citation_count": item.get("citationCount", 0),
                        "summary": abstract[:350] + "..." if len(abstract) > 350 else abstract,
                        "pdf_url": pdf_url,
                        "source": "semantic_scholar"
                    })

            return json.dumps({
                "status": "success",
                "count": len(papers),
                "papers": papers
            }, indent=2)

        except Exception as e:
            time.sleep(1)

    return json.dumps({
        "status": "rate_limited_or_error",
        "message": "Semantic Scholar search limit reached or service unavailable. Fall back to arXiv.",
        "papers": []
    })