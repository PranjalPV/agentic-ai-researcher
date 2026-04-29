# tools/semantic_scholar_tool.py
from crewai.tools import tool
import requests
import time
import random

@tool("semantic_scholar_tool")
def semantic_scholar_tool(query: str, limit: int = 2) -> list:
    """
    Search Semantic Scholar for open-access PDFs.

    Input:
    - query: research query string

    Output:
    - list of PDF URLs
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,openAccessPdf"
    }

    for attempt in range(3):
        response = requests.get(url, params=params, timeout=20)

        if response.status_code == 429:
            time.sleep((2 ** attempt) + random.uniform(1, 2))
            continue

        if response.status_code != 200:
            return []

        data = response.json()
        pdf_urls = []

        for paper in data.get("data", []):
            pdf = paper.get("openAccessPdf")
            if pdf and pdf.get("url"):
                pdf_urls.append(pdf["url"])

        return pdf_urls

    return []

        
        