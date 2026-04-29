# tools/pdf_ingestion_tool.py
from crewai.tools import tool
import os
import uuid
import requests

TEMP_DIR = "temp_papers"

@tool("pdf_ingestion_tool")
def pdf_ingestion_tool(pdf_urls: list) -> list:
    """
    Download PDF files from a list of URLs.

    Input:
    - pdf_urls: list of PDF URLs (strings)

    Output:
    - list of local file paths of successfully downloaded PDFs
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    saved_files = []

    for url in pdf_urls:
        try:
            filename = f"{uuid.uuid4().hex}.pdf"
            path = os.path.join(TEMP_DIR, filename)

            response = requests.get(url, timeout=20)
            response.raise_for_status()

            if "pdf" not in response.headers.get("Content-Type", "").lower():
                continue

            with open(path, "wb") as f:
                f.write(response.content)

            saved_files.append(path)

        except Exception:
            continue

    return saved_files

