# tools/arxiv_tool.py
from crewai.tools import tool
import arxiv

@tool("arxiv_tool")
def arxiv_tool(query: str, max_results: int = 2) -> list:
    """
    Search arXiv for research papers.

    Input:
    - query: research topic

    Output:
    - list of PDF URLs
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    pdf_urls = []
    for result in search.results():
        if result.pdf_url:
            pdf_urls.append(result.pdf_url)

    return pdf_urls #return list of string that are urls of research papers in pdf

