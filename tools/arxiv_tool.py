import json
import arxiv
from crewai.tools import tool


@tool("arxiv_academic_search_tool")
def arxiv_tool(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for high-impact academic research papers.

    Input:
    - query: Academic research topic or keywords (e.g. 'mechanistic interpretability transformer attention')
    - max_results: Maximum number of papers to return (default 3)

    Output:
    - JSON-formatted string with paper titles, authors, published dates, summaries, and PDF download URLs.
    """
    try:
        clean_query = query.strip().replace('"', '').replace("'", "")
        client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
        search = arxiv.Search(
            query=clean_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        for result in client.results(search):
            pdf_url = result.pdf_url
            if not pdf_url and result.entry_id:
                pdf_url = result.entry_id.replace("abs", "pdf") + ".pdf"

            if pdf_url:
                papers.append({
                    "title": result.title.replace("\n", " ").strip(),
                    "authors": [a.name for a in result.authors][:5],
                    "published_year": result.published.year if result.published else None,
                    "summary": result.summary.replace("\n", " ")[:350] + "...",
                    "pdf_url": pdf_url,
                    "source": "arxiv"
                })

        if not papers:
            return json.dumps({"status": "no_results", "papers": [], "message": f"No papers found on arXiv for query '{query}'."})

        return json.dumps({"status": "success", "count": len(papers), "papers": papers}, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error searching arXiv: {str(e)}", "papers": []})
