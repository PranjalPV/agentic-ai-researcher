from crewai import Task
from agents.agents import (
    research_agent,
    ingestion_agent,
    comparison_agent,
    insight_agent
)

# ======================================================
# 1. Research Task – Literature Discovery
# ======================================================
research_task = Task(
    description="""
    Search arXiv using arxiv_academic_search_tool for: '{query}'.
    Select 2 relevant papers with active PDF URLs.
    Output: Paper Title, Authors, Year, 2-sentence summary, and direct PDF URL.
    """,
    agent=research_agent,
    expected_output="""
    A list of 2 research papers containing:
    - Title, Authors, Published Year
    - Brief 2-sentence summary
    - Direct PDF download URL
    """
)

# ======================================================
# 2. Ingestion Task – PDF Ingestion & Hybrid Indexing
# ======================================================
ingestion_task = Task(
    description="""
    From the research task results:
    1. Call pdf_download_tool with the PDF URLs to download them.
    2. Call rag_pdf_indexer_tool with the paths to index them into ChromaDB and BM25.
    3. Output confirmation of papers downloaded and total chunks indexed.
    """,
    agent=ingestion_agent,
    context=[research_task],
    expected_output="""
    Confirmation with count of papers downloaded and total chunks indexed.
    """
)

# ======================================================
# 3. Comparison Task – Systematic Comparative Review
# ======================================================
comparison_task = Task(
    description="""
    Use academic_hybrid_rag_tool once to query the indexed papers for key methodology, datasets, and benchmark results for: '{query}'.
    Create a Markdown comparative table with columns:
    | Paper Title | Methodology / Architecture | Datasets | Key Results | Strengths & Limitations |
    Follow with a concise summary paragraph citing page numbers.
    """,
    agent=comparison_agent,
    context=[research_task],
    expected_output="""
    A Markdown comparison table and brief analytical summary grounded in citations.
    """
)

# ======================================================
# 4. Insight Task – Synthesis, Gaps & Future Directions
# ======================================================
insight_task = Task(
    description="""
    Based on the comparison findings, write an Executive Academic Research Dossier for: '{query}'.
    Include:
    1. Executive Summary
    2. 2-3 Critical Research Gaps
    3. 2 High-Impact Future Directions
    4. Recommended Baseline Approach
    5. References with URLs
    Keep formatting clean and concise.
    """,
    agent=insight_agent,
    context=[comparison_task],
    expected_output="""
    A publication-quality Executive Research Dossier in Markdown format.
    """
)
