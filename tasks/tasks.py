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
    Conduct an academic literature search for the topic: '{query}'.
    1. Query arXiv using arxiv_academic_search_tool with relevant keywords.
    2. Query Semantic Scholar using semantic_scholar_academic_search_tool for additional open-access papers.
    3. Screen abstracts for technical depth and relevance.
    4. Compile a curated list of at least 2-4 candidate papers including Paper Title, Authors, Year, Summary, and direct PDF URL.
    Ensure all PDF URLs are explicitly listed.
    """,
    agent=research_agent,
    expected_output="""
    A JSON-parsable list or Markdown section containing:
    - List of papers with Title, Authors, Year, Summary Abstract
    - Direct PDF download URLs for each paper
    """
)

# ======================================================
# 2. Ingestion Task – PDF Ingestion & Hybrid Indexing
# ======================================================
ingestion_task = Task(
    description="""
    Take the PDF URLs identified in the research task:
    1. Use pdf_download_tool to download the PDFs to local temporary storage.
    2. Use rag_pdf_indexer_tool to ingest and index the downloaded PDFs into the Hybrid RAG engine (Dense vector + BM25).
    3. Verify that chunks have been successfully indexed.
    """,
    agent=ingestion_agent,
    context=[research_task],
    expected_output="""
    A summary of the indexing process:
    - Number of papers successfully downloaded
    - Number of text chunks indexed into ChromaDB and BM25
    - Confirmation of readiness for hybrid retrieval
    """
)

# ======================================================
# 3. Comparison Task – Systematic Comparative Review
# ======================================================
comparison_task = Task(
    description="""
    Using the academic_hybrid_rag_tool, perform a detailed comparative review across all indexed papers for: '{query}'.
    Execute specific queries to extract:
    1. Architectural Design & Methodology: How does each paper approach the problem?
    2. Benchmarks & Datasets: Which evaluation datasets are used?
    3. Empirical Results: What quantitative scores (accuracy, F1, latency, throughput) were reported?
    4. Strengths vs. Limitations: What are the trade-offs, computational overheads, or failure cases?

    Ground every insight in retrieved text and cite the paper title and page number whenever possible.
    Format the comparison as a structured Markdown table followed by an analytical discussion.
    """,
    agent=comparison_agent,
    context=[research_task, ingestion_task],
    expected_output="""
    A comprehensive comparative review containing:
    - Markdown Comparative Table (Columns: Paper, Architecture/Method, Dataset, Key Metric, Strengths, Limitations)
    - In-depth analytical comparison with page-grounded citations
    """
)

# ======================================================
# 4. Insight Task – Synthesis, Gaps & Future Directions
# ======================================================
insight_task = Task(
    description="""
    Synthesize all findings into an Executive Academic Research Dossier on: '{query}'.
    Address the following:
    1. Executive Summary: What is the current state-of-the-art in this research domain?
    2. Critical Research Gaps: What problems remain unaddressed or under-evaluated across existing papers?
    3. Scaling Bottlenecks & Open Engineering Challenges: What stops current methods from production or scale?
    4. Actionable Future Research Directions: Propose 3-4 concrete hypotheses or architectural paradigms for future work.
    5. Recommended Baseline Architecture: Which existing method should a practitioner or researcher adopt today?
    6. References: Complete citations of all analyzed papers with URLs.
    """,
    agent=insight_agent,
    context=[research_task, comparison_task],
    expected_output="""
    A publication-quality Executive Research Dossier in Markdown format, thoroughly structured with clear headings, bullet points, citations, and strategic recommendations.
    """
)
