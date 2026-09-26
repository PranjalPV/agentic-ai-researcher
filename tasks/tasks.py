from crewai import Task, Agent
from agents.agents import (
    research_agent,
    ingestion_agent,
    comparison_agent,
    insight_agent
)

# ======================================================
# 1. Research Task – Literature Discovery
# ======================================================
def create_research_task(agent: Agent) -> Task:
    return Task(
        description="""
        Search arXiv and Semantic Scholar for research literature on: '{query}'.
        Select the top 2-3 most authoritative and relevant academic papers.
        For each paper, extract and output:
        - Title
        - Authors & Published Year
        - Core Abstract & Technical Methodology Summary
        - Direct PDF Download URL (e.g., https://arxiv.org/pdf/...)
        """,
        agent=agent,
        expected_output="""
        A structured list of 2-3 academic papers containing Title, Authors, Year, Technical Abstract, and Direct PDF Download Links.
        """
    )

# ======================================================
# 2. Ingestion Task – Hybrid Knowledge Indexing
# ======================================================
def create_ingestion_task(agent: Agent, r_task: Task) -> Task:
    return Task(
        description="""
        From the discovered research literature in the scout results:
        Call rag_pdf_indexer_tool with the paper data to index paper titles, abstracts, methodology, and direct PDF links into the Hybrid RAG engine (ChromaDB + BM25).
        Ensure each entry attaches its direct open-access PDF link for grounded citations.
        Output: Confirmation of indexed papers and chunks.
        """,
        agent=agent,
        context=[r_task],
        expected_output="""
        Confirmation reporting the number of academic papers and chunks indexed into the Hybrid RAG vector store.
        """
    )

# ======================================================
# 3. Comparison Task – Systematic Comparative Review
# ======================================================
def create_comparison_task(agent: Agent, r_task: Task) -> Task:
    return Task(
        description="""
        Use academic_hybrid_rag_tool once to query the indexed literature for key methodology, architectures, datasets, and benchmark results for: '{query}'.
        Construct a Markdown comparative matrix table:
        | Paper Title & Direct Link | Methodology / Architecture | Datasets | Key Results | Strengths & Limitations |
        Follow the table with a concise comparative analysis paragraph citing paper titles and direct links.
        """,
        agent=agent,
        context=[r_task],
        expected_output="""
        A comparative matrix table with direct paper links and analytical summary grounded in citations.
        """
    )

# ======================================================
# 4. Insight Task – Synthesis, Gaps & Direct Paper Links
# ======================================================
def create_insight_task(agent: Agent, r_task: Task, c_task: Task) -> Task:
    return Task(
        description="""
        Synthesize an authoritative Executive Academic Research Dossier for: '{query}' directly from the comparative review and findings.
        Do NOT call any retrieval tools — synthesize directly from the comparison findings provided in your context.
        Structure the dossier as:
        # Academic Research Dossier: {query}
        ## 1. Executive Summary
        ## 2. Comparative Analysis & Benchmark Matrix
        ## 3. Critical Research Gaps & Unexplored Hypotheses
        ## 4. High-Impact Future Research Directions
        ## 5. Recommended Baseline Architecture to Build Upon
        ## 6. Analyzed Papers & Direct Read/Download Links
           List each analyzed paper with a clickable markdown link:
           - **[Paper Title](direct_pdf_url)** — Authors (Year). Summary: ...
        Keep formatting clean, professional, and publication-ready.
        """,
        agent=agent,
        context=[r_task, c_task],
        expected_output="""
        A publication-quality Executive Research Dossier in Markdown format with verified clickable direct paper links.
        """
    )

# Singletons for backward compatibility
research_task = create_research_task(research_agent)
ingestion_task = create_ingestion_task(ingestion_agent, research_task)
comparison_task = create_comparison_task(comparison_agent, research_task)
insight_task = create_insight_task(insight_agent, research_task, comparison_task)
