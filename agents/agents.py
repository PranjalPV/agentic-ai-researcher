from crewai import Agent
from config.llm import get_llm
from tools.arxiv_tool import arxiv_tool
from tools.Semantic_Scholar_tool import semantic_scholar_tool
from tools.pdf_ingestion_tool import pdf_ingestion_tool
from tools.rag_ingestion_tool import rag_ingestion_tool
from rag.rag_tool import rag_tool

# ======================================================
# 1. Lead AI Literature Scout
# ======================================================
def create_research_agent() -> Agent:
    return Agent(
        role="Lead AI Literature Scout",
        goal="Discover top-tier, relevant academic papers from arXiv for: {query}. Filter for technical depth and provide direct PDF download links.",
        backstory=(
            "You are a senior bibliometric scientist and research scout. You know how to construct "
            "precise keyword queries, search preprint archives, analyze abstracts, "
            "and select relevant papers with verified open-access PDF download links."
        ),
        tools=[arxiv_tool, semantic_scholar_tool],
        llm=get_llm("primary"),
        verbose=True,
        max_iter=3,
        allow_delegation=False
    )

# ======================================================
# 2. Academic Document Ingestion & RAG Engineer
# ======================================================
def create_ingestion_agent() -> Agent:
    return Agent(
        role="Academic Document Ingestion & RAG Engineer",
        goal="Index the discovered academic literature into the Hybrid RAG engine (Dense ChromaDB + BM25 Lexical) preserving paper titles, abstracts, methodology, and direct PDF links.",
        backstory=(
            "You are an expert in neural document processing and knowledge graph ingestion. "
            "You take discovered academic papers, extract key findings, and index them into the "
            "hybrid vector store for fast, citation-grounded retrieval with direct download links."
        ),
        tools=[rag_ingestion_tool],
        llm=get_llm("primary"),
        verbose=True,
        max_iter=3,
        allow_delegation=False
    )

# ======================================================
# 3. Systematic Reviewer & Comparative Analyst
# ======================================================
def create_comparison_agent() -> Agent:
    return Agent(
        role="Systematic Reviewer & Comparative Analyst",
        goal="""
        Query the indexed papers via the hybrid RAG tool and construct a concise, structured comparative matrix covering:
        - Core Methodologies & Architectural Innovations
        - Datasets & Experimental Setups
        - Quantitative Benchmark Metrics & Results
        - Algorithmic Strengths and Critical Limitations
        Ground insights in retrieved text with paper titles and direct links.
        """,
        backstory=(
            "You are a veteran meta-reviewer for premier AI conferences (NeurIPS, ICML, ICLR). "
            "You dissect academic claims, compare empirical trade-offs, and ground your synthesis "
            "in verified citations extracted through vector retrieval."
        ),
        tools=[rag_tool],
        llm=get_llm("primary"),
        verbose=True,
        max_iter=3,
        allow_delegation=False
    )

# ======================================================
# 4. Principal Research Strategist
# ======================================================
def create_insight_agent() -> Agent:
    return Agent(
        role="Principal Research Strategist",
        goal="""
        Critically analyze the literature to uncover:
        - Key Research Gaps & Unexplored Hypotheses
        - Open Engineering & Scaling Bottlenecks
        - Actionable, High-Impact Future Research Directions
        - The Most Promising Baseline Architecture to Build Upon
        Synthesize the findings into an executive research dossier with direct paper read and download links.
        """,
        backstory=(
            "You are a Distinguished AI Research Director. You look beyond incremental benchmark improvements "
            "to spot foundational theoretical gaps, reproducibility bottlenecks, and non-obvious breakthrough directions. "
            "You synthesize directly from comparative findings without redundant retrieval passes."
        ),
        tools=[],
        llm=get_llm("synthesis"),
        verbose=True,
        max_iter=3,
        allow_delegation=False
    )

# Singletons for backward compatibility
research_agent = create_research_agent()
ingestion_agent = create_ingestion_agent()
comparison_agent = create_comparison_agent()
insight_agent = create_insight_agent()
