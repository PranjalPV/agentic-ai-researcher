# agents.py
from crewai import Agent

# ---- Tools ----
from config.llm import get_llm
from tools.rag_ingestion_tool import rag_ingestion_tool
from tools.arxiv_tool import arxiv_tool
from tools.Semantic_Scholar_tool import semantic_scholar_tool
from tools.pdf_ingestion_tool import pdf_ingestion_tool
from rag.rag_tool import rag_tool




# ======================================================
# 1️⃣ Research Agent – Finds papers
# ======================================================
research_agent = Agent(
    role="AI Research Agent",
    goal="Find high-quality research papers relevant to the given topic",
    backstory="Expert at academic literature search and filtering",
    tools=[arxiv_tool, semantic_scholar_tool],
    llm=get_llm(), 
    verbose=True
)


# ======================================================
# 2️⃣ Ingestion Agent – RAG over PDFs
# ======================================================
ingestion_agent = Agent(
    role="Ingestion Agent",
    goal="Download and index papers for deep search",
    backstory="Expert in vector search and document retrieval",
    tools=[pdf_ingestion_tool, rag_ingestion_tool],
    llm=get_llm(), 
    verbose=True
)
    

# ======================================================
# 3️⃣ Comparison Agent – Compare papers
# ======================================================
comparison_agent = Agent(
    role="Research Comparison Agent",
    goal="""
    Compare the retrieved papers based on:
    - Methodology
    - Dataset
    - Evaluation metrics
    - Results
    - Strengths & limitations
    """,
    backstory="Expert in systematic literature comparison",
    tools=[rag_tool],
    llm=get_llm(),
    verbose=True
)


# ======================================================
# 4️⃣ Insight Agent – Research gaps & future work
# ======================================================
insight_agent = Agent(
    role="Insight Generation Agent",
    goal="""
    Identify:
    - Research gaps
    - Open challenges
    - Future research directions
    - Most promising approaches
    """,
    backstory="Senior AI researcher generating strategic insights",
    tools=[rag_tool],
    llm=get_llm(),
    verbose=True
)


# ======================================================
# Optional: Export all agents together
# ======================================================
# ALL_AGENTS = {
#     "research": research_agent,
#     "retrieval": retrieval_agent,
#     "comparison": comparison_agent,
#     "insight": insight_agent
# }
