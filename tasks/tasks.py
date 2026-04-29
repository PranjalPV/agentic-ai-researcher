# tasks.py
from crewai import Task

# ---- Agents ----
from agents.agents import (
    research_agent,
    ingestion_agent,
    comparison_agent,
    insight_agent
)

# ======================================================
# 1️⃣ Research Task – Find papers
# ======================================================
research_task = Task(
    description="Search and return the most relevant research papers for the user query.",
    agent=research_agent,
    expected_output="List of papers with metadata and PDF URLs"
)


# ======================================================
# 2️⃣ Ingestion Task – Download & ingest PDFs
# ======================================================
ingestion_task = Task(
    description="Download the research papers and ingest them into the RAG knowledge base.",
    agent=ingestion_agent,
    expected_output="Confirmation that papers are ingested into vector store"
)


# ======================================================
# 3️⃣ Comparison Task – Compare papers
# ======================================================
comparison_task = Task(
    description="Compare the retrieved research papers in a structured manner.",
    agent=comparison_agent,
    expected_output="""
    Comparative analysis including:
    - Methodology
    - Dataset
    - Evaluation metrics
    - Results
    - Strengths & limitations
    """
)


# ======================================================
# 4️⃣ Insight Task – Gaps & future directions
# ======================================================
insight_task = Task(
    description="Generate insights, research gaps, and future research directions.",
    agent=insight_agent,
    expected_output="Key insights, research gaps, and future directions"
)


# ======================================================
# Optional: Export all tasks together
# ======================================================
# ALL_TASKS = [
#     research_task,
#     ingestion_task,
#     comparison_task,
#     insight_task
# ]
