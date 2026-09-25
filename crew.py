import os
import re
from datetime import datetime
from typing import Optional
from crewai import Crew, Process

from agents.agents import (
    research_agent,
    ingestion_agent,
    comparison_agent,
    insight_agent
)

from tasks.tasks import (
    research_task,
    ingestion_task,
    comparison_task,
    insight_task
)

from rag.rag_tool import reset_rag_engine

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Pre-configured Crew instance for direct imports
# max_rpm=2 enforces a 30s pause between requests, strictly adhering to Groq free-tier 1000 OTPM limits
crew = Crew(
    agents=[
        research_agent,
        ingestion_agent,
        comparison_agent,
        insight_agent
    ],
    tasks=[
        research_task,
        ingestion_task,
        comparison_task,
        insight_task
    ],
    process=Process.sequential,
    verbose=True,
    max_rpm=2
)


def run_research(query: str, save_report: bool = True, session_id: Optional[str] = None) -> str:
    """
    Executes the autonomous multi-agent research pipeline for a topic.
    Manages session isolation, execution lifecycle, and report export.
    """
    clean_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", query.strip().lower())[:30]
    session_name = session_id or f"session_{clean_slug}_{int(datetime.now().timestamp())}"

    # Initialize isolated vector session
    reset_rag_engine(collection_name=session_name)

    print(f"\n[AgenticResearcher] Launching research crew for topic: '{query}'")
    print(f"[AgenticResearcher] Session ID: {session_name}\n")

    result = crew.kickoff(inputs={"query": query})
    result_text = str(result)

    if save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"research_report_{clean_slug}_{timestamp}.md"
        report_path = os.path.join(REPORTS_DIR, report_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Academic Research Dossier: {query}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Session:** `{session_name}`  \n\n")
            f.write("---\n\n")
            f.write(result_text)

        print(f"\n[AgenticResearcher] Executive Report saved successfully to: {report_path}")

    return result_text
