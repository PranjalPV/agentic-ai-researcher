import os
import re
from datetime import datetime
from typing import Optional
from crewai import Crew, Process

from agents.agents import (
    create_research_agent,
    create_ingestion_agent,
    create_comparison_agent,
    create_insight_agent,
    research_agent,
    ingestion_agent,
    comparison_agent,
    insight_agent
)

from tasks.tasks import (
    create_research_task,
    create_ingestion_task,
    create_comparison_task,
    create_insight_task,
    research_task,
    ingestion_task,
    comparison_task,
    insight_task
)

from rag.rag_tool import reset_rag_engine

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def build_crew() -> Crew:
    """
    Builds a completely fresh Crew instance with isolated Agent and Task objects.
    Prevents CrewAI 'Executor is already running' concurrency/recycling errors.
    """
    a1 = create_research_agent()
    a2 = create_ingestion_agent()
    a3 = create_comparison_agent()
    a4 = create_insight_agent()

    t1 = create_research_task(a1)
    t2 = create_ingestion_task(a2, t1)
    t3 = create_comparison_task(a3, t1)
    t4 = create_insight_task(a4, t1, t3)

    return Crew(
        agents=[a1, a2, a3, a4],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=True
    )

# Pre-configured Crew instance for direct imports (backwards compatibility)
crew = build_crew()


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

    active_crew = build_crew()
    result = active_crew.kickoff(inputs={"query": query})
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
