from crewai import Crew

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
    verbose=True,
    max_rpm=2
)
