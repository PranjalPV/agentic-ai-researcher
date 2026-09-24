from typing import List, Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Structured academic paper metadata discovered by search agents."""
    title: str = Field(description="Title of the research paper")
    authors: List[str] = Field(default_factory=list, description="List of paper authors")
    year: Optional[int] = Field(default=None, description="Publication year")
    abstract: str = Field(description="Summary abstract of the paper")
    pdf_url: str = Field(description="Direct URL to open-access PDF")
    source: str = Field(default="arxiv", description="Repository source (arxiv, semantic_scholar)")
    citation_count: Optional[int] = Field(default=0, description="Approximate citation count if available")


class PaperComparisonItem(BaseModel):
    """Comparative dimensions extracted for a single paper."""
    title: str = Field(description="Title of the paper")
    methodology: str = Field(description="Core algorithm, architecture, or theoretical framework")
    datasets: List[str] = Field(default_factory=list, description="Benchmarks or datasets evaluated on")
    metrics_and_results: str = Field(description="Key metrics (e.g. Accuracy, F1, BLEU, Latency) and results")
    strengths: List[str] = Field(default_factory=list, description="Key architectural or experimental strengths")
    limitations: List[str] = Field(default_factory=list, description="Identified limitations, assumptions, or failure cases")


class ComparativeAnalysis(BaseModel):
    """Structured matrix comparing multiple papers."""
    papers: List[PaperComparisonItem] = Field(description="List of analyzed papers with standardized dimensions")
    methodology_matrix_summary: str = Field(description="High-level synthesis comparing methodological paradigms")
    common_benchmarks: List[str] = Field(default_factory=list, description="Standard benchmark datasets common across studies")


class StrategicInsights(BaseModel):
    """Critical evaluation identifying research gaps and future avenues."""
    research_gaps: List[str] = Field(description="Unresolved problems or missing evaluations in current literature")
    open_challenges: List[str] = Field(description="Technical or scaling bottlenecks identified across papers")
    promising_directions: List[str] = Field(description="High-potential research opportunities for future exploration")
    recommended_baseline: str = Field(description="Recommended state-of-the-art approach to build upon")


class ExecutiveResearchReport(BaseModel):
    """End-to-end research dossier."""
    topic: str = Field(description="Original research topic query")
    executive_summary: str = Field(description="Executive summary of literature review findings")
    comparative_analysis: ComparativeAnalysis = Field(description="Systematic comparison across papers")
    strategic_insights: StrategicInsights = Field(description="Research gaps and prospective directions")
    references: List[PaperMetadata] = Field(default_factory=list, description="Cited academic literature sources")
