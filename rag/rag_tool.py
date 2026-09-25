from typing import Optional
from crewai.tools import tool
from rag.hybrid_rag import HybridRAG

# Global RAG engine instance (dynamic session initialization)
_RAG_ENGINE: Optional[HybridRAG] = None


def get_rag_engine(collection_name: str = "academic_research") -> HybridRAG:
    """Singleton getter for the persistent hybrid RAG engine."""
    global _RAG_ENGINE
    if _RAG_ENGINE is None or _RAG_ENGINE.collection_name != collection_name:
        _RAG_ENGINE = HybridRAG(collection_name=collection_name)
    return _RAG_ENGINE


def reset_rag_engine(collection_name: str = "academic_research") -> HybridRAG:
    """Explicitly reset the active RAG engine for a fresh research session."""
    global _RAG_ENGINE
    _RAG_ENGINE = HybridRAG(collection_name=collection_name)
    return _RAG_ENGINE


@tool("academic_hybrid_rag_tool")
def rag_tool(query: str) -> str:
    """
    Search the ingested academic research papers using Hybrid Search (Dense Vector + BM25 Lexical).
    Returns grounded excerpts with verified page numbers and source titles for citation.

    Input:
    - query: Specific technical research question, e.g. 'What datasets were used in the evaluation?' or 'What are the main methodology limitations?'

    Output:
    - Grounded textual excerpts from the papers with [Source: ... | Page: ...] citations.
    """
    engine = get_rag_engine()
    results = engine.search(query=query, top_k=3)
    return engine.format_citation_context(results)