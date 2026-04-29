# from crewai.tools import tool
# from rag.rag_tool import rag_tool

# @tool("rag_ingestion_tool")
# def rag_ingestion_tool(pdf_paths: list) -> str:
#     """
#     Add downloaded PDFs into CrewAI's built-in RagTool knowledge base.
#     """
#     for path in pdf_paths:
#         rag_tool.add(
#             data_type="file",
#             path=path
#         )

#     return f"{len(pdf_paths)} PDFs ingested into RAG knowledge base."


from crewai.tools import tool
from rag.rag_tool import rag_tool

@tool("rag_ingestion_tool")
def rag_ingestion_tool(pdf_paths: list) -> str:
    """
    Add downloaded PDFs into the local vector database for healthcare research.
    """
    success_count = 0
    for path in pdf_paths:
        try:
            # PDFSearchTool uses .add() but handles local embeddings better
            rag_tool.add(path)
            success_count += 1
        except Exception as e:
            print(f"Error ingesting {path}: {e}")

    return f"Successfully ingested {success_count} out of {len(pdf_paths)} PDFs into the local knowledge base."