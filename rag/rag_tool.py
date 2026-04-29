import os
from dotenv import load_dotenv
from crewai_tools import RagTool

load_dotenv()

# Force dummy values if they aren't in your .env
# if not os.getenv("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = "not_needed"
# if not os.getenv("CHROMA_HUGGINGFACE_API_KEY"):
#     os.environ["CHROMA_HUGGINGFACE_API_KEY"] = "not_needed"



# IMPORTANT: Remove the "not_needed" dummy keys. 
# If the tool sees 'OPENAI_API_KEY', it often tries to use it.
# If using HuggingFace, it's better to leave the OpenAI key unset.

import os
from crewai_tools import RagTool

# 1. Satisfy the Pydantic validator's hunger for an OpenAI key
# We set this to 'dummy' just to pass the initialization check.

rag_tool = RagTool(
    config={
        "llm": {
            "provider": "groq",
            "config": {
                "model": "llama-3.1-8b-instant",
                "temperature": 0,
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "dir": "db_healthcare_research",
                "collection_name": "healthcare_rag"
            }
        }
    }
)