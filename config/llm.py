from crewai import LLM
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        verbose=False
    )
