import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()


def get_llm(model_type: str = "primary") -> LLM:
    """
    Factory to return an enterprise-grade LLM instance with fallback support.
    Defaults to Groq's high-speed LLaMA-3.3-70B for synthesis and LLaMA-3.1-8B for speed.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if groq_api_key and groq_api_key.strip() != "":
        model_name = "groq/llama-3.3-70b-versatile" if model_type == "primary" else "groq/llama-3.1-8b-instant"
        return LLM(
            model=model_name,
            api_key=groq_api_key,
            temperature=0.2 if model_type == "primary" else 0.0,
            max_tokens=4096,
            verbose=False
        )

    if openai_api_key and openai_api_key.strip() != "":
        return LLM(
            model="openai/gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.2,
            max_tokens=4096,
            verbose=False
        )

    # Fallback to local / standard groq
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_api_key or "not_provided",
        temperature=0.2,
        verbose=False
    )
