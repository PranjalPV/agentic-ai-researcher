import os
import sys
import time
import re
from dotenv import load_dotenv
import litellm
from crewai import LLM

# Force UTF-8 encoding across environments to prevent Windows charmap/cp1252 emoji crashes
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Intelligent rate-limit wrapper that auto-sleeps on Groq quota resets
_orig_litellm_completion = litellm.completion

def _safe_litellm_completion(*args, **kwargs):
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            return _orig_litellm_completion(*args, **kwargs)
        except litellm.exceptions.RateLimitError as e:
            if attempt == max_attempts - 1:
                raise e
            err_msg = str(e)
            # Extract suggested wait time from Groq error message (e.g. "Please try again in 15.12s")
            match = re.search(r"try again in ([\d\.]+)s", err_msg)
            wait_time = float(match.group(1)) + 1.5 if match else 20.0
            print(f"\n[GroqRateLimiter] Rolling token limit reached. Waiting {wait_time:.1f}s for quota reset (attempt {attempt+1}/{max_attempts})...")
            time.sleep(wait_time)

litellm.completion = _safe_litellm_completion

load_dotenv()


def get_llm(model_type: str = "primary") -> LLM:
    """
    Factory to return an enterprise-grade LLM instance with fallback support.
    Defaults to Groq's high-speed, tool-reliable qwen/qwen3.8-27b.
    Limits max_tokens to 500 to stay strictly within Groq free-tier 1,000 OTPM limits.
    Can be overridden via environment variables: GROQ_PRIMARY_MODEL, GROQ_MAX_TOKENS.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "500"))

    if groq_api_key and groq_api_key.strip() != "":
        default_model = os.getenv("GROQ_PRIMARY_MODEL") or os.getenv("GROQ_MODEL") or "groq/qwen/qwen3.8-27b"
        return LLM(
            model=default_model,
            api_key=groq_api_key,
            temperature=0.2 if model_type == "primary" else 0.0,
            max_tokens=max_tokens,
            verbose=False
        )

    if openai_api_key and openai_api_key.strip() != "":
        return LLM(
            model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
            api_key=openai_api_key,
            temperature=0.2,
            max_tokens=max_tokens,
            verbose=False
        )

    # Fallback default
    return LLM(
        model="groq/qwen/qwen3.8-27b",
        api_key=groq_api_key or "not_provided",
        temperature=0.2,
        max_tokens=max_tokens,
        verbose=False
    )
