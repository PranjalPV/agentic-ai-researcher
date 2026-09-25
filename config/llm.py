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

# Automatically drop parameters unsupported by Groq (like prompt caching headers)
litellm.drop_params = True
litellm.num_retries = 5
litellm.request_timeout = 60

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _clean_messages_for_groq(messages):
    """
    Strips unsupported prompt-caching properties like 'cache_breakpoint'
    and 'cache_control' that newer CrewAI/LiteLLM versions inject into messages.
    """
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            # Clean message dictionary
            clean_msg = {k: v for k, v in msg.items() if k not in ("cache_breakpoint", "cache_control")}
            cleaned.append(clean_msg)
        elif hasattr(msg, "model_dump"):
            d = msg.model_dump()
            d.pop("cache_breakpoint", None)
            d.pop("cache_control", None)
            cleaned.append(d)
        else:
            cleaned.append(msg)
    return cleaned


# Intelligent wrapper that removes cache_breakpoint and auto-sleeps on Groq quota resets
_orig_litellm_completion = litellm.completion

def _safe_litellm_completion(*args, **kwargs):
    # 1. Clean message objects of unsupported Groq parameters
    if "messages" in kwargs:
        kwargs["messages"] = _clean_messages_for_groq(kwargs["messages"])
    elif len(args) > 1 and isinstance(args[1], list):
        args_list = list(args)
        args_list[1] = _clean_messages_for_groq(args_list[1])
        args = tuple(args_list)

    # 2. Clean top-level kwargs
    kwargs.pop("cache_breakpoint", None)
    kwargs.pop("cache_control", None)

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            return _orig_litellm_completion(*args, **kwargs)
        except litellm.exceptions.RateLimitError as e:
            if attempt == max_attempts - 1:
                raise e
            err_msg = str(e)
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
