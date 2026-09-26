import os
import sys
import time
import re
import json
from dotenv import load_dotenv
import litellm
from litellm.utils import ModelResponse, Choices, Message
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


# Intelligent wrapper that removes cache_breakpoint, handles Groq tool calls, and auto-sleeps on quota resets
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
    if "tools" not in kwargs:
        kwargs.pop("tool_choice", None)

    current_model = kwargs.get("model", "")
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            return _orig_litellm_completion(*args, **kwargs)
        except litellm.exceptions.RateLimitError as e:
            if attempt == max_attempts - 1:
                raise e
            err_msg = str(e)

            match = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_msg)
            if match:
                mins = float(match.group(1)) if match.group(1) else 0.0
                secs = float(match.group(2))
                wait_time = mins * 60 + secs + 1.5
            else:
                wait_time = 12.0

            if wait_time > 45.0:
                # Quota exceeded for this specific model, attempt dynamic model fallback
                if "120b" in current_model:
                    print(f"\n[GroqFallback] Switching from {current_model} to groq/openai/gpt-oss-20b due to quota wait: {wait_time:.0f}s")
                    kwargs["model"] = "groq/openai/gpt-oss-20b"
                    continue
                elif "qwen" in current_model:
                    print(f"\n[GroqFallback] Switching from {current_model} to groq/openai/gpt-oss-120b due to quota wait: {wait_time:.0f}s")
                    kwargs["model"] = "groq/openai/gpt-oss-120b"
                    continue
                raise RuntimeError(
                    f"Groq token quota reached (requested wait: {wait_time:.0f}s). "
                    f"Please try again in a few moments or provide a key with higher tier quota."
                )

            print(f"\n[GroqRateLimiter] Rate limit on '{kwargs.get('model', current_model)}'. Waiting {wait_time:.1f}s (attempt {attempt+1}/{max_attempts})...")
            time.sleep(wait_time)
            continue

        except litellm.exceptions.BadRequestError as e:
            err_str = str(e)
            if "Tool choice is none, but model called a tool" in err_str:
                # Intercept Groq function calling token rejection and convert to standard CrewAI ReAct text
                match = re.search(r'"failed_generation"\s*:\s*("(?:\\.|[^"\\])*")', err_str)
                if match:
                    try:
                        raw_fg = json.loads(match.group(1))
                        fg_obj = json.loads(raw_fg) if isinstance(raw_fg, str) else raw_fg
                        tool_name = fg_obj.get("name", "")
                        tool_args = fg_obj.get("arguments", {})
                        tool_args_str = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)

                        react_text = f"Action: {tool_name}\nAction Input: {tool_args_str}"
                        msg = Message(content=react_text, role="assistant")
                        choice = Choices(finish_reason="stop", index=0, message=msg)
                        return ModelResponse(
                            id=f"recov_{int(time.time()*1000)}",
                            choices=[choice],
                            model=kwargs.get("model", current_model)
                        )
                    except Exception as parse_err:
                        print(f"[ToolInterceptor] Error parsing failed_generation: {parse_err}")
            raise e

litellm.completion = _safe_litellm_completion

load_dotenv()


def get_llm(model_type: str = "primary") -> LLM:
    """
    Factory to return an enterprise-grade LLM instance with fallback support.
    Defaults to Groq's high-speed openai/gpt-oss-120b with automatic token recovery.
    Allocates generous token limits (2800 for synthesis) so full 6-section dossiers complete cleanly.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if model_type == "synthesis":
        max_tokens = int(os.getenv("GROQ_MAX_TOKENS_SYNTHESIS", "2800"))
    else:
        max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "1200"))

    if groq_api_key and groq_api_key.strip() != "":
        default_model = os.getenv("GROQ_PRIMARY_MODEL") or os.getenv("GROQ_MODEL") or "groq/openai/gpt-oss-120b"
        return LLM(
            model=default_model,
            api_key=groq_api_key,
            temperature=0.2 if model_type == "primary" else 0.1,
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
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key or "not_provided",
        temperature=0.2,
        max_tokens=max_tokens,
        verbose=False
    )

