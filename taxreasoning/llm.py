"""One chat-completion entry point for every model used in the paper.

All three providers we hit (OpenAI, DeepSeek, and the aggregator that fronts
the Gemini / Claude / Qwen baselines) speak the OpenAI chat-completions
protocol, so they differ only in base URL and credential. Keys are read from
the environment -- never hardcode one in a source file.
"""

from __future__ import annotations

import logging
import os
import random
import time

from openai import OpenAI

logger = logging.getLogger(__name__)

# provider -> (base_url, environment variable holding the key)
PROVIDERS = {
    "openai": (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "proxy": (
        os.environ.get("TAXR_PROXY_BASE_URL", "https://api2.aigcbest.top/v1"),
        "TAXR_PROXY_API_KEY",
    ),
}

# Substrings that route a model name to a provider when --provider is omitted.
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-", "chatgpt")
_DEEPSEEK_PREFIXES = ("deepseek-",)


def infer_provider(model: str) -> str:
    """Guess the provider from a model name.

    Everything that is not obviously OpenAI or DeepSeek (Gemini, Claude, Qwen,
    QwQ) went through the aggregator, so that is the fallback.
    """
    lowered = model.lower()
    if lowered.startswith(_OPENAI_PREFIXES):
        return "openai"
    if lowered.startswith(_DEEPSEEK_PREFIXES):
        return "deepseek"
    return "proxy"


def make_client(model: str, provider: str | None = None) -> OpenAI:
    """Build an OpenAI-protocol client for ``model``."""
    provider = provider or infer_provider(model)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {list(PROVIDERS)}")

    base_url, key_var = PROVIDERS[provider]
    api_key = os.environ.get(key_var)
    if not api_key:
        raise RuntimeError(
            f"model {model!r} routes to provider {provider!r}, which needs the "
            f"{key_var} environment variable. Copy .env.example and fill it in."
        )
    logger.info("model=%s provider=%s base_url=%s", model, provider, base_url or "default")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_retries: int = 5,
    **kwargs,
) -> str:
    """Send one chat completion and return the assistant text.

    Retries on transient API failures with exponential backoff plus jitter. The
    original scripts had no retry, so a single rate-limit response aborted a
    whole run partway through.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise varied types
            last_error = exc
            if attempt == max_retries - 1:
                break
            delay = 2**attempt + random.uniform(0, 1)
            logger.warning(
                "API call failed (attempt %d/%d): %s -- retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"chat completion failed after {max_retries} attempts") from last_error
