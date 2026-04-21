"""Multi-provider LLM router with task-aware routing and automatic failover.

Priority order is determined by PRIMARY_MODEL + FALLBACK_MODELS env vars.
Billing/quota errors advance to the next provider; rate-limit errors
trigger exponential back-off on the same provider.

Supported providers
-------------------
Standard (via init_chat_model):
  anthropic, openai, google, mistralai, …

OpenAI-compatible (custom base URL):
  kimi / moonshot  — api.moonshot.cn
  deepseek         — api.deepseek.com
  together         — api.together.xyz
  groq             — api.groq.com

Add more by extending _COMPAT_PROVIDERS below.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from .runtime.router import select_model_for_step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

# OpenAI-compatible providers: (base_url, api_key_env_var)
_COMPAT_PROVIDERS: dict[str, tuple[str, str]] = {
    "kimi":     ("https://api.moonshot.cn/v1",     "KIMI_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1",     "KIMI_API_KEY"),
    "deepseek": ("https://api.deepseek.com",        "DEEPSEEK_API_KEY"),
    "together": ("https://api.together.xyz/v1",     "TOGETHER_API_KEY"),
    "groq":     ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}

# Error signatures that indicate the account/key is exhausted → switch provider
_BILLING_ERRORS: tuple[str, ...] = (
    "credit balance",
    "insufficient_balance",
    "insufficient funds",
    "account balance",
    "billing",
    "payment",
    "quota exceeded",
    "402",
)

# Error signatures that indicate temporary overload → back-off, same provider
_RATE_ERRORS: tuple[str, ...] = (
    "429",
    "rate limit",
    "too many requests",
    "overloaded",
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_provider_model(model_spec: str) -> BaseChatModel:
    """Load any model from a ``provider/model-name`` spec.

    Works for all standard LangChain providers and the OpenAI-compatible
    ones registered in ``_COMPAT_PROVIDERS``.
    """
    provider, model_name = model_spec.split("/", 1)

    if provider in _COMPAT_PROVIDERS:
        from langchain_openai import ChatOpenAI
        base_url, key_env = _COMPAT_PROVIDERS[provider]
        api_key = os.environ.get(key_env)
        if not api_key:
            raise ValueError(f"{key_env} not set — required for provider '{provider}'")
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)

    from langchain.chat_models import init_chat_model
    return init_chat_model(model_name, model_provider=provider)


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

def get_fallback_chain() -> list[str]:
    """Return the ordered list of model specs from env vars.

    PRIMARY_MODEL   — the first model to try (defaults to Claude Sonnet)
    FALLBACK_MODELS — comma-separated list of fallbacks
                      e.g. ``kimi/kimi-k2.5,openai/gpt-4o-mini``
    """
    primary = os.environ.get("PRIMARY_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    raw = os.environ.get("FALLBACK_MODELS", "")
    chain = [primary]
    if raw:
        chain += [m.strip() for m in raw.split(",") if m.strip()]
    return chain


def build_model_chain(primary_override: Optional[str] = None) -> list[str]:
    """Return a de-duplicated primary+fallback chain."""
    base_chain = get_fallback_chain()
    if primary_override:
        chain = [primary_override, *base_chain]
    else:
        chain = list(base_chain)

    deduped: list[str] = []
    seen: set[str] = set()
    for spec in chain:
        if not spec or spec in seen:
            continue
        deduped.append(spec)
        seen.add(spec)
    return deduped

# ---------------------------------------------------------------------------
# Core invoke
# ---------------------------------------------------------------------------

async def invoke_with_fallback(
    primary_model: BaseChatModel,
    messages: List[BaseMessage],
    tools: List,
    max_rl_retries: int = 4,
    primary_spec: Optional[str] = None,
) -> Any:
    """Invoke *primary_model*; fall back to FALLBACK_MODELS on billing errors.

    Args:
        primary_model: Already-configured primary model (with tools bound).
        messages:      Full message list including system prompt.
        tools:         Tool list — re-bound when loading fallback models.
        max_rl_retries: Max back-off retries per provider on rate-limit errors.

    Returns:
        The first successful AIMessage response.

    Raises:
        The last exception if every provider in the chain fails.
    """
    chain = build_model_chain(primary_spec)
    fallback_specs = chain[1:]

    # Queue entries: (pre-built model | None, spec | None)
    queue: list[tuple[Optional[BaseChatModel], Optional[str]]] = (
        [(primary_model, None)]
        + [(None, spec) for spec in fallback_specs]
    )

    last_exc: Optional[Exception] = None

    for model_obj, spec in queue:
        # Load fallback models lazily
        if model_obj is None:
            try:
                base = load_provider_model(spec)
                model_obj = base.bind_tools(tools) if tools else base
                logger.info("Switching to fallback provider: %s", spec)
            except Exception as e:
                logger.warning("Cannot load fallback '%s': %s", spec, e)
                last_exc = e
                continue

        # Try this provider with rate-limit back-off
        rl_attempt = 0
        while True:
            try:
                result = await model_obj.ainvoke(messages)
                if spec:
                    logger.info("Request served by fallback: %s", spec)
                return result

            except Exception as e:
                err = str(e).lower()

                if any(kw in err for kw in _BILLING_ERRORS):
                    logger.warning(
                        "Billing/quota error on %s — advancing to next provider",
                        spec or "primary",
                    )
                    last_exc = e
                    break  # next provider

                elif any(kw in err for kw in _RATE_ERRORS):
                    rl_attempt += 1
                    if rl_attempt <= max_rl_retries:
                        wait = 2 ** rl_attempt
                        logger.warning(
                            "Rate limit on %s — retry %d/%d in %ds",
                            spec or "primary", rl_attempt, max_rl_retries, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(
                            "Rate limit retries exhausted on %s — trying next provider",
                            spec or "primary",
                        )
                        last_exc = e
                        break  # next provider

                else:
                    raise  # unexpected error — propagate immediately

    raise last_exc or RuntimeError("All providers in the fallback chain failed")
