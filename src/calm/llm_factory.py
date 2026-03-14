"""
File: llm_factory.py
Description: LLM factory — creates ChatOpenAI with OpenAI or OpenRouter.
             Uses OPENROUTER_API_KEY if set, else OPENAI_API_KEY.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import os
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o"
OPENAI_DEFAULT_MODEL = "gpt-4o"


def get_llm(
    openrouter_key: str | None = None,
    openai_api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """
    Create LLM instance. Ưu tiên key truyền trực tiếp, sau đó env.

    OpenRouter: get_llm(openrouter_key="sk-or-...", model="openai/gpt-4o-mini")
    OpenAI: get_llm(openai_api_key="sk-...", model="gpt-4o")

    Args:
        openrouter_key: OpenRouter API key (hoặc dùng env OPENROUTER_API_KEY).
        openai_api_key: OpenAI API key (hoặc dùng env OPENAI_API_KEY).
        model: Model ID. OpenRouter: "openai/gpt-4o-mini", OpenAI: "gpt-4o".
        temperature: Sampling temperature.
        **kwargs: Extra args passed to ChatOpenAI.

    Returns:
        ChatOpenAI instance.
    """
    from langchain_openai import ChatOpenAI

    openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    if openrouter_key:
        return ChatOpenAI(
            model=model or OPENROUTER_DEFAULT_MODEL,
            temperature=temperature,
            base_url=OPENROUTER_BASE_URL,
            openai_api_key=openrouter_key,
            **kwargs,
        )
    if openai_key:
        return ChatOpenAI(
            model=model or OPENAI_DEFAULT_MODEL,
            temperature=temperature,
            openai_api_key=openai_key,
            **kwargs,
        )
    raise ValueError(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
    )
