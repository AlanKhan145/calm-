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
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """
    Create LLM instance. Uses OPENROUTER_API_KEY if set, else OPENAI_API_KEY.

    With OpenRouter:
        - Set OPENROUTER_API_KEY
        - model format: "openai/gpt-4o", "anthropic/claude-3.5-sonnet", etc.

    With OpenAI:
        - Set OPENAI_API_KEY
        - model format: "gpt-4o", "gpt-4", etc.

    Args:
        model: Model ID. If None, uses OPENROUTER_DEFAULT_MODEL or OPENAI_DEFAULT_MODEL.
        temperature: Sampling temperature.
        **kwargs: Extra args passed to ChatOpenAI.

    Returns:
        ChatOpenAI instance configured for OpenRouter or OpenAI.
    """
    from langchain_openai import ChatOpenAI

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

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
