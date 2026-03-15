"""
Mô-đun factory tạo LLM — ChatOpenRouter hoặc ChatOpenAI.

Ưu tiên OPENROUTER_API_KEY (OpenRouter), nếu không có thì dùng OPENAI_API_KEY.
Ném ValueError nếu không tìm thấy API key nào. Nên gọi load_env() trước khi
gọi get_llm() nếu dùng file .env.
"""

from __future__ import annotations

import os
from typing import Any

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
    Tạo instance LLM (ChatOpenRouter hoặc ChatOpenAI).

    Ưu tiên key truyền tham số, sau đó biến môi trường. Ném ValueError nếu
    không có OPENROUTER_API_KEY cũng không có OPENAI_API_KEY.

    Tham số:
        openrouter_key: API key OpenRouter (hoặc env OPENROUTER_API_KEY).
        openai_api_key: API key OpenAI (hoặc env OPENAI_API_KEY).
        model: ID model (OpenRouter: "openai/gpt-4o-mini", OpenAI: "gpt-4o").
        temperature: Nhiệt độ sampling.
        **kwargs: Tham số bổ sung cho ChatOpenRouter/ChatOpenAI.

    Trả về:
        Instance chat model (ChatOpenRouter hoặc ChatOpenAI).
    """
    openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    # Loại bỏ params không dùng bởi API
    safe_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("openrouter_key", "openai_api_key")
    }

    if openrouter_key:
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=model or OPENROUTER_DEFAULT_MODEL,
            temperature=temperature,
            api_key=openrouter_key,
            **safe_kwargs,
        )
    if openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or OPENAI_DEFAULT_MODEL,
            temperature=temperature,
            openai_api_key=openai_key,
            **safe_kwargs,
        )
    raise ValueError(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
    )
