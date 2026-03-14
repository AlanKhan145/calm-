"""
File: __init__.py
Description: Package root for CALM — Adaptive Multimodal Wildfire Monitoring.
Author: CALM Team
Created: 2026-03-13
"""

from calm.llm_factory import get_llm

__version__ = "0.1.0"
__all__ = ["__version__", "get_llm"]
